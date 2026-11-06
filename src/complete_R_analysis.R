# ==============================================================
# COMPLETE R ANALYSIS PIPELINE
# Project: Silent Failure Detection in Distributed Systems
# Dataset: Google Borg Cluster Trace
#
# Performs:
# 1. Data loading
# 2. Data preprocessing
# 3. Data validation
# 4. Feature engineering
# 5. Exploratory Data Analysis
# 6. Visualization
# 7. PDF report generation
# ==============================================================


# ==============================================================
# 0. INSTALL / LOAD REQUIRED PACKAGES
# ==============================================================

required_packages <- c(
  "readr",
  "dplyr",
  "ggplot2",
  "tidyr",
  "corrplot",
  "scales"
)

for (pkg in required_packages) {

  if (!require(pkg, character.only = TRUE)) {

    install.packages(pkg)

    library(pkg, character.only = TRUE)

  }

}


# ==============================================================
# 1. PROJECT PATHS
# ==============================================================

cat("\n==============================================\n")
cat(" SILENT FAILURE DETECTION - R ANALYSIS\n")
cat("==============================================\n\n")


# Change this ONLY if your CSV is somewhere else

dataset_path <- "data/raw/borg_traces_data.csv"


# Output directories

output_dir <- "outputs/R_analysis"

plot_dir <- file.path(output_dir, "plots")


if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

if (!dir.exists(plot_dir)) {
  dir.create(plot_dir, recursive = TRUE)
}


# ==============================================================
# 2. LOAD DATASET
# ==============================================================

cat("STEP 1: Loading Google Borg dataset...\n")


if (!file.exists(dataset_path)) {

  stop(
    paste(
      "\nDataset not found at:",
      dataset_path,
      "\n\nPut borg_traces_data.csv inside:",
      "data/raw/"
    )
  )

}


borg_data <- read_csv(
  dataset_path,
  show_col_types = FALSE
)


cat("Dataset loaded successfully!\n\n")


cat("Rows    :", nrow(borg_data), "\n")
cat("Columns :", ncol(borg_data), "\n\n")


# ==============================================================
# 3. CLEAN COLUMN NAMES
# ==============================================================

cat("STEP 2: Cleaning column names...\n")


# Rename unnamed first column

if ("...1" %in% colnames(borg_data)) {

  borg_data <- borg_data %>%
    rename(record_id = `...1`)

}


# Remove accidental duplicate columns

borg_data <- borg_data[, !duplicated(colnames(borg_data))]


cat("Column names cleaned.\n\n")


# ==============================================================
# 4. BASIC DATA VALIDATION
# ==============================================================

cat("STEP 3: Validating dataset...\n")


initial_rows <- nrow(borg_data)


# Count duplicate rows

duplicate_count <- sum(duplicated(borg_data))


cat("Duplicate rows:", duplicate_count, "\n")


# Remove duplicates

borg_data <- borg_data %>%
  distinct()


after_duplicate_rows <- nrow(borg_data)


cat(
  "Rows removed:",
  initial_rows - after_duplicate_rows,
  "\n\n"
)


# ==============================================================
# 5. HANDLE MISSING VALUES
# ==============================================================

cat("STEP 4: Checking missing values...\n")


missing_summary <- data.frame(
  column = names(borg_data),
  missing_count = sapply(
    borg_data,
    function(x) sum(is.na(x))
  )
)


missing_summary <- missing_summary %>%
  arrange(desc(missing_count))


print(missing_summary)


write_csv(
  missing_summary,
  file.path(output_dir, "missing_values.csv")
)


# ==============================================================
# 6. CONVERT NUMERIC VARIABLES
# ==============================================================

cat("\nSTEP 5: Converting numeric variables...\n")


numeric_columns <- c(

  "time",
  "instance_events_type",
  "collection_id",
  "scheduling_class",
  "collection_type",
  "priority",
  "alloc_collection_id",
  "instance_index",
  "machine_id",
  "vertical_scaling",
  "scheduler",
  "start_time",
  "end_time",
  "assigned_memory",
  "page_cache_memory",
  "cycles_per_instruction",
  "memory_accesses_per_instruction",
  "sample_rate"

)


# Convert only columns that actually exist

numeric_columns <- intersect(
  numeric_columns,
  names(borg_data)
)


borg_data[numeric_columns] <-
  lapply(
    borg_data[numeric_columns],
    function(x) suppressWarnings(as.numeric(x))
  )


cat(
  "Converted",
  length(numeric_columns),
  "numeric variables.\n\n"
)


# ==============================================================
# 7. HANDLE INFINITE VALUES
# ==============================================================

cat("STEP 6: Checking infinite values...\n")


for (i in seq_along(borg_data)) {

  if (is.numeric(borg_data[[i]])) {

    borg_data[[i]][
      is.infinite(borg_data[[i]])
    ] <- NA

  }

}


cat("Infinite values handled.\n\n")


# ==============================================================
# 8. FEATURE ENGINEERING
# ==============================================================

cat("STEP 7: Creating analytical features...\n")


# --------------------------------------------------------------
# Task duration
# --------------------------------------------------------------

if (
  all(c("start_time", "end_time") %in% names(borg_data))
) {

  borg_data <- borg_data %>%

    mutate(

      task_duration =
        ifelse(
          !is.na(start_time) &
            !is.na(end_time) &
            end_time >= start_time,

          end_time - start_time,

          NA

        )

    )

}


# --------------------------------------------------------------
# CPU efficiency
# --------------------------------------------------------------

if (
  "cycles_per_instruction" %in% names(borg_data)
) {

  borg_data <- borg_data %>%

    mutate(

      cpu_efficiency =
        ifelse(
          cycles_per_instruction > 0,

          1 / cycles_per_instruction,

          NA

        )

    )

}


# --------------------------------------------------------------
# Memory pressure
# --------------------------------------------------------------

if (
  all(
    c(
      "assigned_memory",
      "page_cache_memory"
    ) %in% names(borg_data)
  )
) {

  borg_data <- borg_data %>%

    mutate(

      memory_pressure =
        page_cache_memory /
        pmax(assigned_memory, 1)

    )

}


# --------------------------------------------------------------
# CPU / memory inefficiency indicator
# --------------------------------------------------------------

if (
  all(
    c(
      "cycles_per_instruction",
      "memory_accesses_per_instruction"
    ) %in% names(borg_data)
  )
) {

  borg_data <- borg_data %>%

    mutate(

      resource_inefficiency =
        cycles_per_instruction *
        memory_accesses_per_instruction

    )

}


# ==============================================================
# 9. CREATE EVENT LABEL
# ==============================================================

if ("instance_events_type" %in% names(borg_data)) {

  borg_data <- borg_data %>%

    mutate(

      event_label =
        factor(instance_events_type)

    )

}


# ==============================================================
# 10. CREATE FAILURE LABEL IF ALREADY AVAILABLE
# ==============================================================

# IMPORTANT:
# We do NOT invent failure labels.
# If the dataset/preprocessing already contains "failed",
# we use it.

if ("failed" %in% names(borg_data)) {

  borg_data <- borg_data %>%

    mutate(
      failed = as.factor(failed)
    )

}


# ==============================================================
# 11. REMOVE EXTREME INVALID VALUES
# ==============================================================

cat("STEP 8: Removing invalid observations...\n")


# Assigned memory cannot be negative

if ("assigned_memory" %in% names(borg_data)) {

  borg_data$assigned_memory[
    borg_data$assigned_memory < 0
  ] <- NA

}


# Priority cannot be negative

if ("priority" %in% names(borg_data)) {

  borg_data$priority[
    borg_data$priority < 0
  ] <- NA

}


# ==============================================================
# 12. SAVE CLEAN DATASET
# ==============================================================

cleaned_path <- file.path(
  output_dir,
  "cleaned_borg_data.csv"
)


write_csv(
  borg_data,
  cleaned_path
)


cat(
  "Clean dataset saved to:",
  cleaned_path,
  "\n\n"
)


# ==============================================================
# 13. PREPROCESSING REPORT
# ==============================================================

report_file <- file.path(
  output_dir,
  "preprocessing_report.txt"
)


sink(report_file)


cat("=============================================\n")
cat(" PREPROCESSING REPORT\n")
cat(" Google Borg Cluster Dataset\n")
cat("=============================================\n\n")


cat("Original rows:", initial_rows, "\n")

cat(
  "Rows after duplicate removal:",
  nrow(borg_data),
  "\n"
)

cat(
  "Original columns:",
  ncol(borg_data),
  "\n\n"
)


cat("Duplicate rows removed:", duplicate_count, "\n\n")


cat("Missing Value Summary\n")
cat("---------------------------------------------\n")

print(missing_summary)


cat("\n\nFinal Columns\n")
cat("---------------------------------------------\n")

print(names(borg_data))


cat("\n\nFinal Dataset Dimensions\n")
cat("---------------------------------------------\n")

cat(
  "Rows:",
  nrow(borg_data),
  "\n"
)

cat(
  "Columns:",
  ncol(borg_data),
  "\n"
)


sink()


# ==============================================================
# 14. DATASET OVERVIEW GRAPH
# ==============================================================

cat("STEP 9: Creating visualizations...\n")


p1 <- ggplot(

  data.frame(
    metric = c(
      "Rows",
      "Columns"
    ),
    value = c(
      nrow(borg_data),
      ncol(borg_data)
    )
  ),

  aes(
    x = metric,
    y = value
  )

) +

  geom_col(
    fill = "steelblue"
  ) +

  labs(

    title =
      "Google Borg Dataset Overview",

    x = NULL,

    y = "Count"

  ) +

  theme_minimal()


ggsave(

  file.path(
    plot_dir,
    "01_dataset_overview.png"
  ),

  p1,

  width = 8,
  height = 5

)


# ==============================================================
# 15. MISSING VALUE GRAPH
# ==============================================================


missing_plot_data <- missing_summary %>%

  filter(
    missing_count > 0
  ) %>%

  arrange(
    desc(missing_count)
  )


if (nrow(missing_plot_data) > 0) {

  p2 <- ggplot(

    missing_plot_data,

    aes(
      x = reorder(
        column,
        missing_count
      ),
      y = missing_count
    )

  ) +

    geom_col(
      fill = "firebrick"
    ) +

    coord_flip() +

    labs(

      title =
        "Missing Values by Feature",

      x = "Feature",

      y = "Missing Values"

    ) +

    theme_minimal()


  ggsave(

    file.path(
      plot_dir,
      "02_missing_values.png"
    ),

    p2,

    width = 9,
    height = 6

  )

}


# ==============================================================
# 16. EVENT DISTRIBUTION
# ==============================================================

if ("instance_events_type" %in% names(borg_data)) {

  p3 <- ggplot(

    borg_data,

    aes(
      x = factor(instance_events_type)
    )

  ) +

    geom_bar(
      fill = "darkorange"
    ) +

    labs(

      title =
        "Instance Event Distribution",

      x =
        "Instance Event Type",

      y =
        "Number of Records"

    ) +

    theme_minimal()


  ggsave(

    file.path(
      plot_dir,
      "03_event_distribution.png"
    ),

    p3,

    width = 9,
    height = 6

  )

}


# ==============================================================
# 17. FAILURE DISTRIBUTION
# ==============================================================

if ("failed" %in% names(borg_data)) {

  p4 <- ggplot(

    borg_data,

    aes(
      x = failed
    )

  ) +

    geom_bar(
      fill = "tomato"
    ) +

    labs(

      title =
        "Failure Distribution",

      x =
        "Failure Status",

      y =
        "Number of Records"

    ) +

    theme_minimal()


  ggsave(

    file.path(
      plot_dir,
      "04_failure_distribution.png"
    ),

    p4,

    width = 8,
    height = 5

  )

}


# ==============================================================
# 18. CPU DISTRIBUTION
# ==============================================================

if ("cycles_per_instruction" %in% names(borg_data)) {

  p5 <- ggplot(

    borg_data,

    aes(
      x = cycles_per_instruction
    )

  ) +

    geom_histogram(

      bins = 50,

      fill = "dodgerblue",

      color = "white"

    ) +

    labs(

      title =
        "CPU Efficiency Indicator Distribution",

      x =
        "Cycles per Instruction",

      y =
        "Frequency"

    ) +

    theme_minimal()


  ggsave(

    file.path(
      plot_dir,
      "05_cpu_distribution.png"
    ),

    p5,

    width = 9,
    height = 6

  )

}


# ==============================================================
# 19. MEMORY DISTRIBUTION
# ==============================================================

if ("assigned_memory" %in% names(borg_data)) {

  p6 <- ggplot(

    borg_data,

    aes(
      x = assigned_memory
    )

  ) +

    geom_histogram(

      bins = 50,

      fill = "mediumpurple",

      color = "white"

    ) +

    labs(

      title =
        "Assigned Memory Distribution",

      x =
        "Assigned Memory",

      y =
        "Frequency"

    ) +

    theme_minimal()


  ggsave(

    file.path(
      plot_dir,
      "06_memory_distribution.png"
    ),

    p6,

    width = 9,
    height = 6

  )

}


# ==============================================================
# 20. CPU VS MEMORY
# ==============================================================

if (
  all(
    c(
      "cycles_per_instruction",
      "assigned_memory"
    ) %in% names(borg_data)
  )
) {

  sample_data <- borg_data %>%

    select(
      cycles_per_instruction,
      assigned_memory
    ) %>%

    filter(
      is.finite(cycles_per_instruction),
      is.finite(assigned_memory)
    ) %>%

    slice_sample(
      n = min(20000, n())
    )


  p7 <- ggplot(

    sample_data,

    aes(

      x =
        cycles_per_instruction,

      y =
        assigned_memory

    )

  ) +

    geom_point(
      alpha = 0.25
    ) +

    labs(

      title =
        "CPU Efficiency vs Assigned Memory",

      x =
        "Cycles per Instruction",

      y =
        "Assigned Memory"

    ) +

    theme_minimal()


  ggsave(

    file.path(
      plot_dir,
      "07_cpu_vs_memory.png"
    ),

    p7,

    width = 9,
    height = 6

  )

}


# ==============================================================
# 21. RESOURCE METRICS BY FAILURE
# ==============================================================

if (
  all(
    c(
      "failed",
      "assigned_memory"
    ) %in% names(borg_data)
  )
) {

  resource_data <- borg_data %>%

    select(
      failed,
      assigned_memory
    ) %>%

    filter(
      !is.na(failed),
      !is.na(assigned_memory)
    )


  p8 <- ggplot(

    resource_data,

    aes(

      x = failed,

      y = assigned_memory

    )

  ) +

    geom_boxplot(
      alpha = 0.7
    ) +

    labs(

      title =
        "Assigned Memory by Failure Status",

      x =
        "Failure Status",

      y =
        "Assigned Memory"

    ) +

    theme_minimal()


  ggsave(

    file.path(
      plot_dir,
      "08_resource_by_failure.png"
    ),

    p8,

    width = 9,
    height = 6

  )

}


# ==============================================================
# 22. PRIORITY DISTRIBUTION
# ==============================================================

if ("priority" %in% names(borg_data)) {

  p9 <- ggplot(

    borg_data,

    aes(
      x = priority
    )

  ) +

    geom_histogram(

      bins = 30,

      fill = "seagreen",

      color = "white"

    ) +

    labs(

      title =
        "Priority Distribution",

      x =
        "Priority",

      y =
        "Frequency"

    ) +

    theme_minimal()


  ggsave(

    file.path(
      plot_dir,
      "09_priority_distribution.png"
    ),

    p9,

    width = 9,
    height = 6

  )

}


# ==============================================================
# 23. MACHINE ACTIVITY
# ==============================================================

if ("machine_id" %in% names(borg_data)) {

  machine_data <- borg_data %>%

    count(
      machine_id,
      name = "records"
    ) %>%

    arrange(
      desc(records)
    ) %>%

    slice_head(
      n = 20
    )


  p10 <- ggplot(

    machine_data,

    aes(

      x = reorder(
        factor(machine_id),
        records
      ),

      y = records

    )

  ) +

    geom_col(
      fill = "steelblue"
    ) +

    coord_flip() +

    labs(

      title =
        "Top 20 Machines by Recorded Activity",

      x =
        "Machine ID",

      y =
        "Number of Records"

    ) +

    theme_minimal()


  ggsave(

    file.path(
      plot_dir,
      "10_machine_activity.png"
    ),

    p10,

    width = 9,
    height = 7

  )

}


# ==============================================================
# 24. TEMPORAL TREND
# ==============================================================

if (
  all(
    c(
      "time",
      "cycles_per_instruction"
    ) %in% names(borg_data)
  )
) {

  temporal_data <- borg_data %>%

    select(
      time,
      cycles_per_instruction
    ) %>%

    filter(
      !is.na(time),
      !is.na(cycles_per_instruction)
    ) %>%

    arrange(time)


  # Divide observations into time windows

  temporal_data <- temporal_data %>%

    mutate(

      time_window =
        ntile(
          time,
          50
        )

    ) %>%

    group_by(
      time_window
    ) %>%

    summarise(

      time =
        median(
          time,
          na.rm = TRUE
        ),

      cpu_efficiency =
        median(
          cycles_per_instruction,
          na.rm = TRUE
        ),

      .groups = "drop"

    )


  p11 <- ggplot(

    temporal_data,

    aes(

      x = time,

      y = cpu_efficiency

    )

  ) +

    geom_line(
      linewidth = 1
    ) +

    labs(

      title =
        "CPU Efficiency Trend Over Time",

      x =
        "Time",

      y =
        "Median Cycles per Instruction"

    ) +

    theme_minimal()


  ggsave(

    file.path(
      plot_dir,
      "11_temporal_trend.png"
    ),

    p11,

    width = 10,
    height = 6

  )

}


# ==============================================================
# 25. CORRELATION HEATMAP
# ==============================================================

cat("Generating correlation heatmap...\n")


numeric_data <- borg_data %>%

  select(
    where(is.numeric)
  )


# Remove columns with no variation

numeric_data <- numeric_data[

  sapply(

    numeric_data,

    function(x) {

      variance <-
        var(
          x,
          na.rm = TRUE
        )

      is.finite(variance) &&
        variance > 0

    }

  )

]


if (ncol(numeric_data) >= 2) {

  correlation_matrix <- cor(

    numeric_data,

    use = "pairwise.complete.obs"

  )


  png(

    filename =
      file.path(
        plot_dir,
        "12_correlation_heatmap.png"
      ),

    width = 1400,

    height = 1100

  )


  corrplot(

    correlation_matrix,

    method = "color",

    type = "upper",

    order = "hclust",

    tl.col = "black",

    tl.cex = 0.65,

    number.cex = 0.5,

    addCoef.col = "black",

    number.digits = 2

  )


  dev.off()

}


# ==============================================================
# 26. GENERATE PDF REPORT
# ==============================================================

cat("\nGenerating PDF report...\n")


pdf(

  file.path(
    output_dir,
    "R_EDA_Report.pdf"
  ),

  width = 10,

  height = 7

)


print(p1)


if (exists("p2")) {
  print(p2)
}

if (exists("p3")) {
  print(p3)
}

if (exists("p4")) {
  print(p4)
}

if (exists("p5")) {
  print(p5)
}

if (exists("p6")) {
  print(p6)
}

if (exists("p7")) {
  print(p7)
}

if (exists("p8")) {
  print(p8)
}

if (exists("p9")) {
  print(p9)
}

if (exists("p10")) {
  print(p10)
}

if (exists("p11")) {
  print(p11)
}


# Correlation heatmap

if (exists("correlation_matrix")) {

  corrplot(

    correlation_matrix,

    method = "color",

    type = "upper",

    order = "hclust",

    tl.col = "black",

    tl.cex = 0.65,

    number.cex = 0.5,

    addCoef.col = "black"

  )

}


dev.off()


# ==============================================================
# 27. FINAL SUMMARY
# ==============================================================

cat("\n")
cat("====================================================\n")
cat("       R ANALYSIS COMPLETED SUCCESSFULLY\n")
cat("====================================================\n\n")


cat("Dataset:\n")

cat(
  "Rows    :",
  nrow(borg_data),
  "\n"
)

cat(
  "Columns :",
  ncol(borg_data),
  "\n\n"
)


cat("Generated:\n")

cat(
  "✓ Cleaned dataset\n"
)

cat(
  "✓ Preprocessing report\n"
)

cat(
  "✓ Missing-value analysis\n"
)

cat(
  "✓ Event distribution\n"
)

cat(
  "✓ Failure distribution\n"
)

cat(
  "✓ CPU distribution\n"
)

cat(
  "✓ Memory distribution\n"
)

cat(
  "✓ CPU vs memory analysis\n"
)

cat(
  "✓ Resource vs failure analysis\n"
)

cat(
  "✓ Priority distribution\n"
)

cat(
  "✓ Machine activity analysis\n"
)

cat(
  "✓ Temporal analysis\n"
)

cat(
  "✓ Correlation heatmap\n"
)

cat(
  "✓ PDF EDA report\n\n"
)


cat(
  "Output directory:",
  output_dir,
  "\n"
)


cat("\n====================================================\n")
cat("                 DONE!\n")
cat("====================================================\n")