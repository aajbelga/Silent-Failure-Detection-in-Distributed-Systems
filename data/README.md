# Dataset Documentation

## Dataset Source

This project uses the Google Borg Cluster Data, which contains
large-scale cluster workload traces.

Official repository:
https://github.com/google/cluster-data

## Local Dataset Setup

The raw dataset is not included in this GitHub repository because
of its large file size.

Download the required dataset and place it at:

data/raw/borg_traces_data.csv

## Expected Format

The dataset is expected in CSV format and should contain the
Borg cluster trace fields required by the preprocessing pipeline.

## Data Processing

The project processes the raw data through:

1. Data cleaning and validation
2. Missing-value handling
3. Resource-feature extraction
4. Feature engineering
5. Target construction
6. Model training and evaluation

See `src/preprocessing.py` and `src/feature_engineering.py`
for implementation details.