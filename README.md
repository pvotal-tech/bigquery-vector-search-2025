# Anomaly Detection with Vector Search
Repository for the Bigquery AI competition 2025 [https://www.kaggle.com/competitions/bigquery-ai-hackathon]


This project implements an end-to-end framework for high-recall anomaly detection in cybersecurity logs. It leverages a semantic, unsupervised approach built on Google BigQuery's native vector search capabilities to identify novel threats that evade traditional detection methods.

It is recommended to view the **`Kaggle_GBQ_BETH.ipynb`** notebook first since that explains the project in depth.

For the purpose of reproducibility, the entire workflow is broken down into a series of modular Python scripts.

## 1. Setup Instructions

### Local Environment Setup
It is highly recommended to run this project in a Python virtual environment.

1.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    The required libraries are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

    **`requirements.txt` contents:**
    ```
    bigframes==1.38.0
    pandas==2.2.3
    tqdm==4.67.1
    scikit-learn==1.6.1
    numpy==2.2.3
    matplotlib==3.10.0
    seaborn==0.13.2
    google-cloud-bigquery==3.29.0
    ```

### Google Cloud (GCP) Setup
This project requires a Google Cloud project with the BigQuery API enabled.

1.  **Authenticate gcloud:**
    Log in with your GCP credentials. This will allow the scripts to access your project resources.
    ```bash
    gcloud auth application-default login
    ```

2.  **Create a BigQuery Dataset:**
    You'll need a dataset in BigQuery to store the tables and indexes. You can create one using the `bq` command-line tool.
    ```bash
    # Replace 'your-project-id' and 'BETH_analysis' with your own names
    bq --location=US mk --dataset your-project-id:BETH_analysis
    ```

### Data Acquisition

The BETH dataset used in this project is not included in this repository due to its large size.

1.  **Download the data** from the official Kaggle page:
    [https://www.kaggle.com/datasets/katehighnam/beth-dataset](https://www.kaggle.com/datasets/katehighnam/beth-dataset)

2.  **Place the files** in the correct directory. After downloading and unzipping, ensure the CSV files are located in a path like: `BETH/beth-dataset-raw/`. Your project root should contain the `BETH` directory.

---
## 2. Project Structure

The pipeline is organized into several modular scripts:

* **`preprocess.py`**: Ingests all raw CSV files from a specified directory, performs validation, engineers the "narrative" feature for each event, deduplicates the data, and outputs a single, clean `processed.csv` file.
* **`prepare_data.py`**: Takes the single processed CSV and performs a stratified train/test split, creating `train.csv` and `test.csv`.
* **`train_index.py`**: Uploads `train.csv` to BigQuery, generates embeddings using a Gemini model, saves them to a new table, and builds the vector search index.
* **`evaluate_model.py`**: Takes `test.csv`, generates embeddings for the test set, runs the vector search against the trained index, and calculates the raw performance numbers (TP, TN, FP, FN) for all five prediction strategies. The results are saved to a `metrics.json` file.
* **`generate_plots.py`**: Reads the `metrics.json` file and uses the provided `calculate_and_plot_metrics` function to generate and save the final confusion matrix visualizations for each evaluation strategy.

---
## 3. Running the Pipeline

The easiest way to run the end-to-end pipeline is using the provided shell script.

> **Important:** Before running, you **must** open `run_pipeline.sh` and edit the configuration variables at the top to match your GCP project ID, dataset name, and local directory paths. The default values are placeholders and will not work on your system.

### Example `run_pipeline.sh` Structure

This script defines all necessary configurations and executes each Python script in the correct order, passing the required arguments automatically.

```bash
#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- (REQUIRED) USER CONFIGURATION ---
export PROJECT_ID="your-gcp-project-id"
export BQ_LOCATION="US"
export BQ_DATASET="BETH_analysis"

# Local directories
RAW_DATA_DIR="./BETH_data" # IMPORTANT: Directory where you downloaded the BETH CSVs
PROCESSED_DATA_DIR="./processed_data"
PLOTS_DIR="./plots"
# --- END OF CONFIGURATION ---


# --- Script Variables (Generated from config) ---
PROCESSED_CSV="$PROCESSED_DATA_DIR/beth_processed.csv"
TRAIN_CSV="$PROCESSED_DATA_DIR/train.csv"
TEST_CSV="$PROCESSED_DATA_DIR/test.csv"
METRICS_FILE="$PROCESSED_DATA_DIR/evaluation_metrics.json"

# BigQuery Table and Index Names
BQ_EMBEDDINGS_TABLE="beth_train_embeddings"
BQ_INDEX_NAME="beth_embedding_index_v1"


# --- Create Directories ---
mkdir -p $PROCESSED_DATA_DIR
mkdir -p $PLOTS_DIR


# --- Pipeline Execution ---
echo "--- STEP 1: PREPROCESSING DATA ---"
python preprocess.py \
    --input-dir "$RAW_DATA_DIR" \
    --output-csv "$PROCESSED_CSV"

echo "--- STEP 2: PREPARING DATA (TRAIN/TEST SPLIT) ---"
python prepare_data.py \
    --input-csv "$PROCESSED_CSV" \
    --train-csv "$TRAIN_CSV" \
    --test-csv "$TEST_CSV" \
    --test-size 0.2

echo "--- STEP 3: TRAINING AND INDEXING ---"
python train_index.py \
    --project-id "$PROJECT_ID" \
    --location "$BQ_LOCATION" \
    --dataset-id "$BQ_DATASET" \
    --train-csv "$TRAIN_CSV" \
    --table-id "$BQ_EMBEDDINGS_TABLE" \
    --index-name "$BQ_INDEX_NAME"

echo "--- STEP 4: EVALUATING MODEL ---"
python evaluate_model.py \
    --project-id "$PROJECT_ID" \
    --location "$BQ_LOCATION" \
    --dataset-id "$BQ_DATASET" \
    --test-csv "$TEST_CSV" \
    --train-table-id "$BQ_EMBEDDINGS_TABLE" \
    --metrics-file "$METRICS_FILE"

echo "--- STEP 5: GENERATING PLOTS ---"
python generate_plots.py \
    --metrics-file "$METRICS_FILE" \
    --output-dir "$PLOTS_DIR"

echo "--- PIPELINE COMPLETE ---"
echo "Processed data is in: $PROCESSED_DATA_DIR"
echo "Evaluation metrics are in: $METRICS_FILE"
echo "All plots are in: $PLOTS_DIR"
```
