import json
import argparse
import bigframes.pandas as bpd
import bigframes.bigquery as bq
from bigframes.ml.llm import TextEmbeddingGenerator
import os

def main(project_id: str, location: str, dataset_id: str, test_csv: str, train_table_id: str, metrics_file: str):
    """
    Uses the trained index to evaluate all 5 prediction strategies on the test set,
    replicating the notebook's logic exactly.
    """
    print("--- Setting up Environment and BigFrames Session ---")
    os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
    os.environ['GRPC_DNS_RESOLVER'] = 'native'
    os.environ['GOOGLE_CLOUD_LOCATION'] = 'global'
    os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'True'
    bpd.options.bigquery.project = project_id
    bpd.options.bigquery.location = location

    print(f"Reading test data from {test_csv}...")
    test_df = bpd.read_csv(test_csv)

    print("Generating embeddings for the test set...")
    embedding_model = TextEmbeddingGenerator(model_name='gemini-embedding-001')
    test_search_query = embedding_model.predict(test_df['embedding_text'])
    

    labels_df_test = test_df[['sus', 'evil']]
    test_embeddings_df = bpd.concat([test_search_query, labels_df_test], axis=1)
    test_embeddings_df = test_embeddings_df.rename(columns={
        'ml_generate_embedding_result': 'test_embedding', 
        'content':'test_content', 'sus': 'test_sus', 'evil': 'test_evil'
    })

    print("Executing vector search to find top 5 neighbors...")
    train_table_path = f"{dataset_id}.{train_table_id}"
    train_table_bf = bpd.read_gbq(train_table_path)
    
    results_df = bq.vector_search(
        query=test_embeddings_df, base=train_table_bf,
        query_column="test_embedding", base_column="train_embedding", top_k=5
    )

    print("Calculating metrics for all 5 strategies...")
    
    # --- Strategy 1: Naive
    
    results_k1_df = results_df.loc[results_df.groupby("test_content")["distance"].idxmin()]
    k1_table_path = f"{dataset_id}.temp_results_k1"
    results_k1_df.to_gbq(k1_table_path, if_exists="replace")
    
    metrics_query = f"""
        SELECT
          COUNTIF(test_evil = 1 AND train_evil = 1) AS tp,
          COUNTIF(test_evil = 0 AND train_evil = 0) AS tn,
          COUNTIF(test_evil = 0 AND train_evil = 1) AS fp,
          COUNTIF(test_evil = 1 AND train_evil = 0) AS fn
        FROM `{k1_table_path}`
    """
    metrics_k1 = bpd.read_gbq(metrics_query).to_pandas().iloc[0].to_dict()

    # --- Strategies 2-5: Majority Votes (using your groupby logic) ---
    results_grouped = results_df.groupby('test_content')
    majority_votes_df = results_grouped.agg({
        'train_evil': 'mean',
        'test_evil': 'first',
        'train_sus': 'mean'
    }).to_pandas()

    all_metrics = {"baseline_k1": metrics_k1}
    y_true = majority_votes_df['test_evil']

    # Strategy 2: Strict Majority (3/5)
    y_pred_3_5 = (majority_votes_df['train_evil'] >= 0.6).astype("Int64")
    all_metrics['majority_3_5'] = {'tp': int(((y_true == 1) & (y_pred_3_5 == 1)).sum()), 'tn': int(((y_true == 0) & (y_pred_3_5 == 0)).sum()), 'fp': int(((y_true == 0) & (y_pred_3_5 == 1)).sum()), 'fn': int(((y_true == 1) & (y_pred_3_5 == 0)).sum())}

    # Strategy 3: Loose Majority (2/5)
    y_pred_2_5 = (majority_votes_df['train_evil'] >= 0.4).astype("Int64")
    all_metrics['majority_2_5'] = {'tp': int(((y_true == 1) & (y_pred_2_5 == 1)).sum()), 'tn': int(((y_true == 0) & (y_pred_2_5 == 0)).sum()), 'fp': int(((y_true == 0) & (y_pred_2_5 == 1)).sum()), 'fn': int(((y_true == 1) & (y_pred_2_5 == 0)).sum())}
    
    # Strategy 4: Maximum Sensitivity (1/5)
    y_pred_1_5 = (majority_votes_df['train_evil'] >= 0.2).astype("Int64")
    all_metrics['majority_1_5'] = {'tp': int(((y_true == 1) & (y_pred_1_5 == 1)).sum()), 'tn': int(((y_true == 0) & (y_pred_1_5 == 0)).sum()), 'fp': int(((y_true == 0) & (y_pred_1_5 == 1)).sum()), 'fn': int(((y_true == 1) & (y_pred_1_5 == 0)).sum())}

    # Strategy 5: Hybrid
    y_pred_hybrid = ((majority_votes_df['train_evil'] >= 0.2) | (majority_votes_df['train_sus'] >= 0.6)).astype("Int64")
    all_metrics['hybrid'] = {'tp': int(((y_true == 1) & (y_pred_hybrid == 1)).sum()), 'tn': int(((y_true == 0) & (y_pred_hybrid == 0)).sum()), 'fp': int(((y_true == 0) & (y_pred_hybrid == 1)).sum()), 'fn': int(((y_true == 1) & (y_pred_hybrid == 0)).sum())}

    print(f"Saving aggregated metrics for all strategies to {metrics_file}...")
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    print("Evaluation complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the vector search model.")
    parser.add_argument("--project-id", required=False, default="playexch-80b6", help="Google Cloud project ID.")
    parser.add_argument("--location", default="US", help="BigQuery location.")
    parser.add_argument("--dataset-id", required=False, default="ePBF_samples", help="BigQuery dataset ID.")
    parser.add_argument("--test-csv", required=False, default="/Users/jeffreysam/pvotal_git/eBPF-Vector-Index/BETH/preprocessed/test.csv" ,help="Path to the test CSV from prepare_data.py.")
    parser.add_argument("--train-table-id", required=False, default="BETH_train_embeddings", help="BQ table with indexed training embeddings.")
    parser.add_argument("--metrics-file", required=False, default="/Users/jeffreysam/pvotal_git/eBPF-Vector-Index/evaluation/metrics.json",help="JSON file to save final evaluation metrics.")
    args = parser.parse_args()
    main(args.project_id, args.location, args.dataset_id, args.test_csv, args.train_table_id, args.metrics_file)