import os
import argparse
import bigframes.pandas as bpd
from bigframes.ml.llm import TextEmbeddingGenerator
import bigframes.bigquery as bq


def main(project_id: str, location: str, dataset_id: str, train_csv: str, table_id: str, index_name: str):
    """
    Uploads training data to BigQuery, generates embeddings, and creates a vector index,
    using the exact logic and functions from the source notebook.
    """
    print("--- Setting up Environment and BigFrames Session ---")
    
    # CRITICAL: Replicating the environment variable setup from the notebook
    os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
    os.environ['GRPC_DNS_RESOLVER'] = 'native'
    os.environ['GOOGLE_CLOUD_LOCATION'] = 'global'
    os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'True'
    
    bpd.options.bigquery.project = project_id
    bpd.options.bigquery.location = location
    bpd.options.bigquery.allow_large_results = True

    print(f"Reading training data from {train_csv}...")
    train_df = bpd.read_csv(train_csv)

    print("Initializing embedding model...")
    embedding_model = TextEmbeddingGenerator(model_name='gemini-embedding-001')

    print("Generating embeddings for training data...")
    train_embeddings_df = embedding_model.predict(train_df['embedding_text'])

    labels_df = train_df[['sus', 'evil']]
    train_embeddings_df = bpd.concat([train_embeddings_df, labels_df], axis=1)
    train_embeddings_df = train_embeddings_df.rename(columns={
        'ml_generate_embedding_result': 'train_embedding',
        'content':'train_content',
        'sus': 'train_sus',
        'evil': 'train_evil'
    })
    train_embeddings_df = train_embeddings_df[['train_content', 'train_embedding', 'train_sus', 'train_evil']]

    full_table_path = f"{dataset_id}.{table_id}"
    print(f"Saving training embeddings to BigQuery table: {full_table_path}")
    train_embeddings_df.to_gbq(full_table_path, if_exists="replace")

    print(f"Creating vector index '{index_name}' using bpd.bigquery.create_vector_index...")
    
    bq.create_vector_index(
        table_id=full_table_path,
        column_name='train_embedding',
        index_name=index_name,
        replace=True,
        index_type='tree_ah',
        distance_type='COSINE'
    )

    print("Training and indexing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and index embeddings in BigQuery.")
    parser.add_argument("--project-id", required=True, help="Google Cloud project ID.")
    parser.add_argument("--location", default="US", help="BigQuery location.")
    parser.add_argument("--dataset-id", required=True, help="BigQuery dataset ID.")
    parser.add_argument("--train-csv", required=True, help="Path to the training CSV from prepare_data.py.")
    parser.add_argument("--table-id", required=True, help="Name for the BQ table to store embeddings.")
    parser.add_argument("--index-name", required=True, help="Name for the vector index.")
    args = parser.parse_args()
    main(args.project_id, args.location, args.dataset_id, args.train_csv, args.table_id, args.index_name)