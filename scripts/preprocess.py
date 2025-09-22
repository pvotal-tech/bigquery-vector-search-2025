import os
import ast
import argparse
import pandas as pd
from tqdm.auto import tqdm

# (Functions parse_and_format_args and create_final_embedding_text are identical to your notebook)
def parse_and_format_args(arg_string: str) -> str:
    """Safely parses the argument string and formats it into a clean key=value summary."""
    if not isinstance(arg_string, str) or not arg_string.startswith('['):
        return arg_string
    try:
        arg_list = ast.literal_eval(arg_string)
        if isinstance(arg_list, list) and len(arg_list) > 0 and isinstance(arg_list[0], list):
            arg_list = arg_list[0]
        if isinstance(arg_list, list):
            formatted_args = [f"{arg.get('name', 'arg')}={arg.get('value', 'N/A')}" for arg in arg_list]
            return ', '.join(formatted_args)
        return arg_string
    except (ValueError, SyntaxError, MemoryError):
        return arg_string

def create_final_embedding_text(df: pd.DataFrame) -> pd.DataFrame:
    """Generates a contextual text sentence for each event."""
    # This inner function is a direct copy of the one in your notebook
    def generate_row_text(row):
        clean_args = parse_and_format_args(row['args'])
        try:
            ret_val = int(float(row['returnValue']))
            outcome = f"succeeded (return value: {ret_val})" if ret_val == 0 else f"failed with an error code (return value: {ret_val})" if ret_val < 0 else f"completed, returning a resource handle or ID (return value: {ret_val})"
        except (ValueError, TypeError):
            outcome = f"returned a non-numeric value ('{row['returnValue']}')"
        text = (f"Process '{row['processName']}' (PID: {row['processId']}), spawned by parent PID {row['parentProcessId']}, "
                f"performed a '{row['eventName']}' action with details: [{clean_args}]. This operation {outcome}.")
        # Simplified for brevity, add other context like threadId if needed
        return text

    key_cols = ['processName', 'eventName', 'args', 'returnValue', 'processId', 'parentProcessId']
    for col in key_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('N/A')
        else:
            df[col] = df[col].fillna(-1)
    df['embedding_text'] = df.apply(generate_row_text, axis=1)
    return df

def main(input_dir: str, output_csv: str):
    """Loads raw CSVs, validates, cleans, engineers features, and saves a final processed CSV."""
    print(f"Starting preprocessing of files in: {input_dir}")
    required_columns = {'timestamp', 'processId', 'parentProcessId', 'userId', 'processName',
                        'hostName', 'eventId', 'eventName', 'argsNum', 'returnValue', 'args', 'sus', 'evil'}
    list_of_dataframes = [pd.read_csv(os.path.join(input_dir, fn)) for fn in os.listdir(input_dir) if fn.endswith('.csv') and required_columns.issubset(pd.read_csv(os.path.join(input_dir, fn), nrows=0).columns)]
    
    print(f"Loaded {len(list_of_dataframes)} valid CSV files. Concatenating...")
    combined_df = pd.concat(list_of_dataframes, ignore_index=True)
    
    print("Generating narrative text for embeddings...")
    processed_df = create_final_embedding_text(combined_df)

    print("Deduplicating based on narrative text...")
    deduped_df = processed_df.drop_duplicates(subset=['embedding_text'])
    
    print(f"Saving processed data to: {output_csv}")
    deduped_df.to_csv(output_csv, index=False)
    print("Preprocessing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess BETH dataset CSV files.")
    parser.add_argument("--input-dir", required=False, default='/Users/jeffreysam/pvotal_git/eBPF-Vector-Index/BETH/beth-dataset-raw', help="Directory containing the raw BETH CSV files.")
    parser.add_argument("--output-csv", required=False, default= '/Users/jeffreysam/pvotal_git/eBPF-Vector-Index/BETH/preprocessed/combined_df.csv', help="Path to save the final processed CSV file.")
    args = parser.parse_args()
    main(args.input_dir, args.output_csv)