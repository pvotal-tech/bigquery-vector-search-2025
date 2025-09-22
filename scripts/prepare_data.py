import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main(input_csv: str, train_csv: str, test_csv: str, test_size: float):
    """
    Splits the processed data into training and testing sets.
    """
    print(f"Reading processed data from {input_csv}...")
    df = pd.read_csv(input_csv)

    print(f"Splitting data with test size {test_size} and stratifying by 'evil'...")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['evil'],
        random_state=42
    )

    print(f"Saving training data to {train_csv}...")
    train_df.to_csv(train_csv, index=False)
    
    print(f"Saving test data to {test_csv}...")
    test_df.to_csv(test_csv, index=False)
    print("Data preparation complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split processed data into train and test sets.")
    parser.add_argument("--input-csv", required=False, default= '/Users/jeffreysam/pvotal_git/eBPF-Vector-Index/BETH/preprocessed/combined_df.csv',help="Path to the processed CSV from preprocess.py.")
    parser.add_argument("--train-csv", required=False, default= '/Users/jeffreysam/pvotal_git/eBPF-Vector-Index/BETH/preprocessed/train.csv' ,help="Path to save the output training CSV.")
    parser.add_argument("--test-csv", required=False, default='/Users/jeffreysam/pvotal_git/eBPF-Vector-Index/BETH/preprocessed/test.csv', help="Path to save the output testing CSV.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Proportion of the dataset for the test set.")
    args = parser.parse_args()
    main(args.input_csv, args.train_csv, args.test_csv, args.test_size)