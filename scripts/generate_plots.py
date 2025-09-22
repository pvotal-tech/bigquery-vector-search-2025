import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_and_plot_metrics(tp, tn, fp, fn, title_prefix="", output_path=None):
    """Calculates and prints key classification metrics and saves a confusion matrix."""
    tp_f, tn_f, fp_f, fn_f = np.float64(tp), np.float64(tn), np.float64(fp), np.float64(fn)
    accuracy = (tp_f + tn_f) / (tp_f + tn_f + fp_f + fn_f) if (tp_f + tn_f + fp_f + fn_f) > 0 else 0
    precision = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0
    recall = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"--- {title_prefix} Classification Metrics ---")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-Score:    {f1:.4f}\n")

    conf_matrix = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', cbar=False, annot_kws={"size": 14})
    plt.title(f'{title_prefix} Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    tick_labels = ['Negative (0)', 'Positive (1)']
    plt.xticks(ticks=[0.5, 1.5], labels=tick_labels)
    plt.yticks(ticks=[0.5, 1.5], labels=tick_labels, rotation=0, va="center")
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved confusion matrix to {output_path}")
    else:
        plt.show() # Default to showing if no path is given
    plt.close()

def main(metrics_file: str, output_dir: str):
    """
    Reads calculated metrics and generates a confusion matrix plot for each strategy.
    """
    print(f"Reading metrics from {metrics_file}...")
    os.makedirs(output_dir, exist_ok=True)
    with open(metrics_file, 'r') as f:
        all_metrics = json.load(f)

    title_map = {
        "baseline_k1": "Baseline (k=1)",
        "majority_3_5": "Attack Predictions (Strict Majority 3/5)",
        "majority_2_5": "Attack Predictions (Loose Majority 2/5)",
        "majority_1_5": "Attack Predictions (Maximum Sensitivity 1/5)",
        "hybrid": "Attack Predictions (Hybrid Rule)"
    }

    for strategy, values in all_metrics.items():
        if strategy in title_map:
            title = title_map[strategy]
            plot_path = os.path.join(output_dir, f"confusion_matrix_{strategy}.png")
            calculate_and_plot_metrics(
                tp=values['tp'], tn=values['tn'], fp=values['fp'], fn=values['fn'],
                title_prefix=title, output_path=plot_path
            )
            
    print("All plots generated.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate confusion matrix plots from metrics.")
    parser.add_argument("--metrics-file", required=False, default="/Users/jeffreysam/pvotal_git/eBPF-Vector-Index/evaluation/metrics.json", help="JSON file with evaluation metrics from evaluate_model.py.")
    parser.add_argument("--output-dir", required=False, default= "/Users/jeffreysam/pvotal_git/eBPF-Vector-Index/plots",help="Directory to save the generated plots.")
    args = parser.parse_args()
    main(args.metrics_file, args.output_dir)