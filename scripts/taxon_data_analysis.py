import pandas as pd
import matplotlib.pyplot as plt

# File paths
taxon_summary_file = "/home/yash/POL-ID/data/classification/processed_crops/comprehensive_taxon_summary.csv"
metrics_file = "/home/yash/POL-ID/models/par_outputs/final_77_small/parallel_fusion/training_outputs_parallel_fusion/per_class_metrics.csv"

# Load data
taxon_summary = pd.read_csv(taxon_summary_file)
metrics = pd.read_csv(metrics_file)

# --- PART 1: Totals ---
total_images = taxon_summary["Original Images"].sum()
total_stacks = taxon_summary["Stacks"].sum()
total_cropped = taxon_summary["Total Cropped Grains"].sum()

print("=== Totals from Taxon Summary ===")
print(f"Total Original Images: {total_images}")
print(f"Total Stacks: {total_stacks}")
print(f"Total Cropped Grains: {total_cropped}")

# --- PART 2: Merge with metrics ---
# Ensure taxon names align (strip whitespace, unify underscores vs no underscores)
taxon_summary["Taxon"] = taxon_summary["Taxon"].str.strip().str.replace(" ", "_")
metrics["taxon"] = metrics["taxon"].str.strip().str.replace(" ", "_")

merged = pd.merge(metrics, taxon_summary, left_on="taxon", right_on="Taxon", how="inner")

# --- PART 3: Graph F1 vs Stacks ---
plt.figure(figsize=(8, 6))
plt.scatter(merged["Stacks"], merged["f1-score"], color="royalblue", alpha=0.7)

# Annotate each point with taxon name
for _, row in merged.iterrows():
    plt.text(row["Stacks"] + 0.1, row["f1-score"], row["taxon"], fontsize=8)

plt.xlabel("Number of Stacks")
plt.ylabel("F1-Score")
plt.title("F1-Score vs Number of Stacks")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 7))

# Scatter each taxon separately for unique legend entries
for taxon, group in merged.groupby("taxon"):
    plt.scatter(group["Stacks"], group["f1-score"], label=taxon)

plt.xlabel("Stacks")
plt.ylabel("F1-Score")
plt.title("F1-Score vs Stacks (per Taxon)")

# Multi-column legend
plt.legend(
    bbox_to_anchor=(1.05, 1), 
    loc='upper left', 
    fontsize=7,
    ncol=2  # <-- adjust this number for more/less columns
)

plt.tight_layout()
plt.show()
