import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Config ---
INPUT_CSV = "/home/yash/POL-ID/data/sample_input/expert_compositions.csv"  # Path to the CSV file
OUTPUT_DIR = "/home/yash/POL-ID/data/sample_input/expert_composition_plots"  # Folder to save the graphs
# --- End Config ---

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the expert composition CSV
df = pd.read_csv(INPUT_CSV)

# Group by sample
samples = df['Sample'].unique()

print(f"Generating plots for {len(samples)} samples...")

for sample in samples:
    sample_df = df[df['Sample'] == sample].copy()

    # Sort by percentage for clean plotting
    sample_df.sort_values(by='Percentage', ascending=False, inplace=True)

    # Plot setup
    plt.figure(figsize=(10, 6))
    sns.barplot(data=sample_df, x='Taxon', y='Percentage', palette='viridis')

    plt.title(f"Expert Composition: {sample}")
    plt.ylabel("% Representation")
    plt.xlabel("Taxon")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save to file
    output_path = os.path.join(OUTPUT_DIR, f"{sample}_composition.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Saved: {output_path}")

print("\nAll plots generated.")
