import os
import pandas as pd
import matplotlib.pyplot as plt
import re

def normalize_taxon(taxon):
    """Cleans up taxon names for consistent comparison."""
    taxon = taxon.lower()
    taxon = taxon.replace(" ", "_")
    taxon = taxon.replace(".", "")
    taxon = re.sub(r"sp_?(\d+)", r"sp\1", taxon)
    taxon = re.sub(r"_+", "_", taxon)
    return taxon.strip('_')

# Path definitions 
base_dir = '/home/yash/POL-ID/outputs/'
run_folder = 'FINAL_77_WEIGHTED_0,1' 

sample_dir = os.path.join(base_dir, 'full_pipeline_outputs', run_folder)
expert_path = os.path.join(base_dir, 'expert_compositions.csv')
plots_dir = os.path.join(base_dir, 'plots', run_folder)
os.makedirs(plots_dir, exist_ok=True)


# Create a directory to store the text file comparisons
text_output_dir = os.path.join(base_dir, 'top5_comparisons', run_folder)
os.makedirs(text_output_dir, exist_ok=True)


# Load expert compositions
expert_df = pd.read_csv(expert_path)

# Process each sample 
for sample in expert_df['Sample'].unique():
    expert_sample_df = expert_df[expert_df['Sample'] == sample][['Taxon', 'Percentage']].copy()
    sample_folder = os.path.join(sample_dir, sample)
    model_file = os.path.join(sample_folder, f'{sample}_composition.csv')

    if not os.path.exists(model_file):
        print(f"Skipping: Missing model file for {sample}")
        continue

    model_df = pd.read_csv(model_file)
    model_sample_df = model_df[['Taxon', '% Composition']].copy()
    model_sample_df.rename(columns={'% Composition': 'Percentage'}, inplace=True)

    # Normalize Taxon Labels 
    label_map = {'Uncertain': 'unknown_noise'}
    expert_sample_df['Taxon'] = expert_sample_df['Taxon'].replace(label_map).apply(normalize_taxon)
    model_sample_df['Taxon'] = model_sample_df['Taxon'].replace(label_map).apply(normalize_taxon)

    # Compare the top 5 taxa for expert and model 
    expert_top5 = expert_sample_df.nlargest(5, 'Percentage')
    model_top5 = model_sample_df.nlargest(5, 'Percentage')

    # Prepare the comparison text
    comparison_text = f"Top 5 Taxa Comparison for Sample: {sample}\n"
    comparison_text += "="*50 + "\n\n"

    comparison_text += "--- Expert Top 5 ---\n"
    for _, row in expert_top5.iterrows():
        comparison_text += f"- {row['Taxon']}: {row['Percentage']:.2f}%\n"

    comparison_text += "\n--- Model Top 5 ---\n"
    for _, row in model_top5.iterrows():
        comparison_text += f"- {row['Taxon']}: {row['Percentage']:.2f}%\n"

    # Save the text to a file
    text_file_path = os.path.join(text_output_dir, f'{sample}_top5_comparison.txt')
    with open(text_file_path, 'w') as f:
        f.write(comparison_text)


    # Set index for plotting
    expert_sample_df.set_index('Taxon', inplace=True)
    model_sample_df.set_index('Taxon', inplace=True)

    # Merge taxa for a comprehensive plot
    all_taxa = expert_sample_df.index.union(model_sample_df.index)
    expert_values = expert_sample_df.reindex(all_taxa, fill_value=0)['Percentage']
    model_values = model_sample_df.reindex(all_taxa, fill_value=0)['Percentage']

    # Plotting bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    x = range(len(all_taxa))
    bar_width = 0.35

    ax.bar(x, expert_values, width=bar_width, label='Expert', alpha=0.8)
    ax.bar([i + bar_width for i in x], model_values, width=bar_width, label='Model', alpha=0.8)

    ax.set_title(f'Composition Comparison: {sample}')
    ax.set_xticks([i + bar_width/2 for i in x])
    ax.set_xticklabels(all_taxa, rotation=45, ha='right')
    ax.set_ylabel('Percentage (%)')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{sample}_comparison_bar.png'))
    plt.close()

    print(f"Plots and top 5 comparison saved for {sample}")