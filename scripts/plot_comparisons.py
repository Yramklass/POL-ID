import os
import pandas as pd
import matplotlib.pyplot as plt

import re

def normalize_taxon(taxon):
    taxon = taxon.lower()
    taxon = taxon.replace(" ", "_")
    taxon = taxon.replace(".", "")
    taxon = re.sub(r"sp_?(\d+)", r"sp\1", taxon)  # ensures both 'sp_1' and 'sp1' become 'sp1'
    taxon = re.sub(r"_+", "_", taxon)  # compress multiple underscores
    return taxon.strip('_')  # remove leading/trailing underscores

# Paths 
base_dir = '/home/yash/POL-ID/outputs/'
sample_dir = os.path.join(base_dir,'full_pipeline_outputs')
sample_dir = os.path.join(sample_dir,'76_ray_classifier')
expert_path = os.path.join(base_dir, 'expert_compositions.csv')
plots_dir = os.path.join(base_dir, 'plots')
plots_dir = os.path.join(plots_dir, '76_ray_classifier')
os.makedirs(plots_dir, exist_ok=True)

# Load expert compositions 
expert_df = pd.read_csv(expert_path)

# Process each sample
for sample in expert_df['Sample'].unique():
    expert_sample = expert_df[expert_df['Sample'] == sample][['Taxon', 'Percentage']].copy()
    sample_folder = os.path.join(sample_dir, sample)
    model_file = os.path.join(sample_folder, f'{sample}_composition.csv')
    
    if not os.path.exists(model_file):
        print(f"Missing model file for {sample}")
        continue

    model_df = pd.read_csv(model_file)
    model_sample = model_df[['Taxon', '% Composition']].copy()
    model_sample.rename(columns={'% Composition': 'Percentage'}, inplace=True)

    # Normalize labels 
    label_map = {
        'Uncertain': 'unknown_noise',
        # Add more mappings if needed
    }
  # Apply label map first (if any), then normalize
    expert_sample['Taxon'] = expert_sample['Taxon'].replace(label_map).apply(normalize_taxon)
    model_sample['Taxon'] = model_sample['Taxon'].replace(label_map).apply(normalize_taxon)


    # Set index after renaming
    expert_sample.set_index('Taxon', inplace=True)
    model_sample.set_index('Taxon', inplace=True)
    # Merge both sets of taxa
    all_taxa = expert_sample.index.union(model_sample.index)
    expert_values = expert_sample.reindex(all_taxa, fill_value=0)['Percentage']
    model_values = model_sample.reindex(all_taxa, fill_value=0)['Percentage']

    # Plotting 
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(all_taxa))
    bar_width = 0.35

    ax.bar(x, expert_values, width=bar_width, label='Expert', alpha=0.7)
    ax.bar([i + bar_width for i in x], model_values, width=bar_width, label='Model', alpha=0.7)

    ax.set_title(f'Composition Comparison: {sample}')
    ax.set_xticks([i + bar_width/2 for i in x])
    ax.set_xticklabels(all_taxa, rotation=45, ha='right')
    ax.set_ylabel('Percentage')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{sample}_comparison_bar.png'))
    plt.close()

    # Pie charts 
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # expert_values.plot.pie(ax=axes[0], autopct='%1.1f%%', startangle=90)
    # axes[0].set_title(f'Expert: {sample}')
    # axes[0].set_ylabel('')

    # model_values.plot.pie(ax=axes[1], autopct='%1.1f%%', startangle=90)
    # axes[1].set_title(f'Model: {sample}')
    # axes[1].set_ylabel('')

    # plt.tight_layout()
    # plt.savefig(os.path.join(plots_dir, f'{sample}_comparison_pie.png'))
    # plt.close()

    print(f"Plots saved for {sample}")
