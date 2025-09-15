"""
select_samples_by_class_limit.py

Description:
    Creates a list of taxa to be included. Includes <= the maximum number of taxa specified. Contains the maximum number of honeys possible while including all taxa from each included honey.

Usage:
    python select_samples_by_class_limit.py <expert_compositions.csv> <max_taxa>

Inputs:
    - Expert compositions CSV file path
    - Maximum number of taxa to include

Outputs:
    - List of taxa (< max_taxa) from as many honeys as possible
"""

import pandas as pd
import argparse
from contextlib import redirect_stdout
import sys

def convert_taxon_name(taxon):
    """Convert taxon names from CSV format to data folder format.
    Example: 'Erica sp. 1' -> 'Erica_sp1'"""
    if ' sp. ' in taxon:
        # Replace ' sp. ' with '_sp' and remove space before number
        return taxon.replace(' sp. ', '_sp')
    if taxon.lower()=='uncertain':
        return ("# {}".format(taxon))
    return taxon

def select_samples_with_class_limit(csv_path, max_classes, output_file=None):
    # Load CSV and convert taxon names
    df = pd.read_csv(csv_path)
    df['Taxon'] = df['Taxon'].apply(convert_taxon_name)

    # Build a dict: sample -> set of taxa
    sample_to_taxa = df.groupby("Sample")["Taxon"].apply(set).to_dict()

    selected_samples = []
    included_taxa = set()

    # Sort samples by how many unique taxa they have (descending)
    sorted_samples = sorted(sample_to_taxa.items(), key=lambda x: len(x[1]))

    for sample, taxa_set in sorted_samples:
        # How many new taxa would be added if we included this sample?
        new_taxa = taxa_set - included_taxa
        if len(included_taxa) + len(new_taxa) <= max_classes:
            selected_samples.append(sample)
            included_taxa.update(taxa_set)

    # Prepare output
    output_lines = [
        f"#Selected {len(selected_samples)} samples",
        f"#Total unique taxa: {len(included_taxa)}",
        "\n#Samples to include:"
    ]
    output_lines.extend(f"#  - {s}" for s in selected_samples)
    output_lines.extend([
        "\n#Taxa to include:",
        *sorted(included_taxa)
    ])

    # Output to console
    print('\n'.join(output_lines))

    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"\n#Output saved to {output_file}")

    return selected_samples, included_taxa


# CLI usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select samples constrained by max number of taxa.")
    parser.add_argument("csv", help="Path to expert_compositions.csv")
    parser.add_argument("max_classes", type=int, help="Maximum number of taxa allowed")
    parser.add_argument("--output", "-o", default="classes_to_include.txt", 
                       help="Output file path (default: classes_to_include.txt)")
    args = parser.parse_args()

    select_samples_with_class_limit(args.csv, args.max_classes, args.output)