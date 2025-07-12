import pandas as pd
import argparse

def select_samples_with_class_limit(csv_path, max_classes):
    # Load CSV
    df = pd.read_csv(csv_path)

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

    # Final output
    print(f"✅ Selected {len(selected_samples)} samples")
    print(f"🔢 Total unique taxa: {len(included_taxa)}")
    print(f"\n📦 Samples to include:")
    for s in selected_samples:
        print(f"  - {s}")
    print(f"\n🌿 Taxa to include:")
    for t in sorted(included_taxa):
        print(f"  - {t}")

    return selected_samples, included_taxa


# CLI usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select samples constrained by max number of taxa.")
    parser.add_argument("csv", help="Path to expert_compositions.csv")
    parser.add_argument("max_classes", type=int, help="Maximum number of taxa allowed")
    args = parser.parse_args()

    select_samples_with_class_limit(args.csv, args.max_classes)
