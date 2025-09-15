"""
extract_expert_compositions.py

Description:
    Extracts expert honey compositions from Excel sheets and write to structured CSV file.

Usage:
    python extract_expert_compositions.py

Inputs:
    - Expert composition Excel file (set in script: EXCEL_PATH)
    - Output CSV path (set in script: OUTPUT_CSV)

Outputs:
    - Expert compositions CSV file
"""

import pandas as pd
import os

# Configuration 

# Path to expert honey composition Excel file
EXCEL_PATH = 'path/to/file'

# Output CSV path
OUTPUT_CSV = 'path/to/file'

# Check if the input file exists before proceeding
if not os.path.exists(EXCEL_PATH):
    print(f"Error: Input file not found at '{EXCEL_PATH}'")
else:
    # Load the entire workbook
    try:
        xlsx = pd.ExcelFile(EXCEL_PATH, engine='openpyxl')
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        exit()

    # This list will store all the extracted data rows 
    composition_rows = []

    # Process each sheet in the Excel file
    print(f"Found {len(xlsx.sheet_names)} sheets to process...")
    for sheet_name in xlsx.sheet_names:
        print(f"Processing sheet: '{sheet_name}'...")
        # Read the sheet without a default header row
        df = pd.read_excel(xlsx, sheet_name=sheet_name, header=None)

        # Dynamic Row Finding 
        percent_row_idx = None
        for idx, val in enumerate(df.iloc[:, 0]):
            if isinstance(val, str) and val.strip().lower().startswith('% representation'):
                percent_row_idx = idx
                break

        if percent_row_idx is None:
            print(f"  - WARNING: Could not find '% representation' row in sheet '{sheet_name}'. Skipping.")
            continue

        # Data Extraction 
        taxa_headers = df.iloc[0]
        percent_values = df.iloc[percent_row_idx]

        # Iterate through the columns
        for col_idx in range(1, df.shape[1]):
            taxa_name = taxa_headers.iloc[col_idx]
            percent_value = percent_values.iloc[col_idx]

            if pd.notna(taxa_name) and pd.notna(percent_value):
                try:
                    cleaned_str = str(percent_value).strip().replace('%', '').replace(',', '.')
                    cleaned_percent = float(cleaned_str)

                    if cleaned_percent > 0:
                        # Standardize taxa names
                        taxa_str = str(taxa_name).strip()

                        # Skip 'total' entries
                        if taxa_str.lower() == "total":
                            continue

                        # Map PAL0007 & PAL0008 to Monocot sp. 5
                        if taxa_str in ["PAL0007", "PAL0008"]:
                            taxa_str = "Monocot sp. 5"

                        composition_rows.append({
                            "Sample": sheet_name,
                            "Taxon": taxa_str,
                            "Percentage": cleaned_percent
                        })
                except (ValueError, TypeError):
                    continue

    # Save to CSV 
    if composition_rows:
        composition_df = pd.DataFrame(composition_rows)

        # Merge duplicate taxa (e.g., PAL0007 + PAL0008 → Monocot sp. 5)
        composition_df = (
            composition_df
            .groupby(["Sample", "Taxon"], as_index=False)["Percentage"]
            .sum()
        )

        composition_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nProcessing complete. Expert compositions saved to '{OUTPUT_CSV}'")
        print(f"Extracted {len(composition_df)} rows of data.")
    else:
        print("\nProcessing complete, but no data was extracted. The output file was not created.")
