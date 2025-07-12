import pandas as pd
import os

# Honey composition Excel path
EXCEL_PATH = "/home/yash/POL-ID/data/sample_input/HoneyPhotoComposition_POLID.xlsx"

# Output CSV path
OUTPUT_CSV = "/home/yash/POL-ID/data/sample_input/expert_compositions.csv"


# Check if the input file exists before proceeding
if not os.path.exists(EXCEL_PATH):
    print(f"Error: Input file not found at '{EXCEL_PATH}'")
    print("Please make sure the file is in the same directory as the script, or provide the full path.")
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
        # Find the row number that contains '% representation' by searching the first column
        percent_row_idx = None
        # We use enumerate on the first column (iloc[:, 0]) to get the index and value
        for idx, val in enumerate(df.iloc[:, 0]):
            # Check if the cell value is a string and matches our target
            if isinstance(val, str) and val.strip().lower().startswith('% representation'):
                percent_row_idx = idx
                break # Stop searching once we've found it

        # If we didn't find the '% representation' row, we can't process this sheet.
        if percent_row_idx is None:
            print(f"  - WARNING: Could not find '% representation' row in sheet '{sheet_name}'. Skipping.")
            continue

        # Data Extraction 
        # Get the headers from the first row (index 0) and percentages from the dynamically found row.
        # We assume the taxon names are always in the first row of the sheet.
        taxa_headers = df.iloc[0]
        percent_values = df.iloc[percent_row_idx]

        # Iterate through the columns to get each taxon and its percentage
        # We start from column 1 to skip the first column (which contains labels like 'photo')
        for col_idx in range(1, df.shape[1]):
            taxa_name = taxa_headers.iloc[col_idx]
            percent_value = percent_values.iloc[col_idx]

            # Ensure both the taxon name and its percentage value exist (are not blank)
            if pd.notna(taxa_name) and pd.notna(percent_value):
                
                try:
                    # Clean and standardize decimal separator
                    cleaned_str = str(percent_value).strip().replace('%', '').replace(',', '.')
                    cleaned_percent = float(cleaned_str)

                    if cleaned_percent > 0:
                        composition_rows.append({
                            "Sample": sheet_name,
                            "Taxon": str(taxa_name).strip(),
                            "Percentage": cleaned_percent
                        })
                except (ValueError, TypeError):
                    continue


    # Save to CSV 
    # Convert list of data into a DataFrame and save it.
    if composition_rows:
        composition_df = pd.DataFrame(composition_rows)
        composition_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nProcessing complete. Expert compositions saved to '{OUTPUT_CSV}'")
        print(f"Extracted {len(composition_df)} rows of data.")
    else:
        print("\nProcessing complete, but no data was extracted. The output file was not created.")
