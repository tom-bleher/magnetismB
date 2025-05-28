import pandas as pd
import numpy as np
import os
from pathlib import Path

def analyze_emf_data():
    """
    Analyze EMF data from folders A and B, extract peak and minimum values
    for each height, and save to Excel with two sheets.
    """
    
    # Define paths - fixed to point to actual data location
    base_path = Path("/home/tom/Desktop/magnetism/MagnetismB/data")
    folder_a = base_path / "A"
    folder_b = base_path / "B"
    
    def process_folder(folder_path, folder_name):
        """Process all CSV files in a folder and extract peak/min EMF values"""
        results = []
        
        # Check if folder exists
        if not folder_path.exists():
            print(f"Warning: Folder {folder_path} does not exist")
            return pd.DataFrame()
        
        # Get all CSV files in the folder
        csv_files = list(folder_path.glob("*.csv"))
        
        for csv_file in csv_files:
            # Extract height from filename (e.g., "65.5cm.csv" -> 65.5)
            height_cm = float(csv_file.stem.replace("cm", ""))
            height_m = height_cm / 100  # Convert to meters
            
            # Read the CSV file
            try:
                df = pd.read_csv(csv_file)
                
                # Extract voltage column (EMF values)
                voltage_col = "Voltage I/O-1(V)"
                if voltage_col in df.columns:
                    emf_values = df[voltage_col]
                    
                    # Calculate peak (maximum) and minimum EMF
                    peak_emf = emf_values.max()
                    min_emf = emf_values.min()
                    
                    results.append({
                        'Height (m)': height_m,
                        'Peak EMF (V)': peak_emf,
                        'Min EMF (V)': min_emf
                    })
                    
                    print(f"Processed {csv_file.name}: Height={height_m}m, Peak={peak_emf:.4f}V, Min={min_emf:.4f}V")
                
            except Exception as e:
                print(f"Error processing {csv_file.name}: {e}")
        
        # Sort results by height
        results.sort(key=lambda x: x['Height (m)'])
        
        return pd.DataFrame(results)
    
    # Process both folders independently
    print("Processing folder A...")
    df_a = process_folder(folder_a, "A")
    
    print("\nProcessing folder B...")
    df_b = process_folder(folder_b, "B")
    
    # Create Excel file with two sheets
    output_file = "emf_analysis_results.xlsx"
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        if not df_a.empty:
            df_a.to_excel(writer, sheet_name='A', index=False)
        if not df_b.empty:
            df_b.to_excel(writer, sheet_name='B', index=False)
    
    print(f"\nResults saved to {output_file}")
    print(f"Sheet A contains {len(df_a)} measurements")
    print(f"Sheet B contains {len(df_b)} measurements")
    
    # Display summary
    print("\n=== SUMMARY ===")
    if not df_a.empty:
        print("\nFolder A Results:")
        print(df_a.to_string(index=False))
    else:
        print("\nFolder A: No data found")
    
    if not df_b.empty:
        print("\nFolder B Results:")
        print(df_b.to_string(index=False))
    else:
        print("\nFolder B: No data found")
    
    return df_a, df_b

if __name__ == "__main__":
    analyze_emf_data()
