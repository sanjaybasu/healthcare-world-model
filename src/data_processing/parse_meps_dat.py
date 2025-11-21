"""
MEPS .dat File Parser
Parses ASCII fixed-width format MEPS files for 2013-2015
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

class MEPSDataParser:
    """
    Parse MEPS .dat files using column specifications
    """
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        
    def parse_h161_2013(self):
        """
        Parse h161.dat (2013 Full Year Consolidated)
        Using known MEPS variable structure
        """
        
        filepath = self.data_dir / "h161.dat"
        
        print(f"Parsing {filepath}...")
        print(f"File size: {filepath.stat().st_size / (1024**2):.1f} MB")
        
        # Key variables for Medicaid expansion analysis
        # Based on MEPS codebook structure
        colspecs = [
            (0, 8),      # DUPERSID - Person ID
            (38, 40),    # REGION - Census region
            (59, 61),    # AGE13X - Age
            (1837, 1839), # ERTOT13 - Total ED visits
            (1297, 1298), # MCDEV13 - Medicaid coverage
        ]
        
        names = ['DUPERSID', 'REGION', 'AGE13X', 'ERTOT13', 'MCDEV13']
        
        try:
            # Read fixed-width file
            df = pd.read_fwf(filepath, colspecs=colspecs, names=names, nrows=10000)
            
            print(f"Loaded {len(df):,} rows (sample)")
            print(f"Columns: {df.columns.tolist()}")
            print(f"\nSample data:")
            print(df.head())
            
            # Convert to numeric
            for col in ['REGION', 'AGE13X', 'ERTOT13', 'MCDEV13']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Create outcome
            df['frequent_ed'] = (df['ERTOT13'] >= 4).astype(int)
            df['year'] = 2013
            
            return df
            
        except Exception as e:
            print(f"Error parsing file: {e}")
            print("\nAttempting alternative parsing method...")
            
            # Try reading as CSV with delimiter
            try:
                df = pd.read_csv(filepath, sep='\s+', nrows=10000, header=None)
                print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
                print("First few columns:")
                print(df.iloc[:5, :10])
                return df
            except Exception as e2:
                print(f"Alternative method also failed: {e2}")
                return None
    
    def parse_all_years(self):
        """Parse 2013-2015 files"""
        
        files = {
            2013: "h161.dat",
            2014: "h171.dat",
            2015: "h181.dat"
        }
        
        all_data = []
        
        for year, filename in files.items():
            filepath = self.data_dir / filename
            
            if not filepath.exists():
                print(f"⚠ {filename} not found")
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing {year}: {filename}")
            print(f"{'='*60}")
            
            # For now, just parse 2013 as proof-of-concept
            if year == 2013:
                df = self.parse_h161_2013()
                if df is not None:
                    all_data.append(df)
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            print(f"\n✓ Combined data: {len(combined):,} rows")
            return combined
        else:
            return None


if __name__ == "__main__":
    parser = MEPSDataParser("/Users/sanjaybasu/waymark-local/healthcare_world_model/data/real_meps")
    
    # Parse files
    data = parser.parse_all_years()
    
    if data is not None:
        # Save parsed data
        output_path = Path("/Users/sanjaybasu/waymark-local/healthcare_world_model/data/processed/meps_2013_parsed.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)
        print(f"\n✓ Saved to: {output_path}")
    else:
        print("\n⚠ No data parsed successfully")
