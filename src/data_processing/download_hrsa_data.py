"""
HRSA AHRF Data Download and Processing
Downloads Area Health Resources File data and prepares system capacity features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
import zipfile
import io

class HRSADataLoader:
    """
    Downloads and processes HRSA Area Health Resources File (AHRF) data
    for system-level capacity features
    """
    
    def __init__(self, output_dir="healthcare_world_model/data/hrsa"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_ahrf_data(self, year=2023):
        """
        Download AHRF data from HRSA
        Note: This is a placeholder - actual download requires manual step
        from https://data.hrsa.gov/data/download
        """
        print(f"HRSA AHRF Data Download Instructions")
        print("="*60)
        print("1. Go to: https://data.hrsa.gov/data/download")
        print("2. Under 'Health Workforce', find 'Area Health Resources Files'")
        print(f"3. Download AHRF {year} County-Level Data")
        print("4. Save to:", self.output_dir)
        print("="*60)
        print("\nExpected file: AHRF_*.xlsx or AHRF_SAS_*.zip")
        
        # Check if already downloaded
        existing_files = list(self.output_dir.glob("AHRF*.xlsx")) + \
                        list(self.output_dir.glob("AHRF*.csv"))
        
        if existing_files:
            print(f"\n✓ Found existing AHRF file: {existing_files[0]}")
            return existing_files[0]
        else:
            print("\n⚠ No AHRF file found. Please download manually.")
            return None
    
    def extract_system_capacity_features(self, ahrf_file):
        """
        Extract relevant system capacity variables from AHRF
        
        Key variables:
        - Primary care physicians per 100K (e.g., f0892113, f1198513)
        - Total hospital beds (e.g., f0453013, f0997113) 
        - ICU beds (if available)
        - ED visits (if available)
        - HPSA designation
        """
        print(f"\nProcessing AHRF file: {ahrf_file}")
        
        # Load data (handle different formats)
        if str(ahrf_file).endswith('.xlsx'):
            df = pd.read_excel(ahrf_file)
        elif str(ahrf_file).endswith('.csv'):
            df = pd.read_csv(ahrf_file)
        else:
            raise ValueError(f"Unsupported file format: {ahrf_file}")
        
        print(f"Loaded {len(df)} counties")
        print(f"Columns: {len(df.columns)}")
        
        # Common AHRF variable patterns (these vary by year)
        # Will need to check data dictionary for exact variable names
        capacity_vars = {
            'fips': ['fips', 'FIPS', 'f04439'],  # County FIPS code
            'state': ['state', 'STATE', 'f04440'],  # State FIPS
            'county_name': ['county', 'COUNTY'],
            'pcp_per_100k': ['f0892119', 'f0892120', 'f0892121'],  # Primary care physicians
            'total_physicians': ['f0889919', 'f0889920'],
            'hospital_beds': ['f0453019', 'f0453020', 'f0453021'],  # Total hospital beds
            'hospitals': ['f0888919', 'f0888920'],  # Number of hospitals
            'population': ['f1198519', 'f1198520', 'f1198521'],  # Total population
            'hpsa_shortage': ['f1525119', 'f1525120'],  # Primary care HPSA score
        }
        
        # Find actual column names in data
        selected_cols = {}
        for feature, possible_names in capacity_vars.items():
            for col in possible_names:
                if col in df.columns:
                    selected_cols[feature] = col
                    break
        
        print(f"\nFound {len(selected_cols)} capacity variables:")
        for k, v in selected_cols.items():
            print(f"  {k}: {v}")
        
        # Extract subset
        if len(selected_cols) < 3:
            print("\n⚠ Warning: Few capacity variables found.")
            print("Available columns (first 50):")
            print(df.columns[:50].tolist())
            return None
        
        # Create clean dataset
        rename_map = {v: k for k, v in selected_cols.items()}
        capacity_df = df[list(selected_cols.values())].copy()
        capacity_df = capacity_df.rename(columns=rename_map)
        
        # Derive additional features
        if 'pcp_per_100k' in capacity_df.columns and 'population' in capacity_df.columns:
            capacity_df['total_pcps'] = (capacity_df['pcp_per_100k'] / 100000) * capacity_df['population']
        
        if 'hospital_beds' in capacity_df.columns and 'population' in capacity_df.columns:
            capacity_df['beds_per_1000'] = (capacity_df['hospital_beds'] / capacity_df['population']) * 1000
        
        # Create capacity stress index
        capacity_df['capacity_index'] = self._create_capacity_index(capacity_df)
        
        return capacity_df
    
    def _create_capacity_index(self, df):
        """
        Create composite capacity index
        Higher = better capacity (less stressed)
        """
        index = pd.Series(0.0, index=df.index)
        
        if 'pcp_per_100k' in df.columns:
            # Normalize PCP availability (higher is better)
            pcp_norm = (df['pcp_per_100k'] - df['pcp_per_100k'].mean()) / df['pcp_per_100k'].std()
            index += pcp_norm * 0.4
        
        if 'beds_per_1000' in df.columns:
            beds_norm = (df['beds_per_1000'] - df['beds_per_1000'].mean()) / df['beds_per_1000'].std()
            index += beds_norm * 0.3
        
        if 'hpsa_shortage' in df.columns:
            # HPSA score: higher = worse shortage, so invert
            hpsa_norm = -(df['hpsa_shortage'] - df['hpsa_shortage'].mean()) / df['hpsa_shortage'].std()
            index += hpsa_norm * 0.3
        
        return index
    
    def aggregate_to_state_level(self, county_df):
        """
        Aggregate county data to state level for MEPS linkage
        (MEPS public files only have state identifiers)
        """
        print("\nAggregating to state level...")
        
        # Group by state
        state_df = county_df.groupby('state').agg({
            'pcp_per_100k': 'mean',
            'beds_per_1000': 'mean',
            'capacity_index': 'mean',
            'population': 'sum',
            'hospital_beds': 'sum',
            'hospitals': 'sum'
        }).reset_index()
        
        state_df.columns = ['state_fips'] + [f'state_{col}' for col in state_df.columns[1:]]
        
        print(f"Created state-level capacity data for {len(state_df)} states")
        return state_df
    
    def save_processed_data(self, df, filename):
        """Save processed capacity data"""
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        print(f"\n✓ Saved to: {output_path}")
        return output_path


if __name__ == "__main__":
    loader = HRSADataLoader()
    
    # Step 1: Download (manual step)
    ahrf_file = loader.download_ahrf_data(year=2023)
    
    if ahrf_file:
        # Step 2: Extract capacity features
        capacity_df = loader.extract_system_capacity_features(ahrf_file)
        
        if capacity_df is not None:
            # Step 3: Aggregate to state level
            state_capacity = loader.aggregate_to_state_level(capacity_df)
            
            # Step 4: Save
            loader.save_processed_data(state_capacity, "state_capacity_features.csv")
            loader.save_processed_data(capacity_df, "county_capacity_features.csv")
            
            print("\n" + "="*60)
            print("HRSA data processing complete!")
            print(f"State-level features: {len(state_capacity)} states")
            print(f"County-level features: {len(capacity_df)} counties")
