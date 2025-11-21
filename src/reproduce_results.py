"""
RSSM Reproducibility Script
Orchestrates the entire pipeline from data preparation to result generation.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"✅ {description} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}.")
        sys.exit(1)

def main():
    print("Starting RSSM Reproduction Pipeline...")
    
    # 1. Data Preparation
    # Assuming raw data is in place, run the prep scripts
    # Note: Actual raw data download requires manual steps due to MEPS/AHRF restrictions,
    # but we assume the files are in data/real_meps/ as per setup.
    
    # run_command("python rssm_data_loader.py", "Data Preparation (General)")
    # run_command("python rssm_medicaid_data_prep.py", "Data Preparation (Medicaid)")
    
    # 2. Training
    # Train the model on the prepared data
    run_command("python rssm_training.py", "RSSM Model Training")
    
    # 3. Validation - COVID-19
    # Validate against the COVID-19 natural experiment
    run_command("python rssm_validate.py", "COVID-19 Validation")
    
    # 4. Validation - Medicaid Expansion
    # Validate against the Medicaid expansion natural experiment
    run_command("python rssm_medicaid_validation.py", "Medicaid Expansion Validation")
    
    # 5. Novel Insights
    # Generate figures for capacity-demand coupling, shock propagation, etc.
    run_command("python rssm_novel_insights.py", "Novel Insights Generation")
    
    print("\n" + "="*60)
    print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Results and figures are available in the current directory.")

if __name__ == "__main__":
    # Ensure we are in the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
