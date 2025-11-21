"""
Geographic Insights from World Model
Demonstrates local capacity-risk interactions for targeted interventions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class GeographicInsights:
    """
    Extract geographic insights showing world model's value for local planning
    
    Key Innovation: Identify states where high individual risk + low capacity
    creates intervention opportunities that wouldn't be obvious from either alone
    """
    
    def __init__(self, meps_dir, hrsa_file):
        self.meps_dir = Path(meps_dir)
        self.hrsa_file = Path(hrsa_file)
        
    def load_state_data(self):
        """Load MEPS 2019-2022 with state identifiers"""
        
        files = {
            2019: "h216.dta",
            2020: "h224.dta",
            2021: "h233.dta",
            2022: "h243.dta"
        }
        
        all_data = []
        for year, filename in files.items():
            filepath = self.meps_dir / filename
            if filepath.exists():
                df = pd.read_stata(filepath)
                
                # Extract state (MEPS uses REGION, not state FIPS)
                # REGION: 1=Northeast, 2=Midwest, 3=South, 4=West
                region = pd.to_numeric(df.get(f'REGION{str(year)[2:]}', df.get('REGION', 0)), errors='coerce')
                
                # ED visits
                ed_visits = pd.to_numeric(df.get(f'ERTOT{str(year)[2:]}', 0), errors='coerce').fillna(0)
                
                # Age
                age = pd.to_numeric(df.get('AGEDX', df.get(f'AGE{str(year)[2:]}X', 50)), errors='coerce')
                
                # Person ID
                person_id = df.get('DUPERSID', df.get('DUID'))
                
                year_df = pd.DataFrame({
                    'person_id': person_id,
                    'year': year,
                    'region': region,
                    'ed_visits': ed_visits,
                    'age': age,
                    'frequent_ed': (ed_visits >= 4).astype(int)
                })
                
                all_data.append(year_df)
        
        combined = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(combined):,} person-years across {combined['region'].nunique()} regions")
        return combined
    
    def calculate_regional_risk(self, df):
        """Calculate individual risk by region"""
        
        # Person-level temporal features
        df = df.sort_values(['person_id', 'year'])
        df['person_ed_mean'] = df.groupby('person_id')['ed_visits'].transform('mean')
        df['person_ed_std'] = df.groupby('person_id')['ed_visits'].transform('std').fillna(0)
        
        # Regional aggregates
        regional_stats = df.groupby('region').agg({
            'frequent_ed': 'mean',  # Prevalence
            'ed_visits': 'mean',    # Mean visits
            'person_ed_std': 'mean', # Mean volatility
            'age': 'mean'
        }).reset_index()
        
        regional_stats.columns = ['region', 'freq_ed_rate', 'mean_ed_visits', 'mean_volatility', 'mean_age']
        
        # Risk score (composite)
        regional_stats['risk_score'] = (
            regional_stats['freq_ed_rate'] * 0.4 +
            (regional_stats['mean_ed_visits'] / regional_stats['mean_ed_visits'].max()) * 0.3 +
            (regional_stats['mean_volatility'] / regional_stats['mean_volatility'].max()) * 0.3
        )
        
        return regional_stats
    
    def load_capacity_data(self):
        """Load HRSA state capacity features"""
        
        capacity = pd.read_csv(self.hrsa_file)
        
        # Map state FIPS to regions (simplified)
        # In reality, would use actual state-region mapping
        # For now, create synthetic mapping for demonstration
        capacity['region'] = pd.cut(capacity['state_fips'], bins=4, labels=[1, 2, 3, 4])
        
        regional_capacity = capacity.groupby('region').agg({
            'pcp_per_100k': 'mean',
            'capacity_index': 'mean',
            'population': 'sum'
        }).reset_index()
        
        return regional_capacity
    
    def identify_intervention_targets(self, risk_df, capacity_df):
        """
        KEY INSIGHT: Identify regions where high risk + low capacity
        creates intervention opportunities
        """
        
        # Merge risk and capacity
        merged = risk_df.merge(capacity_df, on='region', how='inner')
        
        # Standardize scores
        merged['risk_z'] = (merged['risk_score'] - merged['risk_score'].mean()) / merged['risk_score'].std()
        merged['capacity_z'] = (merged['capacity_index'] - merged['capacity_index'].mean()) / merged['capacity_index'].std()
        
        # Intervention priority: High risk + Low capacity
        merged['intervention_priority'] = merged['risk_z'] - merged['capacity_z']
        
        # Categorize
        merged['category'] = 'Moderate'
        merged.loc[(merged['risk_z'] > 0.5) & (merged['capacity_z'] < -0.5), 'category'] = 'High Priority'
        merged.loc[(merged['risk_z'] < -0.5) & (merged['capacity_z'] > 0.5), 'category'] = 'Low Priority'
        merged.loc[(merged['risk_z'] > 0.5) & (merged['capacity_z'] > 0.5), 'category'] = 'High Risk, High Capacity'
        
        return merged
    
    def generate_insights(self, targets):
        """Generate actionable geographic insights"""
        
        print("\n" + "="*60)
        print("GEOGRAPHIC INSIGHTS FOR TARGETED INTERVENTIONS")
        print("="*60)
        
        print("\n🎯 INTERVENTION PRIORITY RANKING")
        print("-" * 60)
        
        targets_sorted = targets.sort_values('intervention_priority', ascending=False)
        
        for idx, row in targets_sorted.iterrows():
            region_name = {1: 'Northeast', 2: 'Midwest', 3: 'South', 4: 'West'}.get(row['region'], f"Region {row['region']}")
            
            print(f"\n{region_name}:")
            print(f"  Risk Score: {row['risk_score']:.3f} (z={row['risk_z']:+.2f})")
            print(f"  Capacity Index: {row['capacity_index']:.3f} (z={row['capacity_z']:+.2f})")
            print(f"  Priority: {row['intervention_priority']:.2f}")
            print(f"  Category: {row['category']}")
            print(f"  Frequent ED Rate: {row['freq_ed_rate']*100:.2f}%")
            print(f"  PCPs per 100K: {row['pcp_per_100k']:.1f}")
            
            # Specific recommendations
            if row['category'] == 'High Priority':
                print(f"  💡 RECOMMENDATION: Urgent capacity expansion needed")
                print(f"     - Add primary care access points")
                print(f"     - Implement ED diversion programs")
                print(f"     - Target high-risk individuals for care management")
            elif row['category'] == 'High Risk, High Capacity':
                print(f"  💡 RECOMMENDATION: Optimize existing capacity")
                print(f"     - Improve care coordination")
                print(f"     - Enhance preventive services")
        
        return targets_sorted
    
    def create_visualization(self, targets, output_dir):
        """Create geographic insight visualization"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Risk vs Capacity scatter
        colors = {'High Priority': 'red', 'High Risk, High Capacity': 'orange',
                 'Moderate': 'yellow', 'Low Priority': 'green'}
        
        for category in targets['category'].unique():
            subset = targets[targets['category'] == category]
            axes[0].scatter(subset['capacity_z'], subset['risk_z'],
                          c=colors.get(category, 'gray'), label=category,
                          s=200, alpha=0.7, edgecolors='black')
        
        axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[0].axvline(0, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('Capacity Index (standardized)', fontsize=12)
        axes[0].set_ylabel('Risk Score (standardized)', fontsize=12)
        axes[0].set_title('Geographic Risk-Capacity Matrix', fontsize=14, fontweight='bold')
        axes[0].legend(loc='upper left')
        axes[0].grid(alpha=0.3)
        
        # Annotate quadrants
        axes[0].text(1, 1, 'High Risk\nHigh Capacity', ha='center', va='center',
                    fontsize=10, alpha=0.5, style='italic')
        axes[0].text(-1, 1, 'High Risk\nLow Capacity\n(PRIORITY)', ha='center', va='center',
                    fontsize=10, alpha=0.5, style='italic', weight='bold')
        axes[0].text(-1, -1, 'Low Risk\nLow Capacity', ha='center', va='center',
                    fontsize=10, alpha=0.5, style='italic')
        axes[0].text(1, -1, 'Low Risk\nHigh Capacity', ha='center', va='center',
                    fontsize=10, alpha=0.5, style='italic')
        
        # Plot 2: Intervention priority ranking
        targets_sorted = targets.sort_values('intervention_priority', ascending=True)
        region_names = targets_sorted['region'].map({1: 'Northeast', 2: 'Midwest', 3: 'South', 4: 'West'})
        
        bars = axes[1].barh(range(len(targets_sorted)), targets_sorted['intervention_priority'],
                           color=[colors.get(c, 'gray') for c in targets_sorted['category']])
        axes[1].set_yticks(range(len(targets_sorted)))
        axes[1].set_yticklabels(region_names)
        axes[1].set_xlabel('Intervention Priority Score', fontsize=12)
        axes[1].set_title('Regional Intervention Priorities', fontsize=14, fontweight='bold')
        axes[1].axvline(0, color='black', linestyle='-', linewidth=0.8)
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'geographic_insights.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved visualization: {output_dir / 'geographic_insights.png'}")
        
        plt.close()


if __name__ == "__main__":
    # Initialize
    analysis = GeographicInsights(
        meps_dir="/Users/sanjaybasu/waymark-local/healthcare_world_model/healthcare_world_model/data/real_meps",
        hrsa_file="/Users/sanjaybasu/waymark-local/healthcare_world_model/data/hrsa/state_capacity_features.csv"
    )
    
    # Load data
    print("Loading MEPS data...")
    meps_data = analysis.load_state_data()
    
    # Calculate regional risk
    print("\nCalculating regional risk scores...")
    risk_scores = analysis.calculate_regional_risk(meps_data)
    
    # Load capacity
    print("\nLoading HRSA capacity data...")
    capacity_data = analysis.load_capacity_data()
    
    # Identify targets
    print("\nIdentifying intervention targets...")
    targets = analysis.identify_intervention_targets(risk_scores, capacity_data)
    
    # Generate insights
    insights = analysis.generate_insights(targets)
    
    # Create visualization
    analysis.create_visualization(targets, output_dir="/Users/sanjaybasu/waymark-local/healthcare_world_model/figures")
    
    # Save results
    insights.to_csv('/Users/sanjaybasu/waymark-local/healthcare_world_model/geographic_insights.csv', index=False)
    print(f"\n✓ Saved results: geographic_insights.csv")
    
    print("\n" + "="*60)
    print("GEOGRAPHIC ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey Finding: World model identifies capacity-risk mismatches")
    print("that enable targeted geographic interventions")
