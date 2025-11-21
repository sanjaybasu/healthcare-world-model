"""
COVID-19 Natural Experiment Validation
Tests world model's ability to predict 2020 ED demand shock
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

class COVIDNaturalExperiment:
    """
    Natural experiment: COVID-19 pandemic demand shock (2020)
    
    Test: Can world model trained on 2019 predict 2020 behavior change?
    Hypothesis: System-aware model captures demand shock better than individual-only
    """
    
    def __init__(self, meps_dir):
        self.meps_dir = Path(meps_dir)
        self.results = {}
        
    def load_pre_post_data(self):
        """Load 2019 (pre-COVID) and 2020 (COVID) MEPS data"""
        print("="*60)
        print("COVID-19 NATURAL EXPERIMENT")
        print("="*60)
        
        # 2019: Pre-COVID baseline
        df_2019 = pd.read_stata(self.meps_dir / "h216.dta")
        
        # 2020: COVID impact
        df_2020 = pd.read_stata(self.meps_dir / "h224.dta")
        
        # Extract key variables
        data_2019 = self._extract_features(df_2019, year=2019)
        data_2020 = self._extract_features(df_2020, year=2020)
        
        print(f"\n2019 (Pre-COVID): {len(data_2019):,} individuals")
        print(f"2020 (COVID): {len(data_2020):,} individuals")
        
        return data_2019, data_2020
    
    def _extract_features(self, df, year):
        """Extract consistent features across years"""
        
        # Person ID
        person_id = df.get('DUPERSID', df.get('DUID'))
        
        # ED visits
        ed_col = f'ERTOT{str(year)[2:]}'
        ed_visits = pd.to_numeric(df.get(ed_col, 0), errors='coerce').fillna(0)
        
        # Demographics
        age = pd.to_numeric(df.get('AGEDX', df.get(f'AGE{str(year)[2:]}X', 50)), errors='coerce')
        
        # Insurance
        insurance = pd.to_numeric(df.get(f'INS{str(year)[2:]}X', 1), errors='coerce')
        
        # Chronic conditions (if available)
        chronic = pd.to_numeric(df.get(f'CHOLDX', 0), errors='coerce')
        
        # Region
        region = pd.to_numeric(df.get(f'REGION{str(year)[2:]}', df.get('REGION', 0)), errors='coerce')
        
        return pd.DataFrame({
            'person_id': person_id,
            'year': year,
            'ed_visits': ed_visits,
            'age': age,
            'insurance': insurance,
            'chronic': chronic,
            'region': region,
            'frequent_ed': (ed_visits >= 4).astype(int)
        })
    
    def analyze_demand_shock(self, data_2019, data_2020):
        """Quantify COVID-19 impact on ED utilization"""
        
        print("\n" + "="*60)
        print("DEMAND SHOCK ANALYSIS")
        print("="*60)
        
        # Overall ED visit rates
        rate_2019 = data_2019['ed_visits'].mean()
        rate_2020 = data_2020['ed_visits'].mean()
        
        pct_change = ((rate_2020 - rate_2019) / rate_2019) * 100
        
        print(f"\nED Visits per Person:")
        print(f"  2019: {rate_2019:.3f}")
        print(f"  2020: {rate_2020:.3f}")
        print(f"  Change: {pct_change:+.1f}%")
        
        # Frequent user rates
        freq_2019 = data_2019['frequent_ed'].mean()
        freq_2020 = data_2020['frequent_ed'].mean()
        
        print(f"\nFrequent ED Users (≥4 visits):")
        print(f"  2019: {freq_2019*100:.2f}%")
        print(f"  2020: {freq_2020*100:.2f}%")
        
        # By age group
        print(f"\nBy Age Group:")
        for age_group in [(0, 18), (18, 35), (35, 50), (50, 65), (65, 120)]:
            mask_2019 = (data_2019['age'] >= age_group[0]) & (data_2019['age'] < age_group[1])
            mask_2020 = (data_2020['age'] >= age_group[0]) & (data_2020['age'] < age_group[1])
            
            rate_19 = data_2019[mask_2019]['ed_visits'].mean()
            rate_20 = data_2020[mask_2020]['ed_visits'].mean()
            change = ((rate_20 - rate_19) / rate_19) * 100 if rate_19 > 0 else 0
            
            print(f"  Age {age_group[0]}-{age_group[1]}: {rate_19:.3f} → {rate_20:.3f} ({change:+.1f}%)")
        
        self.results['demand_shock'] = {
            'rate_2019': rate_2019,
            'rate_2020': rate_2020,
            'pct_change': pct_change
        }
        
        return self.results['demand_shock']
    
    def test_prediction_accuracy(self, data_2019, data_2020, model):
        """
        Test: Train on 2019, predict 2020
        Compare predicted vs actual 2020 rates
        """
        
        print("\n" + "="*60)
        print("PREDICTION ACCURACY TEST")
        print("="*60)
        
        # Train on 2019
        features = ['age', 'insurance', 'chronic', 'region']
        X_train = data_2019[features].fillna(0)
        y_train = data_2019['frequent_ed']
        
        print(f"\nTraining on 2019 data: {len(X_train):,} samples")
        model.fit(X_train, y_train)
        
        # Predict 2020
        X_test = data_2020[features].fillna(0)
        y_test = data_2020['frequent_ed']
        y_pred = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        auc = roc_auc_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_pred)
        
        # Aggregate prediction vs reality
        pred_rate = y_pred.mean()
        actual_rate = y_test.mean()
        error = abs(pred_rate - actual_rate)
        mape = (error / actual_rate) * 100
        
        print(f"\nIndividual-Level Prediction:")
        print(f"  AUC: {auc:.3f}")
        print(f"  Brier: {brier:.3f}")
        
        print(f"\nAggregate Rate Prediction:")
        print(f"  Predicted 2020 rate: {pred_rate*100:.2f}%")
        print(f"  Actual 2020 rate: {actual_rate*100:.2f}%")
        print(f"  Absolute error: {error*100:.2f} pp")
        print(f"  MAPE: {mape:.1f}%")
        
        self.results['prediction'] = {
            'auc': auc,
            'brier': brier,
            'pred_rate': pred_rate,
            'actual_rate': actual_rate,
            'mape': mape
        }
        
        return self.results['prediction']
    
    def compare_static_vs_adaptive(self, data_2019, data_2020):
        """
        Compare:
        - Static model: Assumes 2020 = 2019 (no adaptation)
        - Adaptive model: Learns from 2019, predicts 2020 change
        """
        
        print("\n" + "="*60)
        print("STATIC VS ADAPTIVE COMPARISON")
        print("="*60)
        
        actual_2020 = data_2020['ed_visits'].mean()
        
        # Static model: Predict 2020 = 2019
        static_pred = data_2019['ed_visits'].mean()
        static_error = abs(static_pred - actual_2020)
        static_mape = (static_error / actual_2020) * 100
        
        print(f"\nStatic Model (2020 = 2019):")
        print(f"  Prediction: {static_pred:.3f}")
        print(f"  Actual: {actual_2020:.3f}")
        print(f"  Error: {static_error:.3f}")
        print(f"  MAPE: {static_mape:.1f}%")
        
        # Adaptive model: Use world model prediction
        # (This would use the trained model's aggregate prediction)
        # For now, use the prediction from test_prediction_accuracy
        if 'prediction' in self.results:
            adaptive_pred = self.results['prediction']['pred_rate'] * data_2020['ed_visits'].max()
            adaptive_error = abs(adaptive_pred - actual_2020)
            adaptive_mape = (adaptive_error / actual_2020) * 100
            
            print(f"\nAdaptive Model (World Model):")
            print(f"  Prediction: {adaptive_pred:.3f}")
            print(f"  Actual: {actual_2020:.3f}")
            print(f"  Error: {adaptive_error:.3f}")
            print(f"  MAPE: {adaptive_mape:.1f}%")
            
            improvement = static_mape - adaptive_mape
            print(f"\nImprovement: {improvement:.1f} pp MAPE reduction")
        
        return {
            'static_mape': static_mape,
            'adaptive_mape': adaptive_mape if 'prediction' in self.results else None
        }
    
    def create_visualization(self, data_2019, data_2020, output_dir):
        """Create figures showing COVID impact"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Figure 1: ED visit distribution shift
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 2019 distribution
        axes[0].hist(data_2019['ed_visits'], bins=20, alpha=0.7, label='2019', color='blue')
        axes[0].set_xlabel('ED Visits')
        axes[0].set_ylabel('Count')
        axes[0].set_title('2019 (Pre-COVID)')
        axes[0].axvline(data_2019['ed_visits'].mean(), color='red', linestyle='--', label=f'Mean: {data_2019["ed_visits"].mean():.2f}')
        axes[0].legend()
        
        # 2020 distribution
        axes[1].hist(data_2020['ed_visits'], bins=20, alpha=0.7, label='2020', color='orange')
        axes[1].set_xlabel('ED Visits')
        axes[1].set_ylabel('Count')
        axes[1].set_title('2020 (COVID)')
        axes[1].axvline(data_2020['ed_visits'].mean(), color='red', linestyle='--', label=f'Mean: {data_2020["ed_visits"].mean():.2f}')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'covid_ed_distribution_shift.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {output_dir / 'covid_ed_distribution_shift.png'}")
        
        plt.close()


if __name__ == "__main__":
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Initialize
    experiment = COVIDNaturalExperiment(
        meps_dir="healthcare_world_model/data/real_meps"  # Fixed path
    )
    
    # Load data
    data_2019, data_2020 = experiment.load_pre_post_data()
    
    # Analyze demand shock
    shock_results = experiment.analyze_demand_shock(data_2019, data_2020)
    
    # Test prediction
    world_model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    pred_results = experiment.test_prediction_accuracy(data_2019, data_2020, world_model)
    
    # Compare static vs adaptive
    comparison = experiment.compare_static_vs_adaptive(data_2019, data_2020)
    
    # Create visualizations
    experiment.create_visualization(data_2019, data_2020, 
                                   output_dir="healthcare_world_model/figures")
    
    print("\n" + "="*60)
    print("COVID-19 NATURAL EXPERIMENT COMPLETE")
    print("="*60)
    print(f"\nKey Finding: World model {'captures' if comparison.get('adaptive_mape', 100) < comparison['static_mape'] else 'struggles with'} COVID demand shock")
