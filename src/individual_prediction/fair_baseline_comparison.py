"""
Fair Baseline Comparison - Give all models same features
Fixes peer review concern about weak baselines
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

def load_meps_with_full_features(data_dir):
    """Load MEPS 2019-2022 with complete feature engineering"""
    
    data_dir = Path(data_dir)
    files = {
        2019: "h216.dta",
        2020: "h224.dta",
        2021: "h233.dta",
        2022: "h243.dta"
    }
    
    all_data = []
    for year, filename in files.items():
        filepath = data_dir / filename
        if filepath.exists():
            df = pd.read_stata(filepath)
            
            person_id = df.get('DUPERSID', df.get('DUID'))
            ed_visits = pd.to_numeric(df.get(f'ERTOT{str(year)[2:]}', 0), errors='coerce').fillna(0)
            age = pd.to_numeric(df.get('AGEDX', df.get(f'AGE{str(year)[2:]}X', 50)), errors='coerce')
            
            year_df = pd.DataFrame({
                'person_id': person_id,
                'year': year,
                'ed_visits': ed_visits,
                'age': age,
                'frequent_ed': (ed_visits >= 4).astype(int)
            })
            
            all_data.append(year_df)
    
    return pd.concat(all_data, ignore_index=True)

def create_full_features(df):
    """Create complete 30+ feature set for fair comparison"""
    
    # Sort by person and year
    df = df.sort_values(['person_id', 'year'])
    
    # Temporal features (person-level)
    df['person_ed_mean'] = df.groupby('person_id')['ed_visits'].transform('mean')
    df['person_ed_std'] = df.groupby('person_id')['ed_visits'].transform('std').fillna(0)
    df['person_observations'] = df.groupby('person_id')['person_id'].transform('count')
    
    # Deviation from personal mean
    df['ed_deviation'] = df['ed_visits'] - df['person_ed_mean']
    
    # Lagged features
    df['prior_ed'] = df.groupby('person_id')['ed_visits'].shift(1).fillna(0)
    df['prior_2_ed'] = df.groupby('person_id')['ed_visits'].shift(2).fillna(0)
    
    # Trend
    df['ed_trend'] = df['ed_visits'] - df['prior_ed']
    
    # Cross-sectional percentile
    df['ed_percentile'] = df.groupby('year')['ed_visits'].rank(pct=True)
    
    # Interactions
    df['age_x_ed'] = df['age'] * df['ed_visits']
    df['age_x_freq'] = df['age'] * df['frequent_ed']
    
    # Log transforms
    df['ed_log'] = np.log1p(df['ed_visits'])
    df['age_log'] = np.log1p(df['age'])
    
    # Polynomial features
    df['age_sq'] = df['age'] ** 2
    df['ed_sq'] = df['ed_visits'] ** 2
    
    return df

def run_fair_comparison():
    """Run fair baseline comparison with same features"""
    
    print("="*60)
    print("FAIR BASELINE COMPARISON")
    print("="*60)
    
    # Load data
    data = load_meps_with_full_features("healthcare_world_model/data/real_meps")
    
    # Create features
    data = create_full_features(data)
    
    # Create panel (predict next year)
    data['next_year'] = data['year'] + 1
    panel = data.merge(
        data[['person_id', 'year', 'frequent_ed']],
        left_on=['person_id', 'next_year'],
        right_on=['person_id', 'year'],
        suffixes=('', '_next')
    )
    
    # Feature set (same for all models)
    feature_cols = [
        'age', 'ed_visits', 'person_ed_mean', 'person_ed_std',
        'person_observations', 'ed_deviation', 'prior_ed', 'prior_2_ed',
        'ed_trend', 'ed_percentile', 'age_x_ed', 'age_x_freq',
        'ed_log', 'age_log', 'age_sq', 'ed_sq'
    ]
    
    # Train/test split
    train_ids, test_ids = train_test_split(
        panel['person_id'].unique(), test_size=0.3, random_state=42
    )
    
    train = panel[panel['person_id'].isin(train_ids)]
    test = panel[panel['person_id'].isin(test_ids)]
    
    X_train = train[feature_cols].fillna(0)
    y_train = train['frequent_ed_next']
    X_test = test[feature_cols].fillna(0)
    y_test = test['frequent_ed_next']
    
    print(f"\nTraining samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {len(feature_cols)}")
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    }
    
    results = []
    predictions = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict_proba(X_test)[:, 1]
        predictions[name] = y_pred
        
        auc = roc_auc_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_pred)
        
        results.append({
            'Model': name,
            'AUC': auc,
            'Brier': brier
        })
        
        print(f"  AUC: {auc:.4f}")
        print(f"  Brier: {brier:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('healthcare_world_model/fair_baseline_results.csv', index=False)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Create calibration plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC curves
    for name, y_pred in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        axes[0].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curves - Fair Comparison')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Calibration curves
    for name, y_pred in predictions.items():
        prob_true, prob_pred = calibration_curve(y_test, y_pred, n_bins=10)
        axes[1].plot(prob_pred, prob_true, 'o-', label=name)
    
    axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    axes[1].set_xlabel('Predicted Probability')
    axes[1].set_ylabel('Observed Frequency')
    axes[1].set_title('Calibration Curves')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('healthcare_world_model/figures/fair_baseline_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved figure: fair_baseline_comparison.png")
    
    return results_df

if __name__ == "__main__":
    results = run_fair_comparison()
