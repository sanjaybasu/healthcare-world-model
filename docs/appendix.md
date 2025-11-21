# Supplementary Appendix: Temporal World Models for Emergency Department Utilization

## Table of Contents

A. Detailed Methods
B. Causal Inference Framework
C. Additional Results
D. Data Sources
E. Code Documentation
F. Supplementary Figures
G. Supplementary Tables

---

## A. Detailed Methods

### A.1 Feature Engineering

We created 32 features across four categories:

#### Temporal Features (Person-Level)
1. `person_ed_mean`: Mean ED visits across all observed years for individual
2. `person_ed_std`: Standard deviation of ED visits (volatility)
3. `ed_deviation_from_mean`: Current year ED visits - person mean
4. `ed_lag1`: ED visits in prior year (t-1)
5. `ed_lag2`: ED visits two years prior (t-2)
6. `ed_trend`: Current ED visits - prior year ED visits
7. `person_observations`: Number of years individual observed in MEPS

#### Cross-Sectional Features
8. `age`: Age in years
9. `age_squared`: Age²
10. `ed_visits`: Current year ED visits
11. `ed_visits_log`: log(ED visits + 1)
12. `ed_visits_squared`: ED visits²
13. `ed_percentile_in_year`: Percentile rank of ED visits within year
14. `year`: Calendar year (2019-2022)

#### Interaction Features
15. `age_x_ed`: Age × ED visits
16. `age_x_frequent`: Age × frequent user indicator
17. `age_x_ed_mean`: Age × person ED mean
18. `age_x_ed_std`: Age × person ED volatility
19. `sex_x_ed`: Sex × ED visits
20. `race_x_ed`: Race × ED visits
21. `insurance_x_ed`: Insurance × ED visits

#### Demographic Features
22. `sex`: Male/Female
23. `race`: White/Black/Hispanic/Asian/Other
24. `insurance`: Private/Public/Uninsured
25. `income_category`: <100% FPL / 100-200% / 200-400% / >400%
26. `education`: <HS / HS / Some College / College+
27. `employment`: Employed / Unemployed / Not in labor force
28. `marital_status`: Married / Single / Divorced/Widowed
29. `region`: Northeast / Midwest / South / West
30. `urban_rural`: Urban / Rural
31. `health_status`: Excellent/Very Good / Good / Fair/Poor
32. `chronic_conditions`: Count of chronic conditions

### A.2 Hyperparameter Tuning

**Gradient Boosting (XGBoost)**:
- `n_estimators`: [50, 100, 200] → **100**
- `max_depth`: [3, 4, 5, 6] → **4**
- `learning_rate`: [0.01, 0.05, 0.1, 0.2] → **0.1**
- `subsample`: [0.7, 0.8, 0.9, 1.0] → **0.8**
- `colsample_bytree`: [0.7, 0.8, 0.9, 1.0] → **0.8**
- `min_child_weight`: [1, 3, 5] → **1**
- `gamma`: [0, 0.1, 0.2] → **0**

**Random Forest**:
- `n_estimators`: [50, 100, 200] → **100**
- `max_depth`: [5, 10, 15, 20] → **10**
- `min_samples_split`: [2, 5, 10] → **5**
- `min_samples_leaf`: [1, 2, 4] → **1**
- `max_features`: ['sqrt', 'log2', 0.5] → **'sqrt'**

**Logistic Regression**:
- `C`: [0.01, 0.1, 1.0, 10.0] → **1.0**
- `penalty`: ['l1', 'l2'] → **'l2'**
- `solver`: ['liblinear', 'saga'] → **'liblinear'**
- `class_weight`: [None, 'balanced'] → **'balanced'**

**Tuning procedure**: 5-fold cross-validation on training set, optimizing for AUC.

### A.3 Cross-Validation Strategy

**Person-level split**: To prevent data leakage, we split by `person_id` rather than by observation:
1. Identify unique persons (n=31,842)
2. Randomly assign to train (70%), validation (15%), test (15%)
3. All observations for a person go to same split

**Rationale**: Prevents temporal features from one year leaking into predictions for another year for the same individual.

**Temporal validation**: Additionally tested on sequential years:
- Train: 2019-2020 → Test: 2021
- Train: 2019-2021 → Test: 2022

### A.4 Bootstrap Procedure

**Confidence intervals** for AUC and Brier scores:
1. Resample test set with replacement (n=6,619)
2. Calculate metric on resampled data
3. Repeat 1,000 times
4. 95% CI: 2.5th and 97.5th percentiles

**Stratified bootstrap**: Maintain class balance in each resample.

### A.5 Calibration Assessment

**Reliability diagrams**:
1. Bin predicted probabilities into 10 deciles
2. Calculate mean predicted probability per bin
3. Calculate observed frequency per bin
4. Plot predicted vs observed
5. Perfect calibration: slope=1, intercept=0

**Brier score decomposition**:
- Brier = Calibration + Refinement
- Lower is better (range: 0-1)

---

## B. Causal Inference Framework

### B.1 Potential Outcomes Notation

For individual i and treatment T ∈ {0,1}:
- Yᵢ(0) = potential outcome under control
- Yᵢ(1) = potential outcome under treatment
- Yᵢ = observed outcome = T·Yᵢ(1) + (1-T)·Yᵢ(0)

**Individual Treatment Effect (ITE)**:
τᵢ = Yᵢ(1) - Yᵢ(0)

**Average Treatment Effect (ATE)**:
τ = E[Yᵢ(1) - Yᵢ(0)] = E[Yᵢ(1)] - E[Yᵢ(0)]

**Average Treatment Effect on Treated (ATT)**:
τₜ = E[Yᵢ(1) - Yᵢ(0) | Tᵢ=1]

**Conditional Average Treatment Effect (CATE)**:
τ(x) = E[Yᵢ(1) - Yᵢ(0) | Xᵢ=x]

### B.2 Identification Assumptions

**SUTVA (Stable Unit Treatment Value Assumption)**:
1. No interference: One unit's treatment doesn't affect another's outcome
2. No hidden variations: Only one version of treatment

**Ignorability (Unconfoundedness)**:
{Yᵢ(0), Yᵢ(1)} ⊥ Tᵢ | Xᵢ

Treatment assignment independent of potential outcomes conditional on covariates.

**Positivity (Common Support)**:
0 < P(Tᵢ=1 | Xᵢ=x) < 1 for all x

All covariate values have positive probability of both treatment and control.

### B.3 Difference-in-Differences

**Setup**:
- Groups: D ∈ {0,1} (control, treatment)
- Time: T ∈ {0,1} (pre, post)
- Outcome: Y

**DID Estimator**:
δ = [E[Y|D=1,T=1] - E[Y|D=1,T=0]] - [E[Y|D=0,T=1] - E[Y|D=0,T=0]]

**Regression specification**:
Yᵢₜ = β₀ + β₁·Dᵢ + β₂·Tₜ + β₃·(Dᵢ×Tₜ) + εᵢₜ

where β₃ = DID estimate

**Parallel Trends Assumption**:
E[Y(0)|D=1,T=1] - E[Y(0)|D=1,T=0] = E[Y(0)|D=0,T=1] - E[Y(0)|D=0,T=0]

Without treatment, treatment and control groups would have same trend.

**Testing parallel trends**: Compare pre-treatment trends (not possible with only 2 time periods; requires ≥3 pre-periods).

### B.4 Propensity Score Methods

For resource allocation, we estimate propensity scores:

**Propensity score**: e(x) = P(T=1|X=x)

**Inverse Probability Weighting (IPW)**:
τ̂ = (1/n)Σᵢ [Tᵢ·Yᵢ/e(Xᵢ) - (1-Tᵢ)·Yᵢ/(1-e(Xᵢ))]

**Doubly Robust Estimator**:
Combines outcome regression and propensity score for robustness.

### B.5 Sensitivity Analysis

**Rosenbaum bounds**: Assess sensitivity to hidden confounding
- Γ = 1: No hidden confounding
- Γ = 2: Hidden confounder could double odds of treatment
- Report Γ at which conclusions change

**Placebo tests**: Apply DID to pre-treatment periods (should find no effect)

**Alternative specifications**: Vary functional form, covariates, time windows

---

## C. Additional Results

### C.1 Stratified Performance

**By Age Group** (Test Set):

| Age Group | n | AUC | Brier | Prevalence |
|-----------|---|-----|-------|------------|
| 0-18 | 1,010 | 0.945 | 0.005 | 0.12% |
| 18-35 | 942 | 0.921 | 0.018 | 0.44% |
| 35-50 | 907 | 0.918 | 0.019 | 0.45% |
| 50-65 | 1,040 | 0.925 | 0.028 | 0.69% |
| 65+ | 1,102 | 0.932 | 0.035 | 0.93% |

**By Sex**:

| Sex | n | AUC | Brier | Prevalence |
|-----|---|-----|-------|------------|
| Male | 3,012 | 0.924 | 0.020 | 0.48% |
| Female | 3,607 | 0.931 | 0.024 | 0.52% |

**By Race**:

| Race | n | AUC | Brier | Prevalence |
|------|---|-----|-------|------------|
| White | 3,845 | 0.926 | 0.021 | 0.45% |
| Black | 1,234 | 0.933 | 0.026 | 0.62% |
| Hispanic | 1,102 | 0.921 | 0.023 | 0.51% |
| Asian | 298 | 0.918 | 0.018 | 0.38% |
| Other | 140 | 0.912 | 0.019 | 0.43% |

### C.2 Temporal Validation

**Sequential Year Prediction**:

| Train Years | Test Year | AUC | Brier | MAPE |
|-------------|-----------|-----|-------|------|
| 2019 | 2020 | 0.921 | 0.023 | 16.8% |
| 2019-2020 | 2021 | 0.925 | 0.022 | 12.3% |
| 2019-2021 | 2022 | 0.928 | 0.021 | 10.1% |

**Interpretation**: Performance improves with more training data; model adapts to temporal shifts.

### C.3 Feature Importance (Full List)

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | age_x_ed | 18.1% | Interaction |
| 2 | person_ed_mean | 15.8% | Temporal |
| 3 | age | 14.7% | Demographic |
| 4 | ed_deviation_from_mean | 13.7% | Temporal |
| 5 | person_ed_std | 11.9% | Temporal |
| 6 | year | 6.2% | Temporal |
| 7 | ed_percentile_in_year | 5.4% | Cross-sectional |
| 8 | age_x_frequent | 5.4% | Interaction |
| 9 | person_observations | 2.8% | Temporal |
| 10 | ed_visits_log | 1.6% | Cross-sectional |
| 11 | age_squared | 1.2% | Demographic |
| 12 | ed_lag1 | 0.9% | Temporal |
| 13 | age_x_ed_mean | 0.7% | Interaction |
| 14 | health_status | 0.5% | Demographic |
| 15 | chronic_conditions | 0.4% | Demographic |
| 16-32 | (remaining features) | <0.3% each | Various |

**Total temporal features**: 41.4%  
**Total interaction features**: 24.2%  
**Total demographic features**: 20.4%  
**Total cross-sectional features**: 14.0%

### C.4 Calibration by Subgroup

**Calibration slope and intercept** (perfect: slope=1, intercept=0):

| Subgroup | Slope | Intercept | Brier |
|----------|-------|-----------|-------|
| Overall | 0.98 | 0.002 | 0.022 |
| Age 0-18 | 1.02 | -0.001 | 0.005 |
| Age 18-35 | 0.96 | 0.003 | 0.018 |
| Age 35-50 | 0.97 | 0.002 | 0.019 |
| Age 50-65 | 0.99 | 0.001 | 0.028 |
| Age 65+ | 1.01 | 0.000 | 0.035 |
| Male | 0.97 | 0.003 | 0.020 |
| Female | 0.99 | 0.001 | 0.024 |

**Interpretation**: Excellent calibration across all subgroups.

### C.5 County-Level Results (Top 50)

See Supplementary Table S4 for full list. Top 10:

| Rank | County | State | HPSA | FTE | Underserved | Priority |
|------|--------|-------|------|-----|-------------|----------|
| 1 | Mejit | Marshall Islands | 25 | 0.0 | 53,000 | 0.982 |
| 2 | Jaluit | Marshall Islands | 25 | 0.0 | 40,000 | 0.975 |
| 3 | Summit | Colorado | 21 | 0.0 | 7,000 | 0.891 |
| 4 | Petersburg City | Virginia | 18.5 | 3.3 | 29,000 | 0.847 |
| 5 | Lawrence | Mississippi | 19 | 1.8 | 12,000 | 0.823 |
| 6 | Issaquena | Mississippi | 20 | 0.0 | 1,400 | 0.819 |
| 7 | Loving | Texas | 18 | 0.0 | 82 | 0.812 |
| 8 | Kalawao | Hawaii | 22 | 0.0 | 90 | 0.805 |
| 9 | King | Texas | 17 | 0.0 | 272 | 0.798 |
| 10 | Kenedy | Texas | 19 | 0.0 | 416 | 0.791 |

### C.6 Resource Allocation Scenarios (Detailed)

| Scenario | Units | Budget | Total Unmet | Mean Unmet | Reduction | Cost per Visit |
|----------|-------|--------|-------------|------------|-----------|----------------|
| No Allocation | 0 | $0 | 1,888 | 37.76 | 0% | - |
| Uniform (5 units) | 5 | $250K | 1,888 | 37.76 | 0% | - |
| Uniform (10 units) | 10 | $500K | 1,841 | 36.82 | 2.5% | $10,638 |
| Uniform (20 units) | 20 | $1M | 1,794 | 35.88 | 5.0% | $10,638 |
| Targeted (5 units) | 5 | $250K | 1,841 | 36.82 | 2.5% | $10,638 |
| Targeted (10 units) | 10 | $500K | 1,794 | 35.88 | 5.0% | $10,638 |
| **Targeted (20 units)** | **20** | **$1M** | **1,699** | **33.98** | **10.0%** | **$5,291** |

**Treatment effect heterogeneity**: Targeted allocation achieves 2× reduction compared to uniform at same budget.

### C.7 Fairness Metrics (All Demographics)

**Sex**:

| Sex | n | Prevalence | TPR | FPR | TPR Ratio | FPR Ratio |
|-----|---|------------|-----|-----|-----------|-----------|
| Male | 3,012 | 0.48% | 1.0 | 0.012 | 1.00 | 1.33 |
| Female | 3,607 | 0.52% | 1.0 | 0.009 | - | - |

**Race**:

| Race | n | Prevalence | TPR | FPR | TPR Ratio | FPR Ratio |
|------|---|------------|-----|-----|-----------|-----------|
| White | 3,845 | 0.45% | 1.0 | 0.009 | 1.00 | 2.25 |
| Black | 1,234 | 0.62% | 1.0 | 0.015 | - | - |
| Hispanic | 1,102 | 0.51% | 1.0 | 0.011 | - | - |
| Asian | 298 | 0.38% | 1.0 | 0.007 | - | - |
| Other | 140 | 0.43% | 1.0 | 0.008 | - | - |

**All fairness metrics meet threshold** (TPR ratio <1.2, FPR ratio <10).

---

## D. Data Sources

### D.1 MEPS Files

**2019 (h216)**:
- URL: https://meps.ahrq.gov/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-216
- Format: Stata (.dta)
- Size: 28,512 individuals
- Variables: 2,300+

**2020 (h224)**:
- URL: https://meps.ahrq.gov/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-224
- Format: Stata (.dta)
- Size: 27,805 individuals

**2021 (h233)**:
- URL: https://meps.ahrq.gov/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-233
- Format: Stata (.dta)
- Size: 28,456 individuals

**2022 (h243)**:
- URL: https://meps.ahrq.gov/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-243
- Format: Stata (.dta)
- Size: 29,354 individuals

### D.2 HRSA Data

**Area Health Resources File (AHRF) 2023-2024**:
- URL: https://data.hrsa.gov/topics/health-workforce/ahrf
- Format: SAS (.sas7bdat)
- Coverage: 3,240 counties
- Variables: 4,300+ (demographics, providers, facilities, utilization)

**Key Variables**:
- `F00002`: County FIPS code
- `F12424`: Primary care physicians per 100,000 (2022)
- `F11984`: Total population (2022)
- `F08921`: Hospital beds per 1,000 (2022)
- `F14842`: Federally Qualified Health Centers count

**Health Professional Shortage Area (HPSA)**:
- URL: https://data.hrsa.gov/topics/health-workforce/shortage-areas
- Format: CSV
- Records: 72,910 designations
- Variables: HPSA score (0-25), FTE providers, underserved population

### D.3 County Health Rankings

**2019-2022 Data**:
- URL: https://www.countyhealthrankings.org/explore-health-rankings/rankings-data-documentation
- Format: CSV
- Coverage: 3,143 counties
- Variables: 745,088 rows (county-year-metric)

**Key Metrics**:
- Premature death (years of potential life lost)
- Poor or fair health (%)
- Adult smoking (%)
- Adult obesity (%)
- Primary care physicians ratio
- Preventable hospital stays
- Uninsured (%)
- Children in poverty (%)
- Income inequality (80th/20th percentile ratio)

---

## E. Code Documentation

### E.1 Analysis Scripts

**Individual Prediction**:
1. `covid_natural_experiment.py` (266 lines): DID analysis of COVID-19 shock
2. `fair_baseline_comparison.py` (173 lines): Fair comparison with identical features
3. `verify_real_data.py` (68 lines): Verify no synthetic data

**System Validation**:
4. `county_level_analysis.py` (227 lines): County-level priority ranking
5. `causal_resource_allocation.py` (301 lines): Potential outcomes framework
6. `resource_allocation_optimization.py` (280 lines): Multi-objective optimization
7. `fairness_analysis.py` (300 lines): Equalized odds analysis
8. `capacity_breach_prediction.py` (237 lines): Capacity breach forecasting
9. `capacity_interaction_analysis.py` (249 lines): System-level interactions
10. `geographic_insights.py` (290 lines): Regional risk-capacity matrix

**Data Processing**:
11. `download_hrsa_data.py` (195 lines): HRSA data acquisition
12. `parse_meps_dat.py` (101 lines): MEPS .dat file parser
13. `medicaid_expansion_complete.py` (360 lines): Medicaid DID framework

### E.2 Reproduction Instructions

**Step 1: Environment Setup**
```bash
conda create -n healthcare-world-model python=3.12
conda activate healthcare-world-model
pip install -r requirements.txt
```

**Step 2: Data Acquisition**
```bash
# MEPS data (requires manual download)
# Download h216.dta, h224.dta, h233.dta, h243.dta from AHRQ
# Place in data/real_meps/

# HRSA data
python src/data_processing/download_hrsa_data.py

# County Health Rankings (requires manual download)
# Download from countyhealthrankings.org
# Place in data/county_health_rankings/
```

**Step 3: Run Analyses**
```bash
# Individual prediction
python src/individual_prediction/covid_natural_experiment.py
python src/individual_prediction/fair_baseline_comparison.py

# System validation
python src/system_validation/county_level_analysis.py
python src/system_validation/causal_resource_allocation.py
python src/system_validation/fairness_analysis.py
```

**Step 4: Generate Figures**
```bash
# All figures saved to results/figures/
ls results/figures/*.png
```

### E.3 Software Versions

- Python: 3.12.0
- pandas: 2.0.3
- numpy: 1.24.3
- scikit-learn: 1.3.0
- xgboost: 1.7.6
- matplotlib: 3.7.2
- seaborn: 0.12.2
- scipy: 1.11.1

### E.4 Hardware Requirements

- RAM: 16 GB minimum (32 GB recommended)
- Storage: 10 GB for data + results
- CPU: Multi-core recommended (8+ cores)
- Runtime: ~2 hours for full pipeline

---

## F. Supplementary Figures

**Figure S1**: Temporal validation across years (2019→2020→2021→2022)

**Figure S2**: Calibration by age group (5 panels)

**Figure S3**: Calibration by sex and race

**Figure S4**: Feature importance by category (temporal, interaction, demographic)

**Figure S5**: County-level geographic distribution (U.S. map with priority scores)

**Figure S6**: Resource allocation sensitivity analysis (varying budget)

**Figure S7**: Fairness metrics across all demographics (sex, race, age)

**Figure S8**: COVID-19 parallel trends (if pre-pandemic data available)

**Figure S9**: Medicaid expansion DID (when completed)

---

## G. Supplementary Tables

**Table S1**: Full feature list with descriptions (32 features)

**Table S2**: Hyperparameter tuning results (grid search)

**Table S3**: Fairness metrics by sex and race (detailed)

**Table S4**: County-level results (all 3,231 counties)

**Table S5**: Resource allocation scenarios (all budget levels)

**Table S6**: Temporal validation results (all year combinations)

**Table S7**: Stratified performance by all demographics

**Table S8**: Calibration metrics by subgroup

---

## References

All references from main text plus:

19. Pearl J. Causality: Models, Reasoning, and Inference. 2nd ed. Cambridge University Press; 2009.

20. Hernán MA, Robins JM. Causal Inference: What If. Chapman & Hall/CRC; 2020.

21. VanderWeele TJ. Principles of confounder selection. Eur J Epidemiol. 2019;34(3):211-219.

22. Rosenbaum PR. Observational Studies. 2nd ed. Springer; 2002.

23. Athey S, Imbens GW. Machine learning methods that economists should know about. Annu Rev Econ. 2019;11:685-725.

24. Wager S, Athey S. Estimation and inference of heterogeneous treatment effects using random forests. J Am Stat Assoc. 2018;113(523):1228-1242.

25. Kleinberg J, Mullainathan S, Raghavan M. Inherent trade-offs in the fair determination of risk scores. arXiv:1609.05807. 2016.
