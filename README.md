# Healthcare World Model - Reproducibility Repository

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the full implementation of the Recurrent State-Space Model (RSSM) for healthcare systems analysis, as described in:

> **"A Recurrent World Model Enables Counterfactual Planning for Healthcare Systems"**  
> Sanjay Basu<sup>1,2</sup>, Seth Berkowitz<sup>3</sup>  
> <sup>1</sup>Department of Medicine, University of California, San Francisco, California, USA  
> <sup>2</sup>Waymark, San Francisco, California, USA  
> <sup>3</sup>University of North Carolina at Chapel Hill, Chapel Hill, North Carolina, USA  
> *Submitted*

The RSSM learns joint latent representations of individual health trajectories and system-level capacity dynamics, enabling:
- **Superior prediction** during systemic shocks (13.3pp improvement over ensemble baselines)
- **Counterfactual reasoning** for policy planning ("what-if" scenarios)
- **Novel insights** into capacity-demand coupling and intervention cascades

## Quick Start

```bash
# Clone the repository
git clone https://github.com/sanjaybasu/healthcare-world-model.git
cd healthcare-world-model

# Install dependencies
pip install -r requirements.txt

# Run the full reproducibility pipeline
python reproduce_results.py
```

This will:
1. Train the RSSM on MEPS data
2. Validate against COVID-19 natural experiment (MAPE 3.5% vs 16.8% baseline)
3. Validate against Medicaid expansion (heterogeneous effects)
4. Generate manuscript figures (coupling, shock propagation, cascades)

## Repository Structure

```
.
├── rssm_architecture.py       # Neural network definitions (Encoder, Transition, Decoder)
├── rssm_training.py           # Main training loop and dataset classes
├── rssm_data_loader.py        # Data preprocessing utilities
├── rssm_validate.py           # COVID-19 validation script
├── rssm_medicaid_validation.py # Medicaid expansion validation
├── rssm_novel_insights.py     # Generate manuscript figures
├── reproduce_results.py       # Master orchestration script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── data/                      # (Not included - see Data Access below)
```

## Data Access

This analysis uses publicly available data:

### Medical Expenditure Panel Survey (MEPS)
- **Source**: Agency for Healthcare Research and Quality (AHRQ)
- **URL**: https://meps.ahrq.gov
- **Files needed**: 
  - Panels 18-20 (2013-2015) for Medicaid validation
  - Panels 23-26 (2019-2022) for COVID validation
  - Download from MEPS website and place in `data/` directory

### Area Health Resources Files (AHRF)
- **Source**: Health Resources and Services Administration (HRSA)
- **URL**: https://data.hrsa.gov
- **File needed**: AHRF 2023-2024
  
**Note**: Due to data use agreements, we cannot redistribute the raw MEPS data. However, all preprocessing code is provided in `rssm_data_loader.py` and `rssm_medicaid_data_prep.py`.

## System Requirements

### Hardware
- **Minimum**: 16GB RAM, 4-core CPU
- **Recommended**: 32GB RAM, 8-core CPU, NVIDIA GPU (T4 or better)
- **Training time**: ~8 hours on NVIDIA A100, ~24 hours on CPU

### Software
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

## Key Results

| Experiment | RSSM | Baseline | Improvement |
|------------|------|----------|-------------|
| **COVID-19 Shock Prediction** | MAPE 3.5% | MAPE 16.8% | **13.3 pp** |
| **Medicaid (High Expansion)** | RMSE 0.519 | RMSE 0.519 | Competitive |
| **Counterfactual Capability** | ✓ Yes | ✗ No | **Novel** |

### Expected Output

Running `reproduce_results.py` should produce results consistent with the manuscript:

- **COVID-19 MAPE**: ~3.5% (±0.2% due to stochastic training)
- **Baseline MAPE**: ~16.8%
- **Medicaid RMSE (High Expansion)**: ~0.519
- **Medicaid RMSE (Mixed Expansion)**: ~0.765 (demonstrating policy sensitivity)

Key figures will be saved to `results/figures/`. If your results differ substantially, please check:
1. CUDA/random seed configuration
2. MEPS data version (2023-2024 release)
3. Python/PyTorch versions match `requirements.txt`

### Novel Insights

1. **Capacity-Demand Coupling**: Non-linear relationship between system stress and individual utilization (OR 2.3, 95% CI 1.8-2.9)
2. **Shock Propagation**: Recovery time constant τ = 5.2 months for pandemic-scale shocks
3. **Intervention Cascades**: Mobile ED units reduce direct visits by 12% and capacity breaches by 18%

## Model Architecture

```
Individual Encoder (Bi-LSTM, hidden=64)
     ↓
  Latent State z_ind (32-dim)
     ↓                           System Encoder (GRU, hidden=32)
     ├───→ Transition Model ←───┤
     │      (GRU, hidden=64)     ↓
     │            ↓           Latent State z_sys (16-dim)
     │    Prior p(z_t | h_t)    │
     │            ↓              │
     └──→ Decoder (MLP) ←───────┘
              ↓
    Predictions (ED visits, etc.)
```

**Total Parameters**: 298,435

## Citation

If you use this code, please cite:

```bibtex
@article{basu2025healthcare,
  title={A Recurrent World Model Enables Counterfactual Planning for Healthcare Systems},
  author={Basu, Sanjay and Berkowitz, Seth A},
  journal={Submitted},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details

## Contact

**Sanjay Basu, MD, PhD**  
Department of Medicine, University of California, San Francisco & Waymark  
GitHub: [@sanjaybasu](https://github.com/sanjaybasu)

**Seth A Berkowitz, MD, MPH**  
University of North Carolina at Chapel Hill

## Acknowledgements

We thank the AHRQ and HRSA for making MEPS and AHRF data publicly available.
