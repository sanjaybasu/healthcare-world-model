# Data Acquisition Guide

This directory contains instructions for acquiring all data sources used in the analysis.

## Required Data Sources

### 1. MEPS (Medical Expenditure Panel Survey)
### 2. HRSA AHRF (Area Health Resources File)
### 3. County Health Rankings

---

## 1. MEPS Data (Manual Download Required)

**Source**: Agency for Healthcare Research and Quality (AHRQ)  
**URL**: https://meps.ahrq.gov/data_stats/download_data_files.jsp

### Files Needed

| Year | File | URL |
|------|------|-----|
| 2019 | h216.dta | https://meps.ahrq.gov/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-216 |
| 2020 | h224.dta | https://meps.ahrq.gov/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-224 |
| 2021 | h233.dta | https://meps.ahrq.gov/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-233 |
| 2022 | h243.dta | https://meps.ahrq.gov/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-243 |

### Download Instructions

1. Visit https://meps.ahrq.gov/data_stats/download_data_files.jsp
2. For each year:
   - Click on "Full Year Consolidated Data Files"
   - Find the corresponding file (e.g., "2019 Full Year Consolidated Data File")
   - Click "Stata File" to download
3. Place downloaded files in `data/meps/`

### Key Variables

- `DUPERSID`: Person ID
- `PANEL`: Panel number
- `ERTOT{YY}`: Total ED visits
- `AGE{YY}X`: Age
- `SEX`: Sex
- `RACETHX`: Race/ethnicity
- `REGION{YY}`: Census region
- `INSCOV{YY}`: Insurance coverage

---

## 2. HRSA AHRF Data (Auto-Download)

**Source**: Health Resources and Services Administration  
**URL**: https://data.hrsa.gov/topics/health-workforce/ahrf

### Automatic Download

```bash
python src/data_processing/download_hrsa_data.py
```

---

## 3. County Health Rankings (Manual Download Required)

**Source**: University of Wisconsin Population Health Institute  
**URL**: https://www.countyhealthrankings.org/

### Download Instructions

1. Visit https://www.countyhealthrankings.org/explore-health-rankings/rankings-data-documentation
2. For each year (2019-2022):
   - Click "National Data & Documentation"
   - Download "Analytic Data" CSV
3. Place downloaded files in `data/county_health_rankings/`

---

## Data Verification

After downloading all data, verify with:

```bash
python src/individual_prediction/verify_real_data.py
```

---

## Storage Requirements

- MEPS: ~500 MB (4 files)
- HRSA: ~200 MB
- County Health Rankings: ~50 MB (4 files)
- **Total**: ~750 MB

---

## License & Terms of Use

All data sources are public domain. Citation required.
