# Intelligent Decision Support System for Drug Inventory Management: A Robust Ensemble Approach in Small Data Regimes

## Abstract

This repository contains the implementation of an intelligent decision support system for pharmaceutical inventory management using a robust ensemble LSTM (Long Short-Term Memory) approach. The system addresses the challenge of demand forecasting in small data regimes, which is particularly relevant for community pharmacies operating with limited historical data. The proposed ensemble methodology combines multiple LSTM models with heterogeneous graph neural networks to capture complex temporal patterns and relational dependencies between drugs, prescribers, and patients.

## Overview

This project implements a comprehensive forecasting framework that:

- Utilizes an ensemble of five LSTM models with robust aggregation (median) to improve prediction stability
- Incorporates heterogeneous graph structures to model relationships between drugs, doctors, and patients
- Applies ABC analysis for intelligent drug selection and prioritization
- Provides comprehensive managerial insights including financial risk analysis and automation readiness assessment
- Achieves a Weighted Mean Absolute Percentage Error (WMAPE) of 17.70% on test data

## Dataset

This study utilizes a fully anonymized public dataset available on GitHub:

**Source:** [Anonymised Community Pharmacy Prescription Dataset (2018–2020)](https://github.com/farvaresh/pharmacy-prescription-demand-dataset)

The dataset contains anonymized prescription records from a single community pharmacy covering the period from March 2018 to October 2020. All patient and prescriber identifiers have been replaced with pseudonyms, ensuring complete privacy compliance.

### Data Preparation

**Important:** Before running the model, the dataset files must be concatenated into a single CSV file. The original dataset repository contains multiple CSV files that need to be combined before loading.

To prepare the data:

1. Download all CSV files from the [dataset repository](https://github.com/farvaresh/pharmacy-prescription-demand-dataset)
2. Concatenate all CSV files into a single file named `anonymized_prescription.csv`
3. Ensure the concatenated file maintains the column structure as described in the dataset documentation

Example Python code for concatenation:

```python
import pandas as pd
import glob

# Load all CSV files from the data directory
csv_files = glob.glob('path/to/dataset/data/*.csv')

# Concatenate all files
df_list = []
for file in csv_files:
    df = pd.read_csv(file, encoding='latin1')
    df_list.append(df)

# Combine into single dataframe
df_combined = pd.concat(df_list, ignore_index=True)

# Save as single file
df_combined.to_csv('anonymized_prescription.csv', index=False, encoding='latin1')
```

## Requirements

### Python Version
- Python 3.8 or higher

### Required Libraries

Install the required packages using pip:

```bash
pip install torch torch-geometric pandas numpy scikit-learn matplotlib seaborn scipy jdatetime
```

Or install from the requirements file (if provided):

```bash
pip install -r requirements.txt
```

### Key Dependencies

- **PyTorch** (≥2.0.0): Deep learning framework for LSTM implementation
- **torch-geometric** (≥2.0.0): Graph neural network library for heterogeneous graph processing
- **pandas** (≥1.3.0): Data manipulation and analysis
- **numpy** (≥1.21.0): Numerical computing
- **scikit-learn** (≥1.0.0): Preprocessing and evaluation metrics
- **matplotlib** (≥3.4.0): Visualization
- **seaborn** (≥0.11.0): Statistical visualization
- **scipy** (≥1.7.0): Statistical functions
- **jdatetime** (≥5.0.0): Persian/Solar calendar conversion (required for date preprocessing)

### GPU Support (Optional but Recommended)

For faster training, CUDA-enabled GPU is recommended. The code automatically detects and uses GPU if available. Ensure you have:

- CUDA-compatible GPU
- CUDA toolkit installed
- PyTorch with CUDA support

## Project Structure

```
.
├── LSTM_V4_Ensemble_latest.ipynb    # Main implementation notebook
├── README.md                         # This file
└── anonymized_prescription.csv      # Dataset file (to be created by user)
```

## Usage

### Step 1: Data Preparation

1. Download the dataset from the [source repository](https://github.com/farvaresh/pharmacy-prescription-demand-dataset)
2. Concatenate all CSV files into `anonymized_prescription.csv` (see Data Preparation section above)
3. Place the file in the project directory

### Step 2: Update File Path

In the notebook, update the data path in Cell 3:

```python
data_path = '/path/to/your/anonymized_prescription.csv'
```

For local execution, use:

```python
data_path = 'anonymized_prescription.csv'
```

### Step 3: Run the Notebook

Execute the notebook cells sequentially:

1. **Cell 0-2**: Environment setup and library imports
2. **Cell 3**: Data loading and ABC analysis filtering
3. **Cell 4-5**: Date conversion (Solar to Gregorian calendar)
4. **Cell 6**: Descriptive statistics and data exploration
5. **Cell 7**: Exploratory data visualization
6. **Cell 8**: Heterogeneous graph construction
7. **Cell 9**: Temporal data preparation and windowing
8. **Cell 10**: LSTM model definition
9. **Cell 11**: Ensemble training and evaluation
10. **Cell 12-24**: Managerial analysis and reporting

## Methodology

### 1. Data Filtering (ABC Analysis)

The system employs a smart filtering strategy combining:
- **Sales volume criterion**: Top 30% by frequency
- **Consistency criterion**: Drugs present in at least 12 months

This ensures that only predictable and commercially relevant drugs are included in the model.

### 2. Heterogeneous Graph Construction

A heterogeneous graph is constructed with three node types:
- **Drug nodes**: Features include normalized price and peak month (seasonality indicator)
- **Doctor nodes**: Represented by specialty indices
- **Patient nodes**: Features include age and gender

Edge types include:
- Doctor-prescribes-Drug (bidirectional)
- Patient-uses-Drug (bidirectional)
- Drug-co-occurs-Drug (weighted by prescription co-occurrence)

### 3. Ensemble LSTM Architecture

- **Base Model**: Two-layer LSTM with 64 hidden units
- **Regularization**: Dropout (0.4) and weight decay (1e-4)
- **Activation**: LeakyReLU to prevent dead neurons
- **Ensemble Size**: 5 models with median aggregation
- **Loss Function**: L1 Loss (Mean Absolute Error)
- **Optimizer**: Adam with learning rate scheduling

### 4. Temporal Feature Engineering

- **Window Size**: 3 months
- **Features**: 
  - Normalized sales volume
  - Seasonal encoding (sin/cos transformation)
  - Month-to-month lag features
- **Scaling**: MinMax normalization based on training period averages

## Results

### Model Performance

- **Mean Absolute Error (MAE)**: 794.05 units
- **Weighted Mean Absolute Percentage Error (WMAPE)**: 17.70%
- **Individual Model Performance**: WMAPE ranging from 17.45% to 18.05%

### Forecast Quality Distribution

- **Excellent (<15% error)**: 148 drugs (17.1%)
- **Good (15-30% error)**: 117 drugs (13.6%)
- **Acceptable (30-50% error)**: 92 drugs (10.7%)
- **Unpredictable (>50% error)**: 179 drugs (20.7%)

### Business Impact

- **Volume Coverage**: 84.5% of total pharmacy sales are predicted with error below 30%
- **Automation Readiness**: 43.3% of drugs can be ordered automatically with acceptable risk
- **Bias Analysis**: Model exhibits conservative bias (83.5% over-forecast), reducing stockout risk

## Output Files

The notebook generates several output files:

- `Table_4_1_Descriptive_Stats.csv`: Statistical summary of quantitative variables
- `Table_4_2_General_Info.csv`: General dataset characteristics
- `Final_Forecast_Report.csv`: Comprehensive forecast report with per-drug accuracy metrics

## Key Features

1. **Robust Ensemble Approach**: Median aggregation reduces impact of outlier predictions
2. **Small Data Regime Handling**: Optimized for limited historical data (24 months)
3. **Managerial Insights**: Financial risk analysis and automation readiness assessment
4. **Comprehensive Visualization**: Multiple charts for model interpretation and business communication
5. **Production-Ready**: Includes bias analysis, error distribution, and quality classification

## Limitations

- Model performance may vary for drugs with very low sales volume
- Results are specific to the dataset characteristics and may not generalize to all pharmacy settings
- The model assumes stable prescription patterns and may require retraining for significant market changes

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@mastersthesis{farvaresh2025,
  title={Intelligent Decision Support System for Drug Inventory Management: A Robust Ensemble Approach in Small Data Regimes},
  author={Farvaresh, Mokhtar},
  year={2025}
}
```

## Dataset Citation

Please also cite the dataset source:

```bibtex
@dataset{farvaresh2025dataset,
  title={Anonymised Community Pharmacy Prescription Dataset (2018–2020)},
  author={Farvaresh, Hamid},
  year={2025},
  url={https://github.com/farvaresh/pharmacy-prescription-demand-dataset},
  version={1.0.0}
}
```

## License

This code is provided for academic and research purposes. Please refer to the dataset license for data usage terms: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

## Contact

For questions regarding this implementation or collaboration opportunities:

**Mokhtar Farvaresh**  
Email: mokhtar.farvaresh@gmail.com

## Acknowledgments

- Dataset provided by Dr. Hamid Farvaresh
- Built using PyTorch and PyTorch Geometric
- Visualization libraries: Matplotlib and Seaborn

## References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
2. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR.
3. Veličković, P., et al. (2018). Graph attention networks. ICLR.

---

**Note**: This is a research implementation. For production deployment, additional considerations such as model monitoring, retraining schedules, and integration with inventory management systems should be addressed.
