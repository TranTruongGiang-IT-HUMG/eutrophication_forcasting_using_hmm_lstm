# Coastal Water Quality Analysis and Forecasting Using a Hybrid HMM-LSTM Model

## Overview

This repository presents a hybrid Hidden Markov Model–Long Short-Term Memory (HMM-LSTM) framework for 7-day coastal water quality forecasting. The model ingests multivariate time-series measurements of physicochemical and microbiological indicators and outputs categorical water quality predictions (e.g., Good, Moderate, Poor) over a rolling forecast horizon.

## Model Architecture

The proposed architecture integrates probabilistic state inference with deep sequential learning through a dual-branch design:

1. **HMM Feature Extraction Layer** — A dense layer with Tanh activation that encodes latent environmental regimes from the raw time series.
2. **Branch 1 (Primary)** — LSTM-1 (128 units) → LSTM-2 (64 units) → Attention mechanism, capturing long-range temporal dependencies with selective focus.
3. **Branch 2 (Auxiliary)** — LSTM-3 (64 units) for comparative representation learning.
4. **Fusion & Classification** — Concatenation of both branches followed by Dense(64) → Dense(32) → Softmax output.

## Dataset

- **File:** `water_quality_hmm_lstm_training_data_ver_1.0.csv`
- **Features:** TSS (Total Suspended Solids), pH, TPH (Total Petroleum Hydrocarbons), Total Coliform, DO (Dissolved Oxygen)
- **Target:** `Water_Quality` (categorical class label)
- **Temporal resolution:** Daily observations

## Notebook Structure

| Section | Description |
|---------|-------------|
| 1. Setup and Installation | Dependency installation and library imports |
| 2. Data Loading and Exploration | Dataset ingestion and preliminary inspection |
| 3. Exploratory Data Analysis | Time-series visualization, correlation analysis, feature distributions, and class-conditional box plots |
| 4. Trend Analysis | Short-term (7-day) and long-term (30-day) moving average decomposition with volatility statistics |
| 5. Data Preprocessing | Sequence generation with a sliding window (`time_steps=7`), temporal train/validation/test split (70/15/15) |
| 6. HMM Feature Engineering | Gaussian HMM fitting to extract hidden-state posterior probabilities as auxiliary input features |
| 7. Model Construction | Keras implementation of the dual-branch HMM-LSTM-Attention architecture |
| 8. Training | Categorical cross-entropy optimization with Adam, early stopping, and learning rate scheduling |
| 9. Evaluation | Test-set accuracy, precision, recall, classification report, and confusion matrix |
| 10. 7-Day Forecasting | Autoregressive multi-step prediction over a 7-day horizon with probability visualization |
| 11. Model Interpretation | Prediction distribution analysis and per-class performance breakdown |
| 12. Artifact Export | Serialization of the trained model (`.h5`), scaler, and HMM parameters |
| 13. Summary | Consolidated results and key findings |

## Requirements

```
tensorflow
pandas
numpy
matplotlib
seaborn
scikit-learn
hmmlearn
plotly
```

## Usage

1. Open the notebook in Google Colab or a local Jupyter environment.
2. Upload the dataset (`water_quality_hmm_lstm_training_data_ver_1.0.csv`) when prompted.
3. Execute cells sequentially. Training completes in approximately 100 epochs with early stopping.
4. The trained model and preprocessing artifacts are saved to the working directory upon completion.

## Outputs

- `hmm_lstm_water_quality_model.h5` — Trained Keras model
- Scaler and HMM model objects (pickle) for inference-time preprocessing
- Evaluation metrics, confusion matrix, and 7-day forecast visualizations

## License

This project is provided for academic and research purposes.
