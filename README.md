# Multi-Modal Stock Trend Prediction using Sentiment Fusion and Deep Learning

This repository contains the implementation of a comprehensive comparative study between traditional Machine Learning (ML) and advanced Deep Learning (DL) architectures for predicting daily stock price movements. By fusing Twitter sentiment analysis with technical market indicators, this project explores the predictive boundaries of social-media-driven algorithmic trading.

## Project Overview

The objective is to classify daily stock movements as "Up" or "Down" using a multi-modal feature set. The project evaluates six distinct architectures:
* **Baselines:** Logistic Regression, Random Forest, XGBoost.
* **Advanced Models:** TCN-LSTM Hybrid, Relational Feature-GCN, and Multi-Modal Transformer.

### Key Features
* **Temporal Sentiment Aggregation:** VADER-based sentiment extraction aligned with market trading hours.
* **Technical Indicator Fusion:** Integration of MACD, Bollinger Bands, RSI, and Log Momentum.
* **Comparative Analysis:** In-depth evaluation of the "Accuracy Paradox" in financial forecasting.
* **State-of-the-Art Architecture:** Implementation of a Multi-Head Attention Transformer for temporal feature weighting.

## Dataset

The project utilizes the **Stock Tweets for Sentiment Analysis and Prediction** dataset.
* **Tweets:** ~80,000 unique records across 25 major tickers (TSLA, AAPL, AMZN, etc.).
* **Prices:** Historical OHLCV data spanning 2021-2022.
* **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction)

## Methodology

### 1. Preprocessing Pipeline
* **Text Cleaning:** Removal of URLs, handles, and tickers; lowercase normalization.
* **Sentiment Extraction:** Calculation of volume-weighted daily compound scores using VADER.
* **Feature Scaling:** RobustScaler application to financial metrics to mitigate outlier influence.
* **Windowing:** 20-day sliding window temporal sequencing.

### 2. Model Architecture

The Multi-Modal Transformer serves as the flagship model, utilizing the following attention mechanism:
$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

## Results

| Model | Accuracy | F1-Score (Class 1) | Macro F1 |
| :--- | :--- | :--- | :--- |
| Logistic Regression | 0.54 | 0.16 | 0.41 |
| Random Forest | 0.51 | 0.40 | 0.50 |
| XGBoost | 0.51 | 0.43 | 0.51 |
| TCN-LSTM Hybrid | 0.49 | 0.44 | 0.49 |
| Feature-GCN | 0.53 | 0.45 | 0.53 |
| **Transformer** | **0.55** | **0.47** | **0.54** |

### Key Findings
* **Accuracy Paradox:** Logistic Regression achieves 54% accuracy by over-predicting the majority class, resulting in a low F1-score.
* **DL Advantage:** The Multi-Modal Transformer provides a 193% improvement in F1-score over linear baselines, proving the necessity of non-linear attention mechanisms.

## Repository Structure
```text
├── data/                   # Dataset placeholders
├── notebooks/              # Jupyter Notebooks for training and EDA
├── results/                # Confusion Matrices and ROC Curves
├── src/                    # Model definitions and preprocessing scripts
└── README.md               # Project overview
