**US Equity Market Return Forecasting using LSTM**

## Overview

This project was developed as part of the **Capital Fund Management (CFM) MSc challenge**.  
The objective is to predict short-term equity return trends on the US market using deep learning techniques.

Specifically, the model forecasts the direction of returns between 2:00 PM and 4:00 PM based on intraday historical data.

---

## Problem Statement

Given historical intraday returns (5-minute frequency), the goal is to classify future return trends into three categories:

- -1 → Return < -25 bps  
- 0 → Return between [-25 bps ; +25 bps]  
- +1 → Return > +25 bps  

---

## Dataset

- ~1,000,000 observations  
- Time granularity: 5-minute returns  
- Input features:
  - `r0` → `r52` (first 4.5 hours of trading, i.e. 53 intervals)
- Target:
  - End-of-day return trend (`Reod`)

---

## Methodology

### 1. Data Preprocessing
- Feature selection (intraday returns)
- Label encoding + one-hot encoding
- Train / validation split (80/20)
- Min-Max normalization

### 2. Model Architecture

A Bidirectional LSTM neural network is used to capture temporal dependencies in financial time series.

**Architecture:**
- Bidirectional LSTM (100 units)
- Dropout (0.4)
- Bidirectional LSTM (50 units)
- Dropout (0.2)
- Dense layer (softmax, 3 classes)

### 3. Training
- Loss: Categorical Crossentropy  
- Optimizer: Adam  
- Early stopping to prevent overfitting 
