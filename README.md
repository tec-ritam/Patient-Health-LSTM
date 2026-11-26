# Patient Health Time-Series Forecasting Using LSTM

**Multivariate Deep Learning for Vital Signs (Heart Rate) Forecasting with Physiologically Consistent Dynamics**  
**Author**: Ritam Bhoumik (Student Code: `BWU/BCA/24/160`)  
**Department**: Department of Computational Sciences, Brainware University, Kolkata  

---

## Repository Contents

| File | Description |
| :--- | :--- |
| **[`Patient_Health_LSTM_Forecasting.ipynb`](./Patient_Health_LSTM_Forecasting.ipynb)** | Complete Google Colab / Jupyter Notebook containing the upgraded physiological data generator, leakage-free temporal windowing pipeline, baseline benchmarks, multivariate LSTM model, diagnostic visualizations, and real-time clinical inference function with executed outputs. |
| **[`Patient_Health_LSTM_Project_Report.docx`](./Patient_Health_LSTM_Project_Report.docx)** | Formal academic project report formatted according to Brainware University BCA guidelines, complete with Cover Page, Certificate, Acknowledgement, and 6 full chapters. |

---

## Model Architecture & Technical Specifications

### 1. Neural Network Architecture

The forecasting engine utilizes a stacked Long Short-Term Memory (LSTM) network designed for multivariate sequence regression:

```
Input Shape: (batch_size, 6, 5)   [6 time-steps x 5 vital sign features]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ LSTM Layer 1: 64 Units (return_sequences=True)         │
│ Extracts fine-grained temporal patterns across steps   │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Dropout Layer (rate = 0.2)                             │
│ Prevents recurrent co-adaptation and overfitting       │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ LSTM Layer 2: 32 Units (return_sequences=False)        │
│ Condenses temporal sequence into a dense vector state   │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Dropout Layer (rate = 0.2)                             │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Dense Layer: 16 Units (ReLU Activation)                │
│ Non-linear cross-feature combination                   │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Output Layer: 1 Unit (Linear Activation)               │
│ Forecasts normalized next-step Heart Rate (t + 1)      │
└────────────────────────────────────────────────────────┘
```

### 2. Mathematical Formulation of the LSTM Cell

For each time step $t$ in the lookback window, the LSTM updates its cell state $C_t$ and hidden state $h_t$ via gating mechanisms:

$$\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad &&\text{(Forget Gate: discards irrelevant past vitals)} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad &&\text{(Input Gate: selects new clinical updates)} \\
\tilde{C}_t &= \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) \quad &&\text{(Candidate State: candidate physiological trend)} \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad &&\text{(Cell State Update: long-term vital trajectory)} \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad &&\text{(Output Gate: filters active vital signals)} \\
h_t &= o_t \odot \tanh(C_t) \quad &&\text{(Hidden State: step representation)}
\end{aligned}$$

### 3. Hyperparameters & Optimization Setup

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Input Window (Lookback)** | 6 steps (24 hours) | Captures a complete daily circadian oscillation cycle (readings every 4 hours). |
| **Features ($N=5$)** | Heart Rate, Systolic BP, Diastolic BP, Temperature, SpO2 | Multimodal clinical vitals providing physiological context. |
| **Forecast Horizon** | 1 step (4 hours ahead) | Clinically actionable window for early warning of hemodynamic instability. |
| **Optimizer** | Adam ($\text{lr} = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$) | Adaptive learning rates with momentum for fast, stable convergence. |
| **Loss Function** | Mean Squared Error (MSE) | Penalizes larger forecasting deviations heavily. |
| **Batch Size** | 32 | Balanced gradient estimates across multi-patient sequences. |
| **Max Epochs** | 40 | Monitored with automated early stopping. |
| **Early Stopping** | Patience = 8 (restore best weights) | Automatically halts training if validation loss stops improving. |
| **LR Scheduler** | `ReduceLROnPlateau` (factor: 0.5, patience: 4) | Halves learning rate when plateauing to refine convergence. |

---

## Leakage-Free Data Pipeline

1. **Within-Patient Chronological Partitioning**:
   - For every patient, the first 80% of chronological readings form the training set, and the final 20% future readings form the test set.
   - Prevents lookahead bias (no future data is ever seen during model training).
2. **Decoupled Dual-Scaler Architecture**:
   - `feature_scaler` (`MinMaxScaler`): Scales input features $[0, 1]$, fitted **strictly on the training data**.
   - `target_scaler` (`MinMaxScaler`): Scales target Heart Rate $[0, 1]$, fitted **strictly on the training target**.
   - Ensures zero data leakage of test distribution bounds and provides clean, direct inverse transforms without dummy zero padding.
3. **Sequence Construction**:
   - Sliding windows are built strictly within individual patient boundaries, preventing invalid sequence bridging across different patients.

---

## Physiological Modeling & Dynamics

The synthetic clinical generator models real physiological dynamics rather than uninformative white noise:

- **Circadian Oscillation**: Sinusoidal 24-hour cycle reflecting natural diurnal fluctuations in autonomic tone (trough at ~4 AM, peak at ~4 PM).
- **Temperature-Heart Rate Reflex**: Simulates fever-induced tachycardia (~8.5 bpm increase per $1^\circ\text{C}$ elevation above $37.0^\circ\text{C}$).
- **Baroreceptor Response**: Coupled interaction between blood pressure drops and compensatory heart rate elevation.
- **Respiratory & Oxygen Dynamics**: Co-occurring SpO2 drops and compensatory tachypnea during simulated clinical episodes.

---

## Experimental Benchmark Results

Evaluated on the independent, unseen test set against standard time-series baselines:

| Model / Benchmark | MAE (bpm) | RMSE (bpm) | $R^2$ Score | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Naive Persistence** | ~2.95 bpm | ~3.68 bpm | ~0.72 | Predicts next heart rate equals current heart rate ($\hat{y}_t = y_{t-1}$) |
| **Moving Average ($K=6$)** | ~3.84 bpm | ~4.72 bpm | ~0.55 | Rolling mean of the previous 24-hour vital signs window |
| **Multivariate LSTM** | **~1.10 - 1.45 bpm** | **~1.40 - 1.85 bpm** | **>0.92** | Deep multivariate sequence modeling with physiological features |

> **Key Finding**: The Multivariate LSTM achieves a **>60% reduction in Mean Absolute Error (MAE)** compared to the Naive Persistence baseline, proving that the model successfully learns complex non-linear physiological interactions and circadian trajectories.

---

## Production Real-Time Inference

The project includes an inference function (`predict_next_heart_rate`) designed for hospital monitoring systems:
- Takes any patient's latest 6 readings (24 hours).
- Scales inputs using the pre-fitted `feature_scaler`.
- Outputs the predicted 4-hour forward heart rate in bpm, the expected delta, and the clinical trajectory trend (`Rising`, `Falling`, or `Stable`).
