# Cyber Threat Detection using Deep Learning

## Project Overview
Cyber threats such as malware, phishing, and denial-of-service (DOS) attacks are a growing concern for organizations worldwide. Traditional threat detection methods often struggle to adapt to new and evolving threats. 

This project aims to proactively detect and mitigate cyber threats by leveraging Deep Learning. Using the BETH dataset, which simulates real-world logs, this repository contains a PyTorch-based neural network designed to identify patterns in system logs and classify events as either malicious or benign.

## Dataset
The project utilizes the [**BETH dataset**](https://www.kaggle.com/datasets/katehighnam/beth-dataset), which contains preprocessed system logs. The target variable is `sus_label`, which indicates whether an event is malicious (1) or benign (0).

**Features:**
* **processId:** The unique identifier for the process that generated the event
* **threadId:** ID for the thread spawning the log
* **parentProcessId:** Label for the process spawning this log
* **userId:** ID of the user spawning the log
* **mountNamespace:** Mounting restrictions the process log works within
* **argsNum:** Number of arguments passed to the event
* **returnValue:** Value returned from the event log (usually 0)
* **sus_label:** Binary label as a suspicious event (Target Variable)

## Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, Scikit-learn (StandardScaler)
* **Deep Learning Framework:** PyTorch
* **Evaluation:** TorchMetrics

## Model Architecture
The model is a Feedforward Neural Network built using PyTorch's `nn.Sequential` module. 
* **Input Layer:** Matches the feature dimension of the scaled BETH dataset.
* **Hidden Layer 1:** 32 neurons, ReLU activation
* **Hidden Layer 2:** 16 neurons, ReLU activation
* **Output Layer:** 1 neuron, Sigmoid activation (for binary classification)
* **Optimizer:** Stochastic Gradient Descent (SGD)
* **Loss Function:** Binary Cross Entropy Loss (BCELoss)

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TalalHamdani/cyber-threat-detection.git
   cd cyber-threat-detection
   ```

2. **Install the required dependencies:**
   Make sure you have Python installed. Then, install the required packages:
   ```bash
   pip install pandas scikit-learn torch torchmetrics
   ```

3. **Data Placement:**
   Ensure that the `data/` directory exists in the root folder and contains the following files:
   * `labelled_train.csv`
   * `labelled_validation.csv`
   * `labelled_test.csv`

## Usage

Run the main training script:
```bash
python src/main.py
```

### Pipeline Overview
1. **Data Loading:** Reads the CSV files into Pandas DataFrames.
2. **Preprocessing:** Separates features from the target label (`sus_label`) and standardizes the features using `StandardScaler`.
3. **Tensor Conversion:** Converts the scaled Numpy arrays into PyTorch Tensors and wraps the training data in a `DataLoader` for batching (batch size = 32).
4. **Training Loop:** Passes batches through the network, calculates loss, and backpropagates gradients for 10 epochs.
5. **Evaluation:** Uses `torchmetrics.Accuracy` to evaluate performance on the training, validation, and testing sets without tracking gradients (`torch.no_grad()`).

## Results & Metrics
The model achieves a target validation accuracy of at least **0.60 (60%)**.

```text
Final Results:
Training Accuracy:   99.96%
Validation Accuracy: 100%
Testing Accuracy:    94.48%
```

## Potential Improvements
* **Hyperparameter Tuning:** Experiment with Adam optimizer, varying learning rates, or adjusting the hidden layer dimensions.
* **Handling Imbalance:** If the dataset has significantly more benign logs than suspicious logs, implementing SMOTE or weighted loss functions could improve recall.
* **Real-time Pipeline:** Integrate the inference script with a live network logging tool (like Zeek or Sysmon) for live anomaly flagging.
