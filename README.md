# 📝 Next Word Prediction using RNN & LSTM

A deep learning project that predicts the **next word in a sentence** using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks.

The model is trained on a dataset of quotes and learns word patterns to predict the most probable next word based on the given input sequence.

---

## 🚀 Project Overview

Next-word prediction is an important Natural Language Processing (NLP) task used in applications such as:

- Smart keyboards
- Text autocomplete
- Search suggestions
- Text generation
- Conversational AI

This project implements and compares two recurrent neural network architectures:

- **Simple RNN**
- **LSTM**

The trained LSTM model is used for generating text from a user-provided seed sentence.

---

## 🧠 Project Workflow

```text
Quote Dataset
      ↓
Text Preprocessing
      ↓
Tokenization
      ↓
Sequence Generation
      ↓
Sequence Padding
      ↓
Model Training
      ↓
RNN / LSTM
      ↓
Next Word Prediction
      ↓
Text Generation
```

---

## 📂 Dataset

The project uses a quote dataset:

```text
qoute_dataset.csv
```

The dataset contains quotes in a column named:

```text
quote
```

---

## 🛠️ Tech Stack

### Programming Language

- Python

### Libraries

- NumPy
- Pandas
- Matplotlib
- Seaborn
- TensorFlow
- Keras

### Machine Learning / Deep Learning

- Natural Language Processing
- Word Tokenization
- Word Embeddings
- Recurrent Neural Networks
- LSTM Networks
- Sequence Modeling

---

## 🧠 Model Architecture

### Simple RNN

```text
Input
  ↓
Embedding Layer
  ↓
Simple RNN
  ↓
Dense Layer
  ↓
Softmax
  ↓
Predicted Word
```

### LSTM

```text
Input
  ↓
Embedding Layer
  ↓
LSTM
  ↓
Dense Layer
  ↓
Softmax
  ↓
Predicted Word
```

---

## ⚙️ Model Configuration

| Parameter | Value |
|---|---:|
| Vocabulary Size | 8978 |
| Embedding Dimension | 50 |
| RNN/LSTM Units | 128 |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Output Activation | Softmax |

---

## ✨ Features

- Text preprocessing
- Word-level tokenization
- Sequence generation
- Sequence padding
- Simple RNN implementation
- LSTM implementation
- Next-word prediction
- Multi-word text generation
- Trained model saving
- Tokenizer saving

---

## 📁 Project Structure

```text
next-word-prediction/
│
├── qoute_dataset.csv
├── model.py
├── lstm_model.h5
├── tokenizer.pkl
├── max_len.pkl
├── requirements.txt
└── README.md
```

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/next-word-prediction.git
cd next-word-prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the project:

```bash
python model.py
```

Provide a seed sentence and generate the desired number of words.

Example:

```text
Input:
you will
```

The model predicts the next word based on patterns learned from the quote dataset.

---

## 💾 Saved Models

The project saves the following files:

| File | Purpose |
|---|---|
| `lstm_model.h5` | Trained LSTM model |
| `tokenizer.pkl` | Fitted text tokenizer |
| `max_len.pkl` | Maximum sequence length |

These files can be used to perform predictions without retraining the model.

---

## 📊 Model Comparison

The project includes implementations of both **Simple RNN and LSTM** to explore their performance on sequential text prediction.

| Model | Purpose |
|---|---|
| Simple RNN | Baseline sequence model |
| LSTM | Improved sequence modeling and final prediction model |

---

## ⚠️ Limitations

- Generated text quality depends heavily on the training dataset.
- The current model uses a relatively small vocabulary.
- Predictions are based on learned word patterns rather than semantic understanding.
- Using `argmax` for prediction can produce repetitive outputs.
- Generated sentences may not always be grammatically correct.
- The model does not use modern Transformer-based architectures.

---

## 🚀 Future Improvements

- [ ] Add a Streamlit web interface
- [ ] Compare RNN and LSTM accuracy
- [ ] Add training and validation graphs
- [ ] Implement temperature sampling
- [ ] Implement Top-K sampling
- [ ] Implement Top-P sampling
- [ ] Use a larger dataset
- [ ] Improve text preprocessing
- [ ] Experiment with GRU
- [ ] Experiment with Transformer-based models

---

## 🎓 Concepts Covered

- Natural Language Processing
- Text preprocessing
- Tokenization
- Vocabulary creation
- Sequence modeling
- Word embeddings
- Recurrent Neural Networks
- LSTM
- Deep learning
- Text generation

---

## 👨‍💻 Author

**Vansh Dhiman**

BTech Computer Science Student

⭐ If you found this project useful, consider starring the repository!
