# 🧠 Binary Text Classifier with RNN

A Natural Language Processing (NLP) project using a Recurrent Neural Network (RNN) to classify text messages as related or unrelated to disaster events.

## 📂 Dataset

The dataset used consists of short tweets labeled as either:
- `1`: related to a disaster
- `0`: not related to a disaster

Each record contains raw text and its associated label. The dataset simulates real-world scenarios where urgent classification of social media content is crucial.

## 🎯 Objectives

- Preprocess text data: cleaning, tokenization, and vectorization.
- Build a baseline deep learning model using TensorFlow and Keras.
- Train an RNN-based architecture (with LSTM layers) for binary classification.
- Evaluate model performance with metrics such as Accuracy, Precision, Recall, and F1-score.
- Visualize training history and predictions.

## 🛠️ Tools & Libraries

- Python 3.10
- TensorFlow 2.11
- Keras
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- TensorFlow Hub

## 📊 Model Architecture

The model includes:
- `TextVectorization` layer
- `Embedding` layer
- `LSTM` layer
- `Dropout` and `Dense` layers

The model is compiled with:
```python
loss = 'binary_crossentropy'
optimizer = 'adam'
metrics = ['accuracy']
