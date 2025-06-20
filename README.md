## Sentiment Analysis in Finance

**Project Overview**

This project explores how social media sentiment can influence real-world events — a phenomenon seen in the GameStop stock surge, where a group of traders on Reddit encouraged thousands to invest in GameStop, increasing its stock value by over 100%. Interested in seeing how market sentiment influences stock prices, this project takes the first step in applying a variety of natural language processing techniques to classify the sentiment (positive/negative) of text data. It covers the full pipeline:
- Exploratory data analysis (text cleaning, tokenisation, data inspection, vectorisation)
- Model training and evaluation using:
  - LSTM with word embeddings (using tokenised and padded sequences)
  - Pretrained BERT (fine-tuned)
 
**Tech Stack/Libraries**
- Python 3
- TensorFlow / Keras
- HuggingFace Transformers
- NLTK
- Scikit-learn, Pandas, NumPy

**Folder Structure**
```bash
.
├── bert_sentiment.py                   # BERT model script
├── lstm_sentiment.py                   # LSTM model script
├── exploratory-data-analysis-nlp.py    # Preprocessing/tokenisation/EDA
├── finance_train.csv                   # Training dataset
├── finance_test.csv                    # Test dataset 
└── README.md                           # Project overview
