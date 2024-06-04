# Commrate_uz
A project written using Python, which evaluates the positiveness or negativity of the written comment on a scale from 1 to 5 in Uzbek Language
<br>

# Sentiment Analysis with Various Classification Algorithms and Tokenization Techniques

This project implements a sentiment analysis model using different classification algorithms and tokenization techniques. It follows a structured approach involving data preprocessing, tokenization, feature extraction, model training, and evaluation.

## 1. Data Preprocessing:

The code begins by loading two text files containing positive and negative Uzbek sentences. The sentences are then combined, cleaned by removing newline characters and non-alphanumeric characters, and converted to lowercase.

## 2. Tokenization:

Two tokenizers are used: `BertTokenizerFast` and `AutoTokenizer`. The `BertTokenizerFast` is used to train a new tokenizer based on the provided text corpus. The new tokenizer is then used to tokenize the text data.

## 3. Feature Extraction:

A `CountVectorizer` is used to convert the tokenized text into numerical features.

## 4. Model Training and Evaluation:

Three classification models are trained:
- LightGBM Classifier
- Random Forest Classifier
- Logistic Regression

Each model is trained on the training data and evaluated on the test data. Accuracy and confusion matrices are used to assess the performance of each model.

## 5. Documentation:

The code is well-documented with comments explaining the purpose of each section. The variable names are descriptive and follow Python naming conventions. The code is formatted using consistent indentation and spacing.

## 6. Additional Information:

- The code utilizes several libraries such as `pandas`, `numpy`, `re`, `transformers`, `sklearn`, `matplotlib`, and `seaborn`.
- The `dataset.csv` file is assumed to contain labeled text data for training and evaluation.
- The code can be readily adapted to other sentiment analysis tasks by providing appropriate training data and modifying the model parameters.
