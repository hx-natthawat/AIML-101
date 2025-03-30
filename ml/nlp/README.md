# Natural Language Processing (NLP)

This directory contains practical exercises and examples for Natural Language Processing (NLP) using various libraries and techniques.

## Contents

### 001: Text Processing Basics

- `01-textprocessing.ipynb`: Basic text processing techniques using NLTK for English text

  - Tokenization (sentence and word level)
  - Stopword removal
  - Part-of-speech tagging
  - Stemming

- `02-pythainlp.ipynb`: Thai language text processing using PyThaiNLP

  - Text normalization
  - Word segmentation (tokenization for Thai language)
  - Stopword removal
  - Part-of-speech tagging
  - Handling Thai-specific tokenization challenges

- `03-tfidf.ipynb`: TF-IDF (Term Frequency-Inverse Document Frequency) implementation

  - Creating a TF-IDF vectorizer
  - Converting text documents to TF-IDF features
  - Analyzing word importance across documents
  - Visualizing TF-IDF scores with pandas

- `04-movie-review-classificatiob.ipynb`: Text classification using TF-IDF and machine learning
  - NLTK movie reviews dataset processing
  - Document vectorization with TF-IDF
  - Classification using multiple algorithms:
    - Naive Bayes
    - Support Vector Machines
    - Logistic Regression
  - Model evaluation and performance comparison

## Key Concepts

### English Text Processing

- **Tokenization**: Breaking text into sentences and words
- **Stopword Removal**: Filtering out common words that add little meaning
- **Part-of-Speech Tagging**: Identifying grammatical parts of speech
- **Stemming**: Reducing words to their root/stem form

### Thai Text Processing

- **Word Segmentation**: Unlike English, Thai text doesn't use spaces between words, requiring specialized algorithms
- **Text Normalization**: Standardizing text to handle variations in Thai script
- **Custom Tokenization**: Handling specific cases where default tokenizers may fail

### Text Vectorization and Classification

- **TF-IDF**: A numerical statistic that reflects how important a word is to a document in a collection
  - **Term Frequency (TF)**: How frequently a term appears in a document
  - **Inverse Document Frequency (IDF)**: How rare or common a term is across all documents
- **Text Classification**: Categorizing text documents into predefined classes
  - **Feature Extraction**: Converting text to numerical features using TF-IDF
  - **Supervised Learning**: Training classification models on labeled data
  - **Model Evaluation**: Assessing performance using metrics like accuracy and confusion matrices

## Libraries Used

- **NLTK (Natural Language Toolkit)**: Comprehensive library for English NLP and corpus access
- **PyThaiNLP**: Specialized library for Thai language processing
- **scikit-learn**: Machine learning library for text vectorization and classification
  - TfidfVectorizer for text feature extraction
  - Classification algorithms (Naive Bayes, SVM, Logistic Regression)

## Getting Started

### Prerequisites

```bash
# For English text processing
pip install nltk

# For Thai text processing
pip install pythainlp
```

### Required NLTK Resources

```python
import nltk
nltk.download('punkt')  # Required for sentence tokenization
nltk.download('averaged_perceptron_tagger')  # Required for POS tagging
nltk.download('stopwords')  # Required for stopword removal
nltk.download('movie_reviews')  # Required for the movie review classification example
```

## Common Issues and Solutions

### Thai Word Segmentation Challenges

Default tokenizers may incorrectly segment certain Thai words. For example, "มากกๆ" might be incorrectly tokenized as "มา", "กก", "ๆ" instead of "มาก", "ๆ".

Solution: Use custom post-processing or specialized tokenizers:

```python
def fix_tokens(tokens):
    fixed_tokens = []
    i = 0
    while i < len(tokens):
        # Check for the specific pattern we want to fix
        if i < len(tokens) - 2 and tokens[i] == 'มา' and tokens[i+1] == 'กก' and tokens[i+2] == 'ๆ':
            fixed_tokens.append('มาก')
            fixed_tokens.append('ๆ')
            i += 3
        else:
            fixed_tokens.append(tokens[i])
            i += 1
    return fixed_tokens
```

## Resources

- [NLTK Documentation](https://www.nltk.org/)
- [PyThaiNLP Documentation](https://pythainlp.github.io/)
- [Thai NLP Resource](https://www.thainlp.org/)
