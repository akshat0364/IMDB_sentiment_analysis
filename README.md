# IMDb Sentiment Analysis Project

## Overview

This project focuses on sentiment analysis of IMDb movie reviews. The goal is to classify reviews as either positive or negative based on their content. The dataset used for this analysis contains 50,000 reviews with corresponding sentiments.

## Table of Contents

- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Text Preprocessing](#text-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Best Model](#best-model)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset used in this project is the "IMDB Dataset" containing 50,000 movie reviews labeled as positive or negative. It is stored in a CSV file named `IMDB Dataset.csv`.

## Exploratory Data Analysis

- Sentiment distribution visualization using seaborn countplot.
- Word count analysis in reviews using histograms.

## Text Preprocessing

- Tokenization, stemming, and removal of stop words.
- Handling HTML tags and special characters.
- Removal of duplicate entries in the dataset.

## Feature Engineering

- Creation of word clouds to visualize most frequent words in positive and negative reviews.
- Counting and visualization of common words in both positive and negative reviews.

## Model Training

Three models were trained for sentiment classification:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear Support Vector Classifier (LinearSVC)

## Model Evaluation

- Evaluation metrics include accuracy, confusion matrix, precision, recall, and F1-score.
- Logistic Regression achieved an accuracy of 89.00%, Multinomial Naive Bayes achieved 86.44%, and LinearSVC achieved 89.22%.

## Hyperparameter Tuning

Grid search was used to find the best hyperparameters for the Linear Support Vector Classifier (LinearSVC). The best parameters were found to be 'C': 1 and 'loss': 'hinge'.

## Best Model

After hyperparameter tuning, the Linear Support Vector Classifier (LinearSVC) achieved the highest accuracy of 89.41%.

## Usage

To use the sentiment analysis model:

1. Clone the repository.
2. Install the required dependencies (`requirements.txt`).
3. Execute the provided Jupyter notebook or script to train and test the model.

## Dependencies

- pandas
- matplotlib
- seaborn
- plotly
- nltk
- wordcloud
- scikit-learn

Install dependencies using:

```bash
pip install -r requirements.txt
