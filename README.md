# 🧠 Mental Health Analysis: Students Anxiety and Depression Dataset
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0-orange)

Welcome to my portfolio repository, where I apply machine learning techniques to analyze **mental health** datasets, specifically focusing on **student anxiety and depression**. My aim is to demonstrate how data-driven approaches can provide valuable insights into mental health issues.

## Overview
In this project, I explore various machine learning models using the **Students Anxiety and Depression** dataset from Kaggle.

- **Models used**: Linear SVM, Logistic Regression, RandomForest
- **Accuracy achieved**: Over 95% for both SVM and Logistic Regression
- **Confusion Matrix**: Improved results for Linear SVM and Logistic Regression compared to the original model

🔗 [Original Kaggle Project](https://www.kaggle.com/code/sasakitetsuya/students-anxiety-and-depression-classify-model)

## 🔍 Key Contributions:
1. Added **Linear SVM** and **Logistic Regression** models.
2. Enhanced the **confusion matrix** results.
3. Applied additional **data analysis** techniques for better data understanding.

## 🔧 Tools and Libraries:
- **Python**, **Scikit-learn**, **Matplotlib**, **Seaborn**

![Confusion Matrix](your-image-url-here)

> **Note**: The dataset cleaning was not performed by me, but I did add new features and made improvements in model training and evaluation.

---

# 🔬 Kaggle: Suicide and Depression Detection using TensorFlow

This project uses **TensorFlow** and **Scikit-learn** to detect suicidal and depressive comments from three subreddits.

- **Subreddits Used**:
  - [r/SuicideWatch](https://www.reddit.com/r/SuicideWatch/) -> Label: Suicide
  - [r/depression](https://www.reddit.com/r/depression/) -> Label: Depression
  - [r/teenagers](https://www.reddit.com/r/teenagers/) -> Label: Non-Depression

---

## 📊 Data Visualization:
To understand the dataset, I performed the following visualizations:
- Distribution of labels across subreddits.
- Word clouds to show the most frequent words associated with each class.
- Correlation heatmaps to detect relationships between variables.

_Add graphs and charts here with the code used to generate them._

---

## 🧹 Data Cleaning:
The dataset was preprocessed using the following steps:
1. Tokenization and removal of stopwords.
2. Normalization by converting all text to lowercase.
3. Lemmatization to reduce words to their base form.
4. Removal of outliers based on word count distributions.

---

## 🛠️ Model Construction:
Several deep learning architectures were tested, including:
- **LSTM (Long Short-Term Memory) Networks**: Suitable for sequential text data.
- **Convolutional Neural Networks (CNNs)**: For text classification.
- **BERT-based Model**: State-of-the-art performance for NLP tasks.

---

## 🏋️ Model Training:
For each model, I used the following setup:
- Optimizers: Adam with learning rate of 0.001
- Loss function: Binary Cross-Entropy
- Batch size: 64
- Epochs: 10

---

## 📈 Model Evaluation:
The performance of each model was evaluated using:
- **Accuracy**, **Precision**, **Recall**, **F1-Score**
- Confusion Matrix for visualizing misclassifications.

Here's a summary of the results:

| Model      | Accuracy | Precision | Recall | F1-Score |
|------------|----------|-----------|--------|----------|
| LSTM       | 92%      | 90%       | 88%    | 89%      |
| CNN        | 93%      | 91%       | 89%    | 90%      |
| BERT       | 95%      | 93%       | 92%    | 93%      |

_Add a confusion matrix image here or another evaluation graph._

---

Feel free to explore the code and make contributions!

## 📝 Related Projects:
- [Leukemia Classifier](https://github.com/nicolasvargaszz/acute-lymphoblastic-leukemia-classifier)

---

**Contact Information**:
- **Email**: nicolasvargaszz2104@gmail.com
- **LinkedIn**: [Nicolas Vargas](https://www.linkedin.com/in/nicol%C3%A1s-vargas-41bb67253/)

---
