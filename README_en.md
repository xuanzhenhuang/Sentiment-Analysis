# 🔥 ***Sentiment-Analysis***
A project of sentiment polarity classification using a fine-tuned BERT model written in Python. It conducts data analysis on the IMDB movie review dataset, and finally, you can obtain a classification model with an accuracy rate of over **90%**!

<img src="https://github.com/xuanzhenhuang/Sentiment-Analysis/blob/main/image/Predict.png?raw=true" width="1000px">


## 💡 Project Introduction
### 🌱 Project Background

With the rapid development of the Internet and social media, the amount of user-generated content (UGC) has exploded, especially in the fields of movie reviews and product reviews. These textual data contain rich emotional information, which can provide valuable insights for enterprises, research institutions, and individuals. For example, movie producers can analyze the audience's comments on movies to understand the audience's emotional tendencies, so as to optimize future movie production and marketing strategies. However, manually analyzing a large amount of textual data is not only time-consuming and labor-intensive but also easily affected by subjective factors. Therefore, automated sentiment analysis technology has emerged as an important research direction in the field of natural language processing (NLP).

Sentiment polarity classification is a core task of sentiment analysis, aiming to divide texts into emotional categories such as positive or negative. Traditional machine learning methods have limitations in handling complex semantics and context dependencies. In recent years, the emergence of pre-trained language models (such as BERT) has greatly improved the performance of sentiment classification tasks. BERT (Bidirectional Encoder Representations from Transformers) captures the context information of texts through a bidirectional Transformer architecture, which can better understand the semantics and emotional tendencies of texts.

This project aims to use the BERT model to perform sentiment polarity classification on the IMDB movie review dataset. The IMDB dataset is a classic dataset in the field of sentiment analysis, containing 50,000 movie reviews, each of which is labeled as positive or negative. Through the data analysis of these reviews, we can deeply understand the audience's emotional tendencies towards movies, and further fine-tune the BERT model to achieve high-precision sentiment classification. In addition, to verify the generalization ability of the model, we also crawled 921 latest movie reviews from 10 movies from 2021 to 2025 on the Internet for the secondary verification of the model.

---

### 🍔 Project Process

1. **Data Preprocessing**  
   Clean and format the IMDB movie review data to meet the input requirements of the BERT model. Use various data processing libraries (such as Pandas, Re, Nltk) to process all txt movie review files into an Excel table, remove HTML tags, special characters, stop words, etc., and shuffle the data.

2. **Data Preview**  
   Conduct a preliminary exploratory data analysis (EDA) on the IMDB dataset, including word frequency statistics, review length statistics, and distribution of sentiment labels. Use visualization tools (such as Matplotlib, Seaborn) to display the basic characteristics of the data.

3. **Model Fine-tuning**  
   Based on the pre-trained BERT model, use the IMDB dataset for fine-tuning to adapt to the sentiment classification task. The specific steps include:
   - Load the pre-trained BERT model (`bert-base-uncased`), initialize the classification model (`BertForSequenceClassification`), which is suitable for binary classification tasks.
   - Use the `AdamW` optimizer, which is suitable for the fine-tuning of the BERT model.
   - Call the GPU to accelerate the training and verification of the model, evaluate the performance of the model and calculate the accuracy rate.

4. **Data Crawling**  
   Crawl 921 latest movie review data from the IMDB official website from 2021 to 2025 to verify the generalization ability of the model. The crawled data includes information such as movie numbers, review contents, ratings, the number of reviewers, and release times.

5. **Data Processing**  
   Clean and format the crawled latest movie review data. This includes removing noisy data, unifying the text format, and labeling the sentiment labels.

6. **Data Analysis**  
   Conduct an exploratory analysis of the crawled latest movie review data and compare it with the IMDB dataset. The analysis content includes the distribution of ratings, the correlation between the number of likes and dislikes, the change trend of the number of reviews in different time periods, word frequency statistics, and word cloud visualization analysis.

7. **Model Prediction**  
   Use the fine-tuned BERT model to perform sentiment classification prediction on the crawled latest movie review data and evaluate the performance of the model on the new data.

8. **Summary**  
   Summarize the overall process and results of the project, analyze the performance differences of the model on the IMDB dataset and the latest movie review data. Discuss the advantages and disadvantages of the model and propose future improvement directions (such as introducing more data augmentation techniques, trying other pre-trained models, etc.).

---

### 🔎 Project Highlights

1. **Combination of Classic and Latest Data**  
   Not only uses the classic IMDB dataset but also introduces the latest movie review data to verify the generalization ability of the model in real-world scenarios.

2. **Complete NLP Process**  
   From data preprocessing, model fine-tuning, data crawling to model prediction, it covers the whole process of natural language processing projects.

3. **Technical Depth**  
   Use the BERT model for fine-tuning, demonstrating the powerful ability of deep learning in sentiment analysis tasks.

4. **Practical Application Value**  
   Through sentiment analysis, it provides valuable user insights for movie producers, marketing teams, etc.

## 🎬 Quick Start

### 📝 Prerequisites

- If you need the complete 50,000 pieces of data, you can download it yourself from the [mirror site](https://ai.stanford.edu/~amaas/data/sentiment/). For the convenience of uploading, this article provides a minidatasets with 5,000 pieces of data (already preprocessed and can be used directly).
- PyTorch (You can install it on the [PyTorch official website](https://pytorch.org/) according to the configuration of your computer).
- CUDA (If you have a GPU and want to use it to accelerate the model training, you need to install CUDA yourself).
- bert-base-uncased (You can download the model to your local computer from the [huggingface official website](https://huggingface.co/google-bert/bert-base-uncased)).
  > If you cannot access the huggingface official website, you can also install it yourself from the [mirror site](https://hf-mirror.com/).

### 🚀 Execution Order

1. **IMDbDataProcessing.py**  
   If you have downloaded the IMDB source data, you can call this program to process all txt files into an Excel table. If you choose to use the minidatasets directly, you can skip this step.

2. **dataAnalysis.ipynb**  
   Conduct a preliminary exploratory data analysis (EDA) and visualization analysis of the data.

3. **SentimentAnalysis.ipynb**  
   Create a model, fine-tune and train the model, verify it, and conduct practical applications to view the prediction effect of the model.

4. **IMDbDataCrawler.ipynb**  
   Crawl a total of 921 pieces of data from ten movies from 2021 to 2025 on the IMDB official website.

5. **SaveBestModel.py**  
   You can experiment with multiple models and finally save the best-performing model to your local computer for subsequent calls.

6. **dataAnalysis.ipynb**  
   Also conduct a preliminary exploratory data analysis (EDA) and visualization analysis of the crawled data.

7. **Model_Prediction.ipynb**  
   Call the best model obtained in step 5 for prediction and view the prediction effect of the model. 
