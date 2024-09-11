# Zero-Shot-Multilingual-Sentiment-Classification
This project implements a zero-shot multilingual sentiment classification model using the Multilingual Universal Sentence Encoder (mUSE) to generate embeddings for text data in multiple languages. It uses a multilayer perceptron model built with PyTorch Lightning to classify sentiment as either positive or negative. The dataset used is a subset of the Yelp Polarity Dataset.

## Features
#### Multilingual Support: 
The model can predict sentiment in various languages like Arabic,
Chinese (PRC), Chinese (Taiwan), Dutch, English, Germanic, German, French, Italian, Latin, Portuguese, Spanish, Japanese, Korean, Russian, Polish, Thai, and Turkish leveraging mUSE.
#### Zero-Shot Learning: 
No language-specific fine-tuning required.
#### Sentiment Classification: 
Binary classification into positive or negative sentiment.
#### Transfer Learning: 
Universal Sentence Encoder for multilingual embedding generation.
#### PyTorch Lightning: 
For easy scalability and efficient model training.

## Dataset
[The Yelp Polarity Dataset](https://huggingface.co/datasets/fancyzhx/yelp_polarity)

## Model Architecture
The model consists of a simple multilayer perceptron (MLP) with the following layers:

#### Input: 
512-dimensional sentence embeddings from mUSE.
#### Hidden Layers: 
Two layers with 768 and 128 hidden units, respectively, followed by ReLU activation and dropout for regularization.
#### Output: 
A softmax layer for binary classification.

## Training & Evaluation
The model is trained using PyTorch Lightning with the following steps:

#### Embedding: 
Text is embedded using the Multilingual Universal Sentence Encoder.
#### Training: 
The model is trained using a cross-entropy loss and optimized with Adam optimizer.
#### Evaluation: 
Validation and test performance are tracked using accuracy.
