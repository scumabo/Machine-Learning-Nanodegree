# Stock-Price-Prediction

## Getting Started

This project uses Pytorch's LSTM for making stock price predictions. It also depends on commonly used libraries pandas, numpy, yfinance, bs4, requests, and matplotlib. All the dependencies should be contained in requirement.txt. 

The project is developed in SageMaker with a Jupyter Notebook instance in conda_pytorch_p36 kernel. It should run in the mentioned environment without the need to install any libraries. 

To reproduce all results in the report:

1. Run "1. Data Preparation.ipynb" to download and clean the data.
2. Run "2. Stock Price Prediction.ipynb" to preprocessing the data, train the model, and evaluate the model. 