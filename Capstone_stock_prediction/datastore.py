import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

from sklearn.preprocessing import MinMaxScaler
import os

class DataStore:
    """ The class preprocesses the data, holds the raw data, and provides data accessors

        Attributes:
            train (pd.DataFrame)            : Raw train data (time series)
            valid (pd.DataFrame)            : Raw valid data (time series)
            test  (pd.DataFrame)            : Raw test data  (time series)
            
            train_dict (Dictionary)         : Stores all normalized data for training and prediction for each ticker
                                              Usage: 
                                                 train_dict['DIS']['scaler'] (MinMaxScaler(copy=True, 
                                                                              feature_range=(0, 1))): 
                                                     Scaler to transform (normalize) and inverse transform data
                                                 
                                                 train_scaler['DIS']['yX'] (DataFrame): Data in model ready format
                                                     (Number of samples, len(lookahead_list) + lookback)
                                                     Each row is a sample. 
                                                     First 'len(lookahead_list)' columns store (true) output data 
                                                     for lookahead days of interest.
                                                     The rest 'lookback' columns store input data.
    """
    def __init__(self, data: pd.DataFrame, tickers: list, valid_size: float, test_size: float, lookback: int, lookahead: int):
        """
            Parameters
            ----------
            data           : Stock universe
            tickers        : A list of stocks to be included in the model
            valid_size     : The proportion of the data used for the validation dataset
            test_size      : The proportion of the data used for the test dataset
            lookback       : Number of days to lookback
            lookahead_list : List of lookahead days. len(lookahead_list) determines the size of output sequence
        """
        self.train, self.valid, self.test = self.train_valid_test_split(data, tickers, valid_size, test_size)
        
        # Stores all normalized data for training and prediction for each ticker
        self.train_dict = self.get_features_and_labels(self.train, lookback, lookahead)
        self.valid_dict = self.get_features_and_labels(self.valid, lookback, lookahead)
        self.test_dict  = self.get_features_and_labels(self.test,  lookback, lookahead)
        
        # (Private) Normalized data in the format of time series for visualization
        self._train_normalized = self.normalize(self.train)
        self._valid_normalized = self.normalize(self.valid)
        self._test_normalized = self.normalize(self.test)
    
    def get_features_and_labels(self, data: pd.DataFrame, lookback: int, lookahead_list: list):     
        """ Generate normalized features (X) and labels (y) for given data and return a dictionary 
            that stores the normalized data as well as corresponding scalers for (inverse) transform.
            
            Parameters:
            ----------
            data           : Time series of train, valid, or test data
            lookback       : Number of days to lookback which determines the size of input sequence
            lookahead_list : List of lookahead days. len(lookahead_list) determines the size of output sequence

            Returns:
            ---------
            data_dict : Dictionary contains scalar and normalized data ready to train/predict for each ticker:
                    
                     train_dict['DIS']['scaler'] (number of samples, 1): 
                         Scaler to transform and inverse transform data

                     train_scaler['DIS']['yX'] (DataFrame): 
                         (Number of samples, len(lookahead_list) + lookback)
                         Each row is a sample. 
                         First 'len(lookahead_list)' columns store (true) output data for lookahead days of 
                         interest.
                         The rest 'lookback' columns store input data.
        """
        data_dict = {}

        for ticker in data.columns:
            data_dict[ticker] = {}

            data_dict[ticker]['yX'] = []
            data_dict[ticker]['scaler'] = MinMaxScaler(feature_range=(0, 1))
            
            # Normalize data for the 'ticker' column
            normalized = data_dict[ticker]['scaler'].fit_transform(data[ticker].to_numpy().reshape(-1, 1)).squeeze()
            
            max_lookahead = max(lookahead_list)
            for i in range(lookback, len(data) - max_lookahead + 1):
                row = []
                for lookahead in lookahead_list:
                    row.append([normalized[i + lookahead - 1]])
                    
                row.append(normalized[i - lookback : i])
                
                row = np.concatenate(row)
                
                assert len(row) == (len(lookahead_list) + lookback)

                data_dict[ticker]['yX'].append(row)
            
            data_dict[ticker]['yX'] = pd.DataFrame(data_dict[ticker]['yX'])
            
        return data_dict
    
    def normalize(self, data: pd.DataFrame)-> pd.DataFrame:
        """ Since stock prices at different point in time may have different scales,
            we need to normalize the data before feeding it into the model. 
            
            Parameters:
            ----------
            data (number of dates, number of tickers): A time series of multiple stock ajdusted prices to be 
                                                       normalized.
            
            Returns:
            ----------
            normalized_data: Normalize by columns (or tickers) to range [0, 1]
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        return pd.DataFrame(scaler.fit_transform(data), index = data.index, columns = data.columns)
    
    def train_valid_test_split(self, data: pd.DataFrame, tickers: list, valid_size: float, test_size: float):
        """ Generate the train, validation, and test dataset for given tickers

            Parameters:
            ----------
            data        : Stock universe
            tickers     : A list of stocks to be included in the model
            valid_size  : The proportion of the data used for the validation dataset
            test_size   : The proportion of the data used for the test dataset

            Returns:
            -------
            train (DataFrame)  : The train samples
            valid (DataFrame)  : The validation samples
            test  (DataFrame)  : The test samples
        """
        assert valid_size >= 0 and valid_size <= 1.0
        assert test_size >= 0 and test_size <= 1.0

        train_size = 1 - valid_size - test_size
        n_days = len(data)
        train_num = int(train_size * n_days)
        valid_num = int(valid_size * n_days)

        train = data[:train_num][tickers]
        valid = data[train_num:train_num+valid_num][tickers]
        test  = data[train_num+valid_num:][tickers]

        return train, valid, test
    
    def save_train_valid_to_csv(self, tickers: list, data_dir: str)->None:
        """ Save training and validatoin data of all tickers to train.csv and valid.csv
            The first len(lookahead_list) columns represent labels and the rest columns represent features.
            
            Parameters
            ----------
            tickers     : A list of tickers. Here we can include tickers that we want to train/valid with.
        """
        train_data = []
        valid_data = []
        for ticker in tickers:
            train_data.append(self.train_dict[ticker]['yX'])
            valid_data.append(self.valid_dict[ticker]['yX'])
            
        pd.concat(train_data).to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)
        pd.concat(valid_data).to_csv(os.path.join(data_dir, 'valid.csv'), header=False, index=False)
        
    def plot(self, tickers: list, isRaw: bool = True)->None:
        """ Plot the train, validation, and test data for the given tickers
        
            Parameters
            ----------
            tickers     : A list of tickers
            isRaw       : If true, will plot the raw data otherwise normalized data.
        """
        for ticker in tickers:
            plt.figure(figsize=(14,4))
            if isRaw:
                plt.plot(self.train[ticker])
                plt.plot(self.valid[ticker])
                plt.plot(self.test[ticker])
                plt.ylabel("Ajusted Close")
                plt.xlabel("Date")
                plt.legend(["Training Set", "Validation Set", "Test Set"])
                plt.title(ticker + " Closing Stock Price")
            else:
                plt.plot(self._train_normalized[ticker])
                plt.plot(self._valid_normalized[ticker])
                plt.plot(self._test_normalized[ticker])
                plt.ylabel("Ajusted Close")
                plt.xlabel("Date")
                plt.legend(["Training Set (Normalized)", "Validation Set (Normalized)", "Test Set (Normalized)"])
                plt.title(ticker + " Closing Stock Price")