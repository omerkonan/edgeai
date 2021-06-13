import random
from io import StringIO

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from .exceptions import *

import boto3

AI_JOB_BUCKET = "empacloud-ai-jobs"


class Preprocessor:
    def __init__(self, file_key: str, environment: str, job_id: str, feature_method: tuple = ("RAW", 0),
                 normalization: bool = False,
                 standardization: bool = False, test_rate: float = 0.2, val_rate: float = 0.2):
        self.environment = environment
        self.job_id = job_id
        self.file_key = file_key
        self.s3_input_path = f"s3://{AI_JOB_BUCKET}/{self.environment}/{self.job_id}/RAW/{self.file_key}"
        self.columns = None
        self.label_header = None
        self.input_file_name = self.s3_input_path.split("/")[-1].split(".")[0]
        self.df = None
        self.__download_and_read_data()
        self.label_list = self.__generate_labels()
        self.__label_encode()
        self.df_label = self.df[self.label_header]
        self.feature_method = feature_method
        self.normalization = normalization
        self.standardization = standardization
        self.df_feature = self.df.drop(self.label_header, axis=1)
        self.processed_data = self.__process_data()
        self.test_rate = test_rate
        self.val_rate = val_rate / (1 - test_rate)
        # self.x_train, self.y_train, self.x_test, self.y_test, self.x_val, self.y_val = self.__split_data()

    def __check_file_type(self):
        return self.s3_input_path.split(".")[-1]

    def __generate_labels(self):
        for column in self.columns:
            if str(column).lower() == 'label':
                self.label_header = column
                break

        if self.label_header:
            return pd.unique(self.df[self.label_header])
        else:
            raise LabelColumnNotFoundException()

    def __download_and_read_data(self):
        file_type = self.__check_file_type()
        if file_type == 'csv':
            self.df = pd.read_csv(self.s3_input_path, engine='c', error_bad_lines=False, warn_bad_lines=False)
        elif file_type == 'json':
            self.df = pd.read_json(self.s3_input_path)
        else:
            raise InvalidFileTypeException()

        self.df = self.df.dropna(axis=1)
        self.columns = self.df.columns
        self.df = self.df.dropna()

    def __label_encode(self):
        for i in range(len(self.label_list)):
            self.df = self.df.replace({self.label_header: self.label_list[i]}, i)

    def __process_data(self):
        if self.normalization is True and self.standardization is True:
            raise InvalidPreprocessParametersException()

        feature_method_name = self.feature_method[0]
        windows_length = self.feature_method[1]

        self.df_feature = self.df_feature.apply(pd.to_numeric, errors='coerce', downcast='float')

        null_df = self.df_feature.isnull()
        null_index = np.where(null_df == True)
        # TODO: Null Satırların kullanıcıya bilgilendirmesi yapılacak

        self.df_feature = pd.concat([self.df_label, self.df_feature], axis=1).dropna().reset_index(drop=True)
        self.df_label = self.df_feature[self.label_header]
        self.df_feature = self.df_feature.drop(self.label_header, axis=1)
        if feature_method_name == "RAW":
            pass
        elif feature_method_name == "SMA":
            self.df_feature = self.SMA(self.df_feature, windows_length)
            self.df_label = self.df_label.iloc[windows_length - 1:].reset_index(drop=True)
            self.input_file_name += "_SMA"
        elif feature_method_name == "SMV":
            self.df_feature = self.SMV(self.df_feature, windows_length)
            self.df_label = self.df_label.iloc[windows_length - 1:].reset_index(drop=True)
            self.input_file_name += "_SMV"
        elif feature_method_name == "CMA":
            self.df_feature = self.CMA(self.df_feature, windows_length)
            self.df_label = self.df_label.iloc[windows_length - 1:].reset_index(drop=True)
            self.input_file_name += "_CMA"
        elif feature_method_name == "CMV":
            self.df_feature = self.CMV(self.df_feature, windows_length)
            self.df_label = self.df_label.iloc[windows_length - 1:].reset_index(drop=True)
            self.input_file_name += "_CMV"
        elif feature_method_name == "EMA":
            self.df_feature = self.EMA(self.df_feature, windows_length)
            self.df_label = self.df_label.iloc[0:].reset_index(drop=True)
            self.input_file_name += "_EMA"
        elif feature_method_name == "EMV":
            self.df_feature = self.EMV(self.df_feature, windows_length)
            self.df_label = self.df_label.iloc[1:].reset_index(drop=True)
            self.input_file_name += "_EMV"
        else:
            raise FeatureMethodNotFoundException()

        self.input_file_name += f"_{windows_length}"

        if self.normalization:
            self.df_feature = self.__apply_normalization(self.df_feature)
            self.input_file_name += "_NORM"
        if self.standardization:
            self.df_feature = self.__apply_standardization(self.df_feature)
            self.input_file_name += "_STAN"

        return pd.concat([self.df_label, self.df_feature], axis=1)

    def SMA(self, df_feature, windows_length):
        """
        Simple Moving Average
        """
        df_SMA = pd.DataFrame()
        for col in df_feature.columns:
            if col != self.label_header:
                df_SMA[str(col)] = df_feature[str(col)].rolling(window=windows_length).mean()
        df_SMA.dropna(inplace=True)
        df_SMA = df_SMA.reset_index(drop=True)
        return df_SMA

    def SMV(self, df_feature, windows_length):
        """
        Simple Moving Variance
        """
        df_SMV = pd.DataFrame()
        for col in df_feature.columns:
            if col != self.label_header:
                df_SMV[str(col)] = df_feature[str(col)].rolling(window=windows_length).var()
        df_SMV.dropna(inplace=True)
        df_SMV = df_SMV.reset_index(drop=True)
        return df_SMV

    def CMA(self, df_feature, windows_length):
        """
        Cumulative Moving Average
        """
        df_CMA = pd.DataFrame()
        for col in df_feature.columns:
            if col != self.label_header:
                df_CMA[str(col)] = df_feature[str(col)].expanding(min_periods=windows_length).mean()
        df_CMA.dropna(inplace=True)
        df_CMA = df_CMA.reset_index(drop=True)
        return df_CMA

    def CMV(self, df_feature, windows_length):
        """
        Cumulative Moving Variance
        """
        df_CMV = pd.DataFrame()
        for col in df_feature.columns:
            if col != self.label_header:
                df_CMV[str(col)] = df_feature[str(col)].expanding(min_periods=windows_length).var()
        df_CMV.dropna(inplace=True)
        df_CMV = df_CMV.reset_index(drop=True)
        return df_CMV

    def EMA(self, df_feature, windows_length):
        """
        Exponential Moving Average
        """
        df_EMA = pd.DataFrame()
        for col in df_feature.columns:
            if col != self.label_header:
                df_EMA[str(col)] = df_feature[str(col)].ewm(span=windows_length, adjust=False).mean()
        df_EMA.dropna(inplace=True)
        df_EMA = df_EMA.reset_index(drop=True)
        return df_EMA

    def EMV(self, df_feature, windows_length):
        """
        Exponential Moving Varience
        """
        df_EMV = pd.DataFrame()
        for col in df_feature.columns:
            if col != self.label_header:
                df_EMV[str(col)] = df_feature[str(col)].ewm(span=windows_length, adjust=False).var()
        df_EMV.dropna(inplace=True)
        df_EMV = df_EMV.reset_index(drop=True)
        return df_EMV

    def __apply_normalization(self, df_feature):
        return pd.DataFrame(MinMaxScaler().fit_transform(df_feature),
                            columns=[x for x in self.df.columns if x != self.label_header])

    def __apply_standardization(self, df_feature):
        scaler = StandardScaler()
        scaler.fit(df_feature)
        return pd.DataFrame(scaler.transform(df_feature),
                            columns=[x for x in self.df.columns if x != self.label_header])

    def __split_data(self):
        x, x_test, y, y_test = train_test_split(self.df_feature, self.df_label, test_size=self.test_rate,
                                                train_size=(1 - self.test_rate))
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=self.val_rate, train_size=(1 - self.val_rate))
        return x_train, y_train, x_test, y_test, x_val, y_val

    def convert_conv1d(self):
        """
        Keras conv1d layer input shape should be 3d. This function adjust data dimension for con1d
        """
        x_train = np.expand_dims(self.x_train, axis=2)
        y_train = np.expand_dims(self.y_train, axis=1)
        x_test = np.expand_dims(self.x_test, axis=2)
        y_test = np.expand_dims(self.y_test, axis=1)
        x_val = np.expand_dims(self.x_val, axis=2)
        y_val = np.expand_dims(self.y_val, axis=1)
        return x_train, y_train, x_test, y_test, x_val, y_val

    def save(self, p_id):
        # TODO : Insert New Pre-Process

        s3 = boto3.client("s3", region_name="eu-west-1")
        output_path = f"{self.environment}/{self.job_id}/PROCESSED/{p_id}/{self.input_file_name}.csv"
        csv_buf = StringIO()
        self.processed_data.to_csv(csv_buf, header=True, index=False)
        csv_buf.seek(0)
        s3.put_object(Bucket=AI_JOB_BUCKET, Body=csv_buf.getvalue(), Key=output_path)
        return 1
