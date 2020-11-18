import pandas as pd
def SMA(df_feature, windows_length):
    
    """
    Simple Moving Average
    """
    df_SMA = pd.DataFrame()
    df_SMA[:,0] = df_feature[:,0].rolling(window=windows_length).mean() 
    df_SMA[:,1] = df_feature[:,1].rolling(window=windows_length).mean()
    df_SMA[:,2] = df_feature[:,2].rolling(window=windows_length).mean()

    return df_SMA
def CMA(df_feature, windows_length):
    """
    Cumulative Moving Average
    """
    df_CMA = pd.DataFrame()
    df_CMA[:,0] = df_feature[:,0].expanding(min_periods=windows_length).mean() 
    df_CMA[:,1] = df_feature[:,1].expanding(min_periods=windows_length).mean()
    df_CMA[:,2] = df_feature[:,2].expanding(min_periods=windows_length).mean()

    return df_CMA

def EMA(df_feature, windows_length):
    """
    Exponentiel Moving Average
    """
    df_EMA = pd.DataFrame(windows_length)
    df_EMA[:,0] = df_feature[:,0].ewm(span=windows_length, adjust=False).mean() 
    df_EMA[:,1] = df_feature[:,1].ewm(span=windows_length, adjust=False).mean()
    df_EMA[:,2] = df_feature[:,2].ewm(span=windows_length, adjust=False).mean()

    return df_EMA

def SMV(df_feature, windows_length):
    
    """
    Simple Moving Variance
    """
    df_SMV = pd.DataFrame()
    df_SMV[:,0] = df_feature[:,0].rolling(window=windows_length).var() 
    df_SMV[:,1] = df_feature[:,1].rolling(window=windows_length).var()
    df_SMV[:,2] = df_feature[:,2].rolling(window=windows_length).var()

    return df_SMV

def SMK(df_feature, windows_length):
    
    """
    Simple Moving Kurtosis
    """
    df_SMK = pd.DataFrame()
    df_SMK[:,0] = df_feature[:,0].rolling(window=windows_length).Kurtosis()
    df_SMK[:,1] = df_feature[:,1].rolling(window=windows_length).Kurtosis()
    df_SMK[:,2] = df_feature[:,2].rolling(window=windows_length).Kurtosis()

    return df_SMK

def SMS(df_feature, windows_length):
    
    """
    Simple Moving Skew
    """
    df_SMS = pd.DataFrame()
    df_SMS[:,0] = df_feature[:,0].rolling(window=windows_length).Skew()
    df_SMS[:,1] = df_feature[:,1].rolling(window=windows_length).Skew()
    df_SMS[:,2] = df_feature[:,2].rolling(window=windows_length).Skew()

    return df_SMS