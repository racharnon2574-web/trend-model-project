import numpy as np

def create_features(df):

    # ----------------
    # time features
    # ----------------

    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["weekofyear"] = df.index.isocalendar().week
    df["dayofyear"] = df.index.dayofyear

    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)


    # ----------------
    # holiday
    # ----------------

    df["is_holiday"] = (df["Holiday"] > 0).astype(int)


    # ----------------
    # lag features
    # ----------------

    df["lag_1"] = df["TOTAL"].shift(1)
    df["lag_7"] = df["TOTAL"].shift(7)
    df["lag_14"] = df["TOTAL"].shift(14)
    df["lag_30"] = df["TOTAL"].shift(30)


    # ----------------
    # rolling mean
    # ----------------

    df["rolling_mean_7"] = df["TOTAL"].shift(1).rolling(7).mean()
    df["rolling_mean_14"] = df["TOTAL"].shift(1).rolling(14).mean()
    df["rolling_mean_30"] = df["TOTAL"].shift(1).rolling(30).mean()


    # ----------------
    # rolling std
    # ----------------

    df["rolling_std_7"] = df["TOTAL"].shift(1).rolling(7).std()

    return df