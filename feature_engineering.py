import numpy as np

def create_features(df):

    # -------------------------
    # Time features
    # -------------------------

    df["month"] = df.index.month
    df["week"] = df.index.isocalendar().week.astype(int)
    df["dayofweek"] = df.index.dayofweek
    df["dayofyear"] = df.index.dayofyear

    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)


    # -------------------------
    # Holiday features
    # -------------------------

    df["is_holiday"] = (df["Holiday"] > 0).astype(int)
    df["is_long_holiday"] = (df["Holiday"] == 2).astype(int)


    # -------------------------
    # Lag features
    # -------------------------

    df["lag_1"] = df["TOTAL"].shift(1)
    df["lag_7"] = df["TOTAL"].shift(7)
    df["lag_14"] = df["TOTAL"].shift(14)
    df["lag_30"] = df["TOTAL"].shift(30)
    df["lag_2"] = df["TOTAL"].shift(2)
    df["lag_3"] = df["TOTAL"].shift(3)
    df["lag_365"] = df["TOTAL"].shift(365)


    # -------------------------
    # Rolling mean
    # -------------------------

    df["rolling_mean_7"] = df["TOTAL"].shift(1).rolling(7).mean()
    df["rolling_mean_14"] = df["TOTAL"].shift(1).rolling(14).mean()
    df["rolling_mean_30"] = df["TOTAL"].shift(1).rolling(30).mean()
    df["rolling_mean_3"] = df["TOTAL"].shift(1).rolling(3).mean()
    df["rolling_std_14"] = df["TOTAL"].shift(1).rolling(14).std()


    # -------------------------
    # Rolling std (volatility)
    # -------------------------

    df["rolling_std_7"] = df["TOTAL"].shift(1).rolling(7).std()


    return df