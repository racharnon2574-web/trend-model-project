from data_loader import load_data
from feature_engineering import create_features

from models.sarima_model import run_sarima
from models.prophet_model import run_prophet
from models.xgboost_model import run_xgboost

from evaluation.metrics import mape


# -------------------
# Load data
# -------------------

df = load_data("raw_data/tourist_data.xlsx")

df = create_features(df)


# -------------------
# Train test split
# -------------------

train = df.loc["2021":"2024"]
test = df.loc["2025"]


# -------------------
# SARIMA
# -------------------

sarima_forecast = run_sarima(
    train["TOTAL"],
    test["TOTAL"]
)

sarima_mape = mape(test["TOTAL"], sarima_forecast)

print("SARIMA MAPE:", sarima_mape)


# -------------------
# Prophet
# -------------------

prophet_forecast = run_prophet(
    train,
    test
)

prophet_mape = mape(test["TOTAL"], prophet_forecast)

print("Prophet MAPE:", prophet_mape)


# -------------------
# XGBoost
# -------------------

features = [
    "month",
    "week",
    "dayofweek",
    "dayofyear",
    "is_weekend",

    "Holiday",
    "is_holiday",
    "is_long_holiday",

    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    "lag_2",
    "lag_3",
    "lag_365",
    

    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
    "rolling_mean_3",

    "rolling_std_14",
    "rolling_std_7"
]
df = create_features(df)

df = df.dropna()

X_train = train[features]
X_test = test[features]

y_train = train["TOTAL"]

xgb_pred = run_xgboost(
    X_train,
    y_train,
    X_test
)

xgb_mape = mape(test["TOTAL"], xgb_pred)

print("XGBoost MAPE:", xgb_mape)