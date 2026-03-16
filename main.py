from data_loader import load_data
from feature_engineering import create_features

from models.sarima_model import run_sarima
from models.prophet_model import run_prophet
from models.xgboost_model import run_xgboost
from models.lightgbm_model import run_lightgbm

from evaluation.metrics import mape


# ------------------
# Load data
# ------------------

df = load_data("raw_data/tourist_data.xlsx")


# ------------------
# Feature engineering
# ------------------

df = create_features(df)

df = df.dropna()


# ------------------
# Train / Test split
# ------------------

train = df.loc["2022":"2024"]
test = df.loc["2025"]


# ------------------
# SARIMA
# ------------------

sarima_pred = run_sarima(
    train["TOTAL"],
    test["TOTAL"]
)

print("SARIMA MAPE:", mape(test["TOTAL"], sarima_pred))


# ------------------
# Prophet
# ------------------

prophet_pred = run_prophet(
    train,
    test
)

print("Prophet MAPE:", mape(test["TOTAL"], prophet_pred))


# ------------------
# Feature list
# ------------------

features = [

"dayofweek",
"month",
"weekofyear",
"dayofyear",

"is_weekend",
"is_holiday",

"lag_1",
"lag_7",
"lag_14",
"lag_30",

"rolling_mean_7",
"rolling_mean_14",
"rolling_mean_30",

"rolling_std_7"

]


X_train = train[features]
X_test = test[features]

y_train = train["TOTAL"]
y_test = test["TOTAL"]


# ------------------
# XGBoost
# ------------------

xgb_pred = run_xgboost(
    X_train,
    y_train,
    X_test
)

print("XGBoost MAPE:", mape(y_test, xgb_pred))


# ------------------
# LightGBM
# ------------------

lgb_pred = run_lightgbm(
    X_train,
    y_train,
    X_test
)

print("LightGBM MAPE:", mape(y_test, lgb_pred))