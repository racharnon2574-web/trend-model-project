import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# ----------------------
# Load data
# ----------------------

df = pd.read_excel("raw_data/tourist_data.xlsx")

df["Date"] = pd.to_datetime(df["Date_changeFormat"])

df = df.sort_values("Date")

df.set_index("Date", inplace=True)

# ----------------------
# Feature selection
# ----------------------

features = [
    "Month_num",
    "Week",
    "DayW",
    "Holiday"
]

X = df[features]

y = df["TOTAL"]

# ----------------------
# Train / Test split
# ----------------------

X_train = X.loc["2021":"2024"]
X_test = X.loc["2025"]

y_train = y.loc["2021":"2024"]
y_test = y.loc["2025"]

print("Train:",X_train.shape)
print("Test:",X_test.shape)

# ----------------------
# Train model
# ----------------------

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model.fit(X_train,y_train)

# ----------------------
# Predict
# ----------------------

pred = model.predict(X_test)

# ----------------------
# MAPE
# ----------------------

mape = np.mean(np.abs((y_test - pred) / y_test)) * 100

print("MAPE:",mape)

# ----------------------
# Plot result
# ----------------------

plt.figure(figsize=(12,5))

plt.plot(y_train.index,y_train,label="Train")

plt.plot(y_test.index,y_test,label="Actual")

plt.plot(y_test.index,pred,label="Prediction")

plt.legend()

plt.title("XGBoost Forecast")

plt.show()