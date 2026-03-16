import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np

# ----------------------
# Load data
# ----------------------

df = pd.read_excel("raw_data/tourist_data.xlsx")

df["Date"] = pd.to_datetime(df["Date_changeFormat"])

df = df.sort_values("Date")

# Prophet ต้องใช้ column ds และ y
prophet_df = df[["Date","TOTAL"]]

prophet_df.columns = ["ds","y"]

# ----------------------
# Train / Test split
# ----------------------

train = prophet_df[prophet_df["ds"] < "2025-01-01"]
test = prophet_df[prophet_df["ds"] >= "2025-01-01"]

print("Train:",train.shape)
print("Test:",test.shape)

# ----------------------
# Create model
# ----------------------

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

model.fit(train)

# ----------------------
# Forecast
# ----------------------

future = model.make_future_dataframe(periods=365)

forecast = model.predict(future)

# ----------------------
# Plot forecast
# ----------------------

model.plot(forecast)

plt.title("Prophet Forecast")

plt.show()

# ----------------------
# Evaluate MAPE
# ----------------------

forecast_2025 = forecast[forecast["ds"] >= "2025-01-01"]

mape = np.mean(np.abs((test["y"].values - forecast_2025["yhat"].values) / test["y"].values)) * 100

print("MAPE:",mape)