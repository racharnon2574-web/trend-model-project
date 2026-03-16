import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import numpy as np

# ----------------------
# Load data
# ----------------------

df = pd.read_excel("raw_data/tourist_data.xlsx")

# แปลง date
df["Date"] = pd.to_datetime(df["Date_changeFormat"])

# เรียงข้อมูล
df = df.sort_values("Date")

# ตั้ง index
df.set_index("Date", inplace=True)


# ----------------------
# Train / Test split
# ----------------------

train = df.loc["2021":"2024"]
test = df.loc["2025"]

print("Train size:", train.shape)
print("Test size:", test.shape)


# ----------------------
# Train SARIMA
# ----------------------

model = SARIMAX(
    train["TOTAL"],
    order=(1,1,1),
    seasonal_order=(1,1,1,7)
)

result = model.fit()

print(result.summary())


# ----------------------
# Forecast
# ----------------------

forecast = result.forecast(steps=len(test))


# ----------------------
# Plot result
# ----------------------

plt.figure(figsize=(12,5))

plt.plot(train.index, train["TOTAL"], label="Train")
plt.plot(test.index, test["TOTAL"], label="Actual")
plt.plot(test.index, forecast, label="Forecast")

plt.legend()

plt.title("SARIMA Forecast")

plt.show()

mape = np.mean(np.abs((test["TOTAL"] - forecast) / test["TOTAL"])) * 100

print("MAPE:", mape)