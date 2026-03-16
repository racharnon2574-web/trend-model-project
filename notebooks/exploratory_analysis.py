import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# อ่านไฟล์ Excel
df = pd.read_excel("raw_data/tourist_data.xlsx")

# แปลง date ให้ py อ่านได้
df['Date'] = pd.to_datetime(df['Date_changeFormat'])

# sort data 
df = df.sort_values("Date")
df = df.sort_values("Date")
df.set_index("Date", inplace=True)
# ดูข้อมูลว่าดึงมาได้ไหม
# print(df.head())
# print(df.info())
# print(df.describe())

# ------

plt.figure(figsize=(15,10))

# ------------------
# Graph 1 : Trend
# ------------------
plt.subplot(3,1,1)

plt.plot(df["TOTAL"])

plt.title("Tourist Trend (2021-2025)")
plt.xlabel("Date")
plt.ylabel("Total Tourists")


# ------------------
# Graph 2 : Monthly pattern
# ------------------
plt.subplot(3,1,2)

df.groupby("Month_num")["TOTAL"].mean().plot(kind="bar")

plt.title("Average Tourists by Month")
plt.xlabel("Month")
plt.ylabel("Average Tourists")


# ------------------
# Graph 3 : Weekday pattern
# ------------------
plt.subplot(3,1,3)

df.groupby("DayW")["TOTAL"].mean().plot(kind="bar")

plt.title("Average Tourists by Weekday")
plt.xlabel("Day of Week")
plt.ylabel("Average Tourists")


plt.tight_layout()

plt.show()

# ------------------
# ACF / PACF ใน figure เดียวกัน (ซ้ายขวา)
# ------------------

fig, axes = plt.subplots(1,2, figsize=(14,5))

# ACF
plot_acf(df["TOTAL"], lags=40, ax=axes[0])
axes[0].set_title("ACF - Autocorrelation")

# PACF
plot_pacf(df["TOTAL"], lags=40, ax=axes[1])
axes[1].set_title("PACF - Partial Autocorrelation")

plt.tight_layout()
plt.show()