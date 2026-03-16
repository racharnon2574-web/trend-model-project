from prophet import Prophet
import pandas as pd

def run_prophet(train, test):

    df = train.reset_index()

    df = df[["Date","TOTAL"]]

    df.columns = ["ds","y"]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True
    )

    model.fit(df)

    future = model.make_future_dataframe(periods=len(test))

    forecast = model.predict(future)

    forecast = forecast.tail(len(test))

    return forecast["yhat"].values
