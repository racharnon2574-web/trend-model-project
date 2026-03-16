from statsmodels.tsa.statespace.sarimax import SARIMAX

def run_sarima(train, test):

    model = SARIMAX(
        train,
        order=(1,1,1),
        seasonal_order=(1,1,1,7)
    )

    result = model.fit()

    forecast = result.forecast(len(test))

    return forecast