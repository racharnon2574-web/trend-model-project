import pandas as pd

def load_data(path):

    df = pd.read_excel(path)

    df["Date"] = pd.to_datetime(df["Date_changeFormat"])

    df = df.sort_values("Date")

    df.set_index("Date", inplace=True)

    df = df.asfreq("D")

    return df