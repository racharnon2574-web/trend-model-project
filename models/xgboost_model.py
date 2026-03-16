from xgboost import XGBRegressor

def run_xgboost(X_train, y_train, X_test):

    model = XGBRegressor(

        n_estimators=2000,
        learning_rate=0.01,

        max_depth=10,
        min_child_weight=5,

        subsample=0.8,
        colsample_bytree=0.8,

        gamma=1,
        random_state=42
    )
    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    return pred