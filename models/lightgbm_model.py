import lightgbm as lgb

def run_lightgbm(X_train, y_train, X_test):

    model = lgb.LGBMRegressor(

        n_estimators=3000,
        learning_rate=0.01,

        max_depth=-1,
        num_leaves=64,

        subsample=0.8,
        colsample_bytree=0.8,

        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    return pred