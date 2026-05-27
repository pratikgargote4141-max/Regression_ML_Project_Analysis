from sklearn.model_selection import GridSearchCV

def tune_model(pipeline,X_train,y_train):

    param_grid = {
        "model__n_estimators": [100,200],
        "model__max_depth": [5,10, None]
    }


    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    grid.fit(X_train,y_train)
    print("\n Best Parameters:",grid.best_params_)
    return grid.best_estimator_