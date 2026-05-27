import numpy as np
from sklearn.metrics import r2_score,mean_squared_error


def evaluate(model,X_test,y_test):
    preds = model.predict(X_test)

    r2 = r2_score(y_test,preds)
    rmse  = np.sqrt(mean_squared_error(y_test,preds))

    return r2,rmse