from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


from src.evalution import evaluate


def get_models():
    return {
        "Linear Regresssion": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(max_iter=10000),
        "Decision Tree": DecisionTreeRegressor(max_depth=5),
        "Random Forest": RandomForestRegressor(random_state=42)
    }


def train_and_comapare(models,X_train,Xtest,X_test,y_train,y_test):
    results = []
    trained_models = {}

    for name,model in models.items():
        model.fit(X_train,y_train)

        r2,rmse = evaluate(model,X_test,y_test)

        print(f"{name}   R2: {r2:.3f},RMSE: {rmse:.2f}")

        results.append((name,r2,rmse))
        trained_models[name] = model

    # Sort by best R2
    results.sort(key=lambda x: x[1],reverse=True)

    best_name = results[0][0]
    best_model = trained_models[best_name]

    print("\n Best Model:", best_name)

    return best_name,best_model,results
