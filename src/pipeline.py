from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


def create_pipeline():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",RandomForestRegressor(random_state=42))
    ])
    return pipe