from sklearn.model_selection import train_test_split

from src.data_preprocessing import preprocess_data
from src.model_training import get_models,train_and_comapare
from src.pipeline import create_pipeline
from src.hyperparameter_tuning import tune_model
from src.evalution import evaluate
from src.utils import save_object



def main():


    df = preprocess_data("data/Automobile_data.csv")

    X = df.drop("price",axis=1)
    y = df["price"]

    save_object('artifacts/features.pkl', X.columns.tolist())
    X_train,X_test,y_train,y_test = train_test_split(
       X,y,test_size=0.2,random_state=42
   )
    
    models = get_models() 
   
    best_name, best_model, results = train_and_comapare(
        models, X_train, X_test,y_train, y_test
    )
    

    if best_name == 'Random Forest':
        print('\n Tuning Best Model...')
        pipeline = create_pipeline()
        best_model = tune_model(pipeline, X_train, y_train)


    r2, rmse = evaluate(best_model, X_test, y_test)

    print('\n Final Model Performance')
    print('R2 Score:', r2)
    print('RMSE:', rmse)

    save_object('artifacts/model.pkl', best_model)

if __name__ == '__main__':
    main()
        