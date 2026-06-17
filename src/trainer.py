from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def train_model(model_choice, X_train, X_test, y_train):

    models = {
        "1": LinearRegression(),
        "2": DecisionTreeRegressor(random_state=42),
        "3": RandomForestRegressor(random_state=42)
    }

    model = models[model_choice]
    model_names = {
    "1": "Linear Regression",
    "2": "Decision Tree",
    "3": "Random Forest"
    }

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, y_pred, model_names[model_choice]