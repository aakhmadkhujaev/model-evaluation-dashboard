from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression


def train_model(X_train, X_test, y_train):

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, y_pred