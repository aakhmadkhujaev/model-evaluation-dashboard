from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def split_features_target(df, target_column):
    if target_column not in df.columns:
       raise ValueError(f"Target column '{target_column}' not found.")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def split_train_test(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=None
):
    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state,
    shuffle=True,
    stratify=stratify
)

    return X_train, X_test, y_train, y_test

def encode_features(X_train, X_test):
    categorical_columns = X_train.select_dtypes(include="object").columns
    numerical_columns = X_train.select_dtypes(include="number").columns
    preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numeric",
            StandardScaler(),
            numerical_columns
        )
    ]
    )
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return preprocessor, X_train_processed, X_test_processed
    

    