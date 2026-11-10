import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import (
    COMPARISON_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
)
from src.data_loader import load_data
from src.evaluate import classification_metrics, save_comparison
from src.preprocessing import build_preprocessor, split_features_target


def train(data_path=RAW_DATA_PATH) -> dict:
    data = load_data(data_path)
    features, target = split_features_target(data)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.assign(Churn=target).to_csv(PROCESSED_DATA_PATH, index=False)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    preprocessor = build_preprocessor(x_train)
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_test_transformed = preprocessor.transform(x_test)

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    comparison = []
    fitted_models = {}
    for name, candidate in candidates.items():
        candidate.fit(x_train_transformed, y_train)
        probabilities = candidate.predict_proba(x_test_transformed)[:, 1]
        row = {"model": name, **classification_metrics(y_test, candidate.predict(x_test_transformed), probabilities)}
        comparison.append(row)
        fitted_models[name] = candidate

    save_comparison(comparison, COMPARISON_PATH)
    best_name = max(comparison, key=lambda row: row["roc_auc"])["model"]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_models[best_name], MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    return {"best_model": best_name, "metrics": next(row for row in comparison if row["model"] == best_name)}


if __name__ == "__main__":
    print(train())
