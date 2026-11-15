import sys
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.config import RANDOM_STATE
from src.data_utils import load_raw_data
from src.preprocessing import split_train_test, build_pipeline_b
from src.evaluation import evaluate_models_on_test


def run_extension_experiments():
    """So sánh Kỹ thuật mở rộng (PCA + K-NN, Random Forest, Gradient Boosting) trên Luồng B."""
    df_raw = load_raw_data()
    X_train, X_test, y_train, y_test = split_train_test(df_raw)

    ext_models = {
        "Stream_B_PCA_KNN (15 Components)": Pipeline([
            ("preprocessor", build_pipeline_b()),
            ("pca", PCA(n_components=15, random_state=RANDOM_STATE)),
            ("model", KNeighborsClassifier(n_neighbors=15, weights="distance"))
        ]),
        "Stream_B_RandomForest (Cost-Sensitive)": Pipeline([
            ("preprocessor", build_pipeline_b()),
            ("model", RandomForestClassifier(
                n_estimators=200, max_depth=8,
                class_weight={0: 1, 1: 5}, random_state=RANDOM_STATE
            ))
        ]),
        "Stream_B_GradientBoosting": Pipeline([
            ("preprocessor", build_pipeline_b()),
            ("model", GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.05,
                max_depth=3, random_state=RANDOM_STATE
            ))
        ])
    }

    for name, pipe in ext_models.items():
        pipe.fit(X_train, y_train)

    res_df = evaluate_models_on_test(ext_models, X_test, y_test)
    print("=== KẾT QUẢ THỬ NGHIỆM KỸ THUẬT MỞ RỘNG (PCA & ENSEMBLE) TRÊN TẬP TEST ===")
    print(res_df[["Model", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "Total_Cost"]].to_string(index=False))
    return res_df


if __name__ == "__main__":
    run_extension_experiments()