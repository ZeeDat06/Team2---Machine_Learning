import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, confusion_matrix

from src.config import RANDOM_STATE, CV_FOLDS, COST_FP, COST_FN
from src.preprocessing import build_pipeline_a, build_pipeline_b


def credit_cost_func(y_true, y_pred) -> float:
    """
    Tính tổng chi phí sai số tín dụng theo Cost Matrix trong german.doc:
    - Dự đoán Bad (1) thành Good (0) [FN]: phạt 5
    - Dự đoán Good (0) thành Bad (1) [FP]: phạt 1
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp * COST_FP + fn * COST_FN)


# Scorer dùng trong cross-validation (giá trị càng gần 0 tức chi phí càng thấp càng tốt)
neg_credit_cost_scorer = make_scorer(credit_cost_func, greater_is_better=False)


def get_base_models() -> dict:
    """Khởi tạo 3 thuật toán cốt lõi của học phần với cấu hình mặc định (Baseline)."""
    return {
        "K-NN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE)
    }


def get_param_grids() -> dict:
    """Định nghĩa không gian tìm kiếm siêu tham số cho GridSearchCV."""
    return {
        "K-NN": {
            "model__n_neighbors": [3, 5, 7, 11, 15, 21, 25],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"]
        },
        "Naive Bayes": {
            "model__var_smoothing": np.logspace(-9, -1, 9)
        },
        "Decision Tree": {
            "model__criterion": ["gini", "entropy"],
            "model__max_depth": [3, 5, 7, 10, None],
            "model__min_samples_split": [2, 10, 20],
            "model__min_samples_leaf": [1, 5, 10],
            "model__class_weight": [None, "balanced", {0: COST_FP, 1: COST_FN}]
        }
    }


def build_full_pipeline(stream: str, estimator) -> Pipeline:
    """Ghép nối Pipeline Tiền xử lý (Luồng A hoặc Luồng B) với Mô hình phân loại."""
    preprocessor = build_pipeline_a() if stream.upper() == "A" else build_pipeline_b()
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])


def evaluate_baselines_cv(X_train: pd.DataFrame, y_train: np.ndarray) -> pd.DataFrame:
    """Đánh giá các mô hình Baseline (tham số mặc định) trên Luồng A và Luồng B bằng 5-Fold CV."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "neg_cost": neg_credit_cost_scorer
    }

    records = []
    for stream in ["A", "B"]:
        for model_name, model in get_base_models().items():
            pipe = build_full_pipeline(stream, model)
            cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            records.append({
                "Stream": f"Luồng {stream}",
                "Model": model_name,
                "Stage": "Baseline (Default)",
                "CV_Accuracy": cv_res["test_accuracy"].mean(),
                "CV_Precision": cv_res["test_precision"].mean(),
                "CV_Recall": cv_res["test_recall"].mean(),
                "CV_F1": cv_res["test_f1"].mean(),
                "CV_ROC_AUC": cv_res["test_roc_auc"].mean(),
                "CV_Avg_Cost": -cv_res["test_neg_cost"].mean()
            })
    return pd.DataFrame(records)


def run_grid_search_experiments(X_train: pd.DataFrame, y_train: np.ndarray):
    """
    Thực thi GridSearchCV cho cả 3 thuật toán trên Luồng A và Luồng B.
    Trả về:
    - summary_df: Bảng tổng hợp kết quả CV tốt nhất
    - best_models: Từ điển lưu các Pipeline đã huấn luyện tối ưu
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    param_grids = get_param_grids()
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "neg_cost": neg_credit_cost_scorer
    }

    records = []
    best_models = {}

    for stream in ["A", "B"]:
        for model_name, base_model in get_base_models().items():
            pipe = build_full_pipeline(stream, base_model)
            grid = GridSearchCV(
                estimator=pipe,
                param_grid=param_grids[model_name],
                scoring=scoring,
                refit="f1",
                cv=cv,
                n_jobs=-1,
                return_train_score=False
            )
            grid.fit(X_train, y_train)

            idx = grid.best_index_
            res = grid.cv_results_
            key = f"Stream_{stream}_{model_name}"
            best_models[key] = grid.best_estimator_

            # Rút gọn tên tham số để hiển thị trong bảng báo cáo
            clean_params = {k.replace("model__", ""): v for k, v in grid.best_params_.items()}

            records.append({
                "Stream": f"Luồng {stream}",
                "Model": model_name,
                "Stage": "Tuned (GridSearchCV)",
                "Best_Params": str(clean_params),
                "CV_Accuracy": res["mean_test_accuracy"][idx],
                "CV_Precision": res["mean_test_precision"][idx],
                "CV_Recall": res["mean_test_recall"][idx],
                "CV_F1": res["mean_test_f1"][idx],
                "CV_ROC_AUC": res["mean_test_roc_auc"][idx],
                "CV_Avg_Cost": -res["mean_test_neg_cost"][idx]
            })

    return pd.DataFrame(records), best_models