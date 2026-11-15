import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from src.config import COST_FP, COST_FN


def evaluate_models_on_test(models_dict: dict, X_test: pd.DataFrame, y_test: np.ndarray) -> pd.DataFrame:
    """
    Đánh giá danh sách các mô hình trên tập kiểm thử (Test set).
    Tính toán đầy đủ các độ đo phân loại và Chi phí sai số tín dụng (Cost Matrix).
    """
    records = []
    for key, model in models_dict.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        total_cost = fp * COST_FP + fn * COST_FN
        avg_cost = total_cost / len(y_test)

        # Phân tách thông tin Luồng và Tên mô hình từ khóa định danh
        if key.startswith("Baseline_A_"):
            stream = "Luồng A (Baseline)"
            model_name = key.replace("Baseline_A_", "")
        elif key.startswith("Stream_A_"):
            stream = "Luồng A (Tuned)"
            model_name = key.replace("Stream_A_", "")
        elif key.startswith("Stream_B_"):
            stream = "Luồng B (Tuned)"
            model_name = key.replace("Stream_B_", "")
        else:
            stream = "Khác"
            model_name = key

        records.append({
            "Key": key,
            "Stream_Stage": stream,
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1_Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, y_prob),
            "TN": int(tn),
            "FP (Cost=1)": int(fp),
            "FN (Cost=5)": int(fn),
            "TP": int(tp),
            "Total_Cost": int(total_cost),
            "Avg_Cost": avg_cost
        })

    return pd.DataFrame(records)


def get_error_analysis_df(model, X_test: pd.DataFrame, y_test: np.ndarray, df_decoded: pd.DataFrame) -> pd.DataFrame:
    """
    Kết hợp kết quả dự đoán trên tập Test với dữ liệu đã giải mã để phân tích lỗi.
    Phân loại rõ 4 nhóm: TP, TN, FP (Nhầm khách Tốt thành Xấu) và FN (Sót khách Xấu thành Tốt).
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    analysis_df = df_decoded.loc[X_test.index].copy()
    analysis_df["Actual"] = y_test
    analysis_df["Predicted"] = y_pred
    analysis_df["Prob_Bad"] = np.round(y_prob, 4)

    conditions = [
        (analysis_df["Actual"] == 1) & (analysis_df["Predicted"] == 1),
        (analysis_df["Actual"] == 0) & (analysis_df["Predicted"] == 0),
        (analysis_df["Actual"] == 0) & (analysis_df["Predicted"] == 1),
        (analysis_df["Actual"] == 1) & (analysis_df["Predicted"] == 0)
    ]
    choices = [
        "TP (Đúng khách Xấu)",
        "TN (Đúng khách Tốt)",
        "FP (Nhầm Tốt -> Xấu | Phạt 1)",
        "FN (Sót Xấu -> Tốt | Phạt 5)"
    ]
    analysis_df["Error_Type"] = np.select(conditions, choices, default="Unknown")
    return analysis_df