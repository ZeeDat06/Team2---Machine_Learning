import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    PowerTransformer,
    OneHotEncoder,
    OrdinalEncoder
)
from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    NUMERICAL_COLS,
    CATEGORICAL_COLS
)

def split_train_test(df: pd.DataFrame):
    """Tách dữ liệu Train/Test phân tầng (Stratified) chống Data Leakage."""
    X = df.drop(columns=["target"])
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Bộ tạo đặc trưng tài chính mới cho Luồng B (Feature Engineering)."""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        # 1. Áp lực trả nợ trung bình mỗi tháng (Số tiền vay / Kỳ hạn tháng)
        X_out["monthly_payment"] = X_out["credit_amount"] / np.maximum(X_out["duration"], 1)
        # 2. Tỷ lệ khoản vay so với độ tuổi khách hàng
        X_out["credit_to_age"] = X_out["credit_amount"] / np.maximum(X_out["age"], 18)
        # 3. Chỉ số gánh nặng tín dụng kết hợp (% thu nhập trả góp * số khoản vay hiện tại)
        X_out["debt_burden_index"] = X_out["installment_commitment"] * X_out["existing_credits"]
        return X_out


def build_pipeline_a() -> ColumnTransformer:
    """
    LUỒNG A: Pipeline cơ bản
    - Biến số (7 cột): SimpleImputer(median) + StandardScaler
    - Biến phân loại (13 cột): SimpleImputer(most_frequent) + OneHotEncoder
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor_a = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_COLS),
            ("cat", cat_pipeline, CATEGORICAL_COLS)
        ],
        remainder="drop"
    )
    return preprocessor_a


def build_pipeline_b() -> Pipeline:
    """
    LUỒNG B: Pipeline cải tiến
    - Bước 1: Tạo 3 đặc trưng tài chính mới (monthly_payment, credit_to_age, debt_burden_index)
    - Bước 2: Biến số (10 cột) -> Imputer + PowerTransformer (khử lệch phải) + RobustScaler (chống ngoại lệ)
    - Bước 3: Biến phân loại có thứ bậc (4 cột) -> OrdinalEncoder theo đúng thứ tự kinh tế trong german.doc
    - Bước 4: Biến phân loại định danh (9 cột) -> OneHotEncoder
    """
    engineered_num_cols = NUMERICAL_COLS + [
        "monthly_payment", "credit_to_age", "debt_burden_index"
    ]

    ordinal_cols = ["checking_status", "savings_status", "employment", "job"]
    ordinal_categories = [
        ["A11", "A12", "A13", "A14"],          # checking_status
        ["A65", "A61", "A62", "A63", "A64"],   # savings_status
        ["A71", "A72", "A73", "A74", "A75"],   # employment
        ["A171", "A172", "A173", "A174"]       # job
    ]

    nominal_cols = [col for col in CATEGORICAL_COLS if col not in ordinal_cols]

    num_pipeline_b = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("power", PowerTransformer(method="yeo-johnson")),
        ("robust", RobustScaler())
    ])

    ord_pipeline_b = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    nom_pipeline_b = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])

    col_transformer_b = ColumnTransformer(
        transformers=[
            ("num", num_pipeline_b, engineered_num_cols),
            ("ord", ord_pipeline_b, ordinal_cols),
            ("nom", nom_pipeline_b, nominal_cols)
        ],
        remainder="drop"
    )

    preprocessor_b = Pipeline([
        ("feature_engineering", CreditFeatureEngineer()),
        ("preprocessor", col_transformer_b)
    ])
    return preprocessor_b