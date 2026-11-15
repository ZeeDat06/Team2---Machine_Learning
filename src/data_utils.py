import pandas as pd
from src.config import RAW_DATA_PATH, COLUMN_NAMES

# Từ điển ánh xạ mã thuộc tính phân loại theo tài liệu german.doc
ATTRIBUTE_MAPPING = {
    "checking_status": {
        "A11": "< 0 DM",
        "A12": "0 - 200 DM",
        "A13": ">= 200 DM / salary",
        "A14": "No checking account"
    },
    "credit_history": {
        "A30": "No credits / all paid",
        "A31": "All paid at this bank",
        "A32": "Existing paid duly",
        "A33": "Delay in past",
        "A34": "Critical / other credits"
    },
    "purpose": {
        "A40": "Car (new)",
        "A41": "Car (used)",
        "A42": "Furniture/equipment",
        "A43": "Radio/television",
        "A44": "Domestic appliances",
        "A45": "Repairs",
        "A46": "Education",
        "A47": "Vacation",
        "A48": "Retraining",
        "A49": "Business",
        "A410": "Others"
    },
    "savings_status": {
        "A61": "< 100 DM",
        "A62": "100 - 500 DM",
        "A63": "500 - 1000 DM",
        "A64": ">= 1000 DM",
        "A65": "Unknown / no savings"
    },
    "employment": {
        "A71": "Unemployed",
        "A72": "< 1 year",
        "A73": "1 - 4 years",
        "A74": "4 - 7 years",
        "A75": ">= 7 years"
    },
    "personal_status": {
        "A91": "Male: divorced/sep",
        "A92": "Female: div/sep/mar",
        "A93": "Male: single",
        "A94": "Male: mar/wid",
        "A95": "Female: single"
    },
    "other_parties": {
        "A101": "None",
        "A102": "Co-applicant",
        "A103": "Guarantor"
    },
    "property_magnitude": {
        "A121": "Real estate",
        "A122": "Savings/life insurance",
        "A123": "Car or other",
        "A124": "Unknown / no property"
    },
    "other_payment_plans": {
        "A141": "Bank",
        "A142": "Stores",
        "A143": "None"
    },
    "housing": {
        "A151": "Rent",
        "A152": "Own",
        "A153": "For free"
    },
    "job": {
        "A171": "Unskilled non-res",
        "A172": "Unskilled resident",
        "A173": "Skilled employee",
        "A174": "Management / highly qual"
    },
    "own_telephone": {
        "A191": "None",
        "A192": "Yes (registered)"
    },
    "foreign_worker": {
        "A201": "Yes",
        "A202": "No"
    }
}

def load_raw_data(filepath=RAW_DATA_PATH) -> pd.DataFrame:
    """Đọc dữ liệu german.data và chuẩn hóa nhãn: 0 = Good (gốc 1), 1 = Bad (gốc 2)."""
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMN_NAMES)
    df["target"] = df["target"].map({1: 0, 2: 1})
    return df

def load_decoded_data(filepath=RAW_DATA_PATH) -> pd.DataFrame:
    """Đọc dữ liệu và giải mã các biến phân loại để phục vụ vẽ biểu đồ EDA."""
    df = load_raw_data(filepath).copy()
    for col, mapping in ATTRIBUTE_MAPPING.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])
    df["target_label"] = df["target"].map({0: "Good (0)", 1: "Bad (1)"})
    return df