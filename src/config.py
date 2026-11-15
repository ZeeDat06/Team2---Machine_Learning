from pathlib import Path

# 1. Cố định tham số tái lập kết quả (Reproducibility)
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# 2. Cấu hình đường dẫn thư mục và tệp dữ liệu
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "german.data"
NUMERIC_DATA_PATH = DATA_DIR / "german.data-numeric"
FIGURES_DIR = BASE_DIR / "figures"

# 3. Danh sách 20 thuộc tính và cột nhãn mục tiêu theo german.doc
COLUMN_NAMES = [
    "checking_status",        # Attribute 1: (qualitative) Status of existing checking account
    "duration",               # Attribute 2: (numerical) Duration in month
    "credit_history",         # Attribute 3: (qualitative) Credit history
    "purpose",                # Attribute 4: (qualitative) Purpose
    "credit_amount",          # Attribute 5: (numerical) Credit amount
    "savings_status",         # Attribute 6: (qualitative) Savings account/bonds
    "employment",             # Attribute 7: (qualitative) Present employment since
    "installment_commitment", # Attribute 8: (numerical) Installment rate in % of disposable income
    "personal_status",        # Attribute 9: (qualitative) Personal status and sex
    "other_parties",          # Attribute 10: (qualitative) Other debtors / guarantors
    "residence_since",        # Attribute 11: (numerical) Present residence since
    "property_magnitude",     # Attribute 12: (qualitative) Property
    "age",                    # Attribute 13: (numerical) Age in years
    "other_payment_plans",    # Attribute 14: (qualitative) Other installment plans
    "housing",                # Attribute 15: (qualitative) Housing
    "existing_credits",       # Attribute 16: (numerical) Number of existing credits at this bank
    "job",                    # Attribute 17: (qualitative) Job
    "num_dependents",         # Attribute 18: (numerical) Number of people being liable to provide maintenance for
    "own_telephone",          # Attribute 19: (qualitative) Telephone
    "foreign_worker",         # Attribute 20: (qualitative) foreign worker
    "target"                  # Target: 1 = Good, 2 = Bad
]

# 4. Phân nhóm 7 biến số và 13 biến phân loại
NUMERICAL_COLS = [
    "duration", "credit_amount", "installment_commitment",
    "residence_since", "age", "existing_credits", "num_dependents"
]

CATEGORICAL_COLS = [
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment", "personal_status", "other_parties", "property_magnitude",
    "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker"
]

# 5. Cấu hình Ma trận chi phí (Cost Matrix) theo german.doc
# Nhãn sau khi chuẩn hóa: 0 = Good (gốc là 1), 1 = Bad (gốc là 2)
# Dự đoán Bad (1) thành Good (0) phạt 5; dự đoán Good (0) thành Bad (1) phạt 1
COST_FP = 1  # Thực tế Good (0) nhưng dự đoán nhầm thành Bad (1)
COST_FN = 5  # Thực tế Bad (1) nhưng dự đoán nhầm thành Good (0)