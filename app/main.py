import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Thêm thư mục gốc vào sys.path để gọi các module trong src
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.config import COST_FP, COST_FN, NUMERICAL_COLS, CATEGORICAL_COLS
from src.data_utils import ATTRIBUTE_MAPPING, load_raw_data
from src.preprocessing import split_train_test, CreditFeatureEngineer
from src.modeling import run_grid_search_experiments, build_full_pipeline, get_base_models

st.set_page_config(
    page_title="Đánh Giá Rủi Ro Tín Dụng - FIT 4013",
    page_icon="🏦",
    layout="wide"
)


@st.cache_resource
def load_or_train_models():
    """Tải mô hình đã huấn luyện từ thư mục models/ hoặc tự động huấn luyện nếu chưa có."""
    models_dir = ROOT_DIR / "models"
    tuned_path = models_dir / "tuned_models.pkl"
    base_path = models_dir / "baseline_models.pkl"

    if tuned_path.exists() and base_path.exists():
        base_models = joblib.load(base_path)
        tuned_models = joblib.load(tuned_path)
        return {**tuned_models, **base_models}

    # Tự động huấn luyện nếu người dùng chạy app trước khi chạy Notebook 02
    df_raw = load_raw_data()
    X_train, _, y_train, _ = split_train_test(df_raw)
    _, tuned_models = run_grid_search_experiments(X_train, y_train)

    base_models = {}
    for name, model in get_base_models().items():
        pipe = build_full_pipeline("A", model)
        pipe.fit(X_train, y_train)
        base_models[f"Baseline_A_{name}"] = pipe

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(base_models, base_path)
    joblib.dump(tuned_models, tuned_path)
    return {**tuned_models, **base_models}


def format_option(col_name: str, code: str) -> str:
    """Hiển thị mã gốc kèm giải nghĩa tiếng Anh chuẩn từ german.doc."""
    desc = ATTRIBUTE_MAPPING.get(col_name, {}).get(code, code)
    return f"{code} — {desc}"


# --- TIÊU ĐỀ & THANH BÊN (SIDEBAR) ---
st.title("🏦 Hệ Thống Đánh Giá Rủi Ro Tín Dụng Khách Hàng")
st.markdown(
    "**Học phần:** Nhập môn Học máy (FIT 4013) — **Đề tài 11:** German Credit Dataset "
    "*(0 = Good Credit, 1 = Bad Credit)*"
)

all_models = load_or_train_models()

st.sidebar.header("⚙️ Cấu hình Mô hình & Chi phí")
model_labels = {
    "Stream_B_Decision Tree": "Luồng B (Cải tiến) — Decision Tree (Tuned)",
    "Stream_B_Naive Bayes": "Luồng B (Cải tiến) — Naive Bayes (Tuned)",
    "Stream_B_K-NN": "Luồng B (Cải tiến) — K-NN (Tuned)",
    "Stream_A_Decision Tree": "Luồng A (Cơ bản) — Decision Tree (Tuned)",
    "Stream_A_Naive Bayes": "Luồng A (Cơ bản) — Naive Bayes (Tuned)",
    "Stream_A_K-NN": "Luồng A (Cơ bản) — K-NN (Tuned)",
    "Baseline_A_Decision Tree": "Baseline Luồng A — Decision Tree (Default)",
    "Baseline_A_Naive Bayes": "Baseline Luồng A — Naive Bayes (Default)",
    "Baseline_A_K-NN": "Baseline Luồng A — K-NN (Default)"
}

selected_key = st.sidebar.selectbox(
    "Chọn Pipeline & Thuật toán:",
    options=list(model_labels.keys()),
    format_func=lambda k: model_labels.get(k, k)
)

# Ngưỡng lý thuyết tối ưu theo Cost Matrix: FP/(FP+FN) = 1/(1+5) = 0.1667
theoretical_threshold = round(COST_FP / (COST_FP + COST_FN), 2)
decision_threshold = st.sidebar.slider(
    "Ngưỡng xác suất kết luận Nợ Xấu (Threshold):",
    min_value=0.10, max_value=0.90, value=0.35, step=0.05,
    help=f"Ngưỡng mặc định là 0.50. Do phạt FN={COST_FN} gấp 5 lần FP={COST_FP}, hạ ngưỡng xuống (gợi ý 0.20 - 0.35) giúp giảm rủi ro bỏ sót nợ xấu."
)

st.sidebar.info(
    f"**Ma trận chi phí (Cost Matrix):**\n"
    f"- Dự đoán nhầm Good thành Bad (FP): **Phạt {COST_FP}**\n"
    f"- Bỏ sót Bad thành Good (FN): **Phạt {COST_FN}**\n"
    f"- Ngưỡng lý thuyết tối ưu chi phí: **{theoretical_threshold}**"
)

# --- MẪU HỒ SƠ THỬ NGHIỆM NHANH ---
st.subheader("1. Chọn nhanh hồ sơ mẫu hoặc tự nhập thông tin")
preset = st.radio(
    "Tải dữ liệu mẫu để kiểm thử nhanh khi vấn đáp:",
    options=[
        "Hồ sơ Tùy chỉnh (Mặc định)",
        "Mẫu Khách hàng Tốt điển hình (Rủi ro thấp)",
        "Mẫu Khách hàng Nợ xấu điển hình (Rủi ro cao)"
    ],
    horizontal=True
)

if preset == "Mẫu Khách hàng Nợ xấu điển hình (Rủi ro cao)":
    defaults = dict(
        checking="A11", duration=48, history="A33", purpose="A40",
        amount=9500, savings="A61", employment="A72", installment=4,
        personal="A93", debtors="A101", residence=2, prop="A124",
        age=23, plans="A141", housing="A151", credits=2,
        job="A173", dependents=1, phone="A191", foreign="A201"
    )
else:
    defaults = dict(
        checking="A14", duration=12, history="A34", purpose="A43",
        amount=1800, savings="A64", employment="A75", installment=2,
        personal="A93", debtors="A101", residence=4, prop="A121",
        age=38, plans="A143", housing="A152", credits=1,
        job="A173", dependents=1, phone="A192", foreign="A201"
    )

# --- BIỂU MẪU NHẬP 20 THUỘC TÍNH ---
with st.form("credit_application_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 💰 Thông tin khoản vay & Tài khoản")
        checking_status = st.selectbox(
            "1. Trạng thái tài khoản vãng lai (checking_status)",
            options=list(ATTRIBUTE_MAPPING["checking_status"].keys()),
            index=list(ATTRIBUTE_MAPPING["checking_status"].keys()).index(defaults["checking"]),
            format_func=lambda x: format_option("checking_status", x)
        )
        duration = st.number_input("2. Thời hạn vay - tháng (duration)", min_value=4, max_value=72, value=defaults["duration"])
        credit_amount = st.number_input("5. Số tiền vay - DM (credit_amount)", min_value=250, max_value=20000, value=defaults["amount"], step=100)
        purpose = st.selectbox(
            "4. Mục đích vay (purpose)",
            options=list(ATTRIBUTE_MAPPING["purpose"].keys()),
            index=list(ATTRIBUTE_MAPPING["purpose"].keys()).index(defaults["purpose"]),
            format_func=lambda x: format_option("purpose", x)
        )
        installment_commitment = st.slider("8. Tỷ lệ trả góp % thu nhập (installment_commitment)", 1, 4, defaults["installment"])
        savings_status = st.selectbox(
            "6. Tài khoản tiết kiệm (savings_status)",
            options=list(ATTRIBUTE_MAPPING["savings_status"].keys()),
            index=list(ATTRIBUTE_MAPPING["savings_status"].keys()).index(defaults["savings"]),
            format_func=lambda x: format_option("savings_status", x)
        )
        other_payment_plans = st.selectbox(
            "14. Kế hoạch trả góp khác (other_payment_plans)",
            options=list(ATTRIBUTE_MAPPING["other_payment_plans"].keys()),
            index=list(ATTRIBUTE_MAPPING["other_payment_plans"].keys()).index(defaults["plans"]),
            format_func=lambda x: format_option("other_payment_plans", x)
        )

    with col2:
        st.markdown("##### 📜 Lịch sử tín dụng & Tài sản")
        credit_history = st.selectbox(
            "3. Lịch sử tín dụng (credit_history)",
            options=list(ATTRIBUTE_MAPPING["credit_history"].keys()),
            index=list(ATTRIBUTE_MAPPING["credit_history"].keys()).index(defaults["history"]),
            format_func=lambda x: format_option("credit_history", x)
        )
        existing_credits = st.slider("16. Số khoản vay hiện có tại ngân hàng (existing_credits)", 1, 4, defaults["credits"])
        other_parties = st.selectbox(
            "10. Người đồng vay / Bảo lãnh (other_parties)",
            options=list(ATTRIBUTE_MAPPING["other_parties"].keys()),
            index=list(ATTRIBUTE_MAPPING["other_parties"].keys()).index(defaults["debtors"]),
            format_func=lambda x: format_option("other_parties", x)
        )
        property_magnitude = st.selectbox(
            "12. Tài sản sở hữu (property_magnitude)",
            options=list(ATTRIBUTE_MAPPING["property_magnitude"].keys()),
            index=list(ATTRIBUTE_MAPPING["property_magnitude"].keys()).index(defaults["prop"]),
            format_func=lambda x: format_option("property_magnitude", x)
        )
        housing = st.selectbox(
            "15. Tình trạng nhà ở (housing)",
            options=list(ATTRIBUTE_MAPPING["housing"].keys()),
            index=list(ATTRIBUTE_MAPPING["housing"].keys()).index(defaults["housing"]),
            format_func=lambda x: format_option("housing", x)
        )
        residence_since = st.slider("11. Số năm cư trú tại địa chỉ hiện tại (residence_since)", 1, 4, defaults["residence"])

    with col3:
        st.markdown("##### 👤 Hồ sơ cá nhân & Việc làm")
        age = st.number_input("13. Độ tuổi (age)", min_value=18, max_value=80, value=defaults["age"])
        employment = st.selectbox(
            "7. Thâm niên làm việc hiện tại (employment)",
            options=list(ATTRIBUTE_MAPPING["employment"].keys()),
            index=list(ATTRIBUTE_MAPPING["employment"].keys()).index(defaults["employment"]),
            format_func=lambda x: format_option("employment", x)
        )
        job = st.selectbox(
            "17. Trình độ công việc (job)",
            options=list(ATTRIBUTE_MAPPING["job"].keys()),
            index=list(ATTRIBUTE_MAPPING["job"].keys()).index(defaults["job"]),
            format_func=lambda x: format_option("job", x)
        )
        personal_status = st.selectbox(
            "9. Tình trạng hôn nhân & Giới tính (personal_status)",
            options=list(ATTRIBUTE_MAPPING["personal_status"].keys()),
            index=list(ATTRIBUTE_MAPPING["personal_status"].keys()).index(defaults["personal"]),
            format_func=lambda x: format_option("personal_status", x)
        )
        num_dependents = st.slider("18. Số người phụ thuộc (num_dependents)", 1, 2, defaults["dependents"])
        own_telephone = st.selectbox(
            "19. Đăng ký số điện thoại (own_telephone)",
            options=list(ATTRIBUTE_MAPPING["own_telephone"].keys()),
            index=list(ATTRIBUTE_MAPPING["own_telephone"].keys()).index(defaults["phone"]),
            format_func=lambda x: format_option("own_telephone", x)
        )
        foreign_worker = st.selectbox(
            "20. Lao động nước ngoài (foreign_worker)",
            options=list(ATTRIBUTE_MAPPING["foreign_worker"].keys()),
            index=list(ATTRIBUTE_MAPPING["foreign_worker"].keys()).index(defaults["foreign"]),
            format_func=lambda x: format_option("foreign_worker", x)
        )

    submitted = st.form_submit_button("🔍 Phân Tích & Dự Đoán Rủi Ro Tín Dụng", use_container_width=True)

# --- XỬ LÝ DỰ ĐOÁN & HIỂN THỊ KẾT QUẢ ---
if submitted:
    input_df = pd.DataFrame([{
        "checking_status": checking_status,
        "duration": duration,
        "credit_history": credit_history,
        "purpose": purpose,
        "credit_amount": credit_amount,
        "savings_status": savings_status,
        "employment": employment,
        "installment_commitment": installment_commitment,
        "personal_status": personal_status,
        "other_parties": other_parties,
        "residence_since": residence_since,
        "property_magnitude": property_magnitude,
        "age": age,
        "other_payment_plans": other_payment_plans,
        "housing": housing,
        "existing_credits": existing_credits,
        "job": job,
        "num_dependents": num_dependents,
        "own_telephone": own_telephone,
        "foreign_worker": foreign_worker
    }])

    model = all_models[selected_key]
    prob_bad = float(model.predict_proba(input_df)[0, 1])
    prob_good = 1.0 - prob_bad
    pred_default = int(model.predict(input_df)[0])
    pred_custom = 1 if prob_bad >= decision_threshold else 0

    # Tính chi phí kỳ vọng (Expected Cost) của 2 quyết định:
    # - Nếu duyệt vay (dự đoán Good = 0): rủi ro gặp khách Bad là prob_bad * COST_FN
    # - Nếu từ chối vay (dự đoán Bad = 1): rủi ro từ chối nhầm khách Good là prob_good * COST_FP
    expected_cost_approve = prob_bad * COST_FN
    expected_cost_reject = prob_good * COST_FP

    st.subheader("2. Kết quả Phán quyết Tín dụng")
    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.metric("Xác suất Nợ Xấu (Prob Bad)", f"{prob_bad * 100:.1f}%")
        st.progress(min(max(prob_bad, 0.0), 1.0))

    with res_col2:
        if pred_custom == 1:
            st.error(f"⚠️ **KẾT LUẬN: RỦI RO CAO - BAD (1)**\n\n(Vượt ngưỡng {decision_threshold:.2f} — Đề xuất: Từ chối / Yêu cầu tài sản đảm bảo)")
        else:
            st.success(f"✅ **KẾT LUẬN: AN TOÀN - GOOD (0)**\n\n(Dưới ngưỡng {decision_threshold:.2f} — Đề xuất: Phê duyệt khoản vay)")

    with res_col3:
        st.write("**Phân tích Chi phí Kỳ vọng (Cost Matrix):**")
        st.write(f"- Chi phí rủi ro nếu **Duyệt vay**: `{expected_cost_approve:.2f}` điểm")
        st.write(f"- Chi phí cơ hội nếu **Từ chối**: `{expected_cost_reject:.2f}` điểm")

    # Hiển thị 3 đặc trưng tài chính tạo mới của Luồng B
    engineered_df = CreditFeatureEngineer().transform(input_df)
    st.markdown("##### 📊 Chỉ số Tài chính Tái tạo (Feature Engineering - Luồng B)")
    f_col1, f_col2, f_col3 = st.columns(3)
    f_col1.info(f"**Áp lực trả góp hàng tháng (`monthly_payment`):** {engineered_df['monthly_payment'].iloc[0]:.1f} DM/tháng")
    f_col2.info(f"**Tỷ số Vay / Tuổi (`credit_to_age`):** {engineered_df['credit_to_age'].iloc[0]:.1f} DM/năm tuổi")
    f_col3.info(f"**Chỉ số Gánh nặng nợ (`debt_burden_index`):** {engineered_df['debt_burden_index'].iloc[0]}")