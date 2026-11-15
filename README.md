# Đánh Giá Rủi Ro Tín Dụng Khách Hàng (FIT 4013)

## 1. Giới thiệu đề tài
Dự án thuộc Nhóm đề tài Dữ liệu dạng bảng (Đề tài 11) của học phần Nhập môn Học máy (FIT 4013). Mục tiêu là xây dựng pipeline học máy phân loại rủi ro tín dụng khách hàng (Good/Bad) có xét đến ma trận chi phí sai số (Cost Matrix).

## 2. Nguồn dữ liệu (German Credit Dataset)
- Tác giả: GS. TS. Hans Hofmann (Đại học Hamburg).
- Quy mô: 1000 mẫu dữ liệu.
- Tệp sử dụng chính: data/german.data (20 thuộc tính gồm 7 biến số, 13 biến phân loại và 1 cột nhãn mục tiêu).
- Tệp đối chiếu: data/german.data-numeric (24 thuộc tính số hóa bởi Đại học Strathclyde).
- Đường dẫn tải gốc: UCI Machine Learning Repository - Statlog (German Credit Data).

## 3. Cấu trúc kho lưu trữ
- data/: Chứa dữ liệu thô (german.data, german.data-numeric, german.doc, Index).
- notebooks/: Chứa các Jupyter Notebook thực nghiệm từ EDA đến đánh giá mô hình.
- src/: Chứa các module Python chuẩn hóa pipeline và cấu hình tham số.
- figures/: Lưu trữ biểu đồ EDA, Confusion Matrix và đồ thị so sánh kết quả.
- reports/: Chứa mã nguồn LaTeX và bản PDF báo cáo hoàn chỉnh.
- app/: Mã nguồn ứng dụng demo dự đoán rủi ro tín dụng.

## 4. Hướng dẫn cài đặt và chạy chương trình
1. Cài đặt thư viện: `pip install -r requirements.txt`
2. Chạy khám phá dữ liệu: Mở và thực thi `notebooks/01_eda.ipynb`
3. Chạy tiền xử lý & huấn luyện: Thực thi `notebooks/02_preprocessing_modeling.ipynb`
4. Chạy phân tích lỗi: Thực thi `notebooks/03_evaluation_error_analysis.ipynb`
5. Khởi chạy ứng dụng demo: `streamlit run app/main.py`