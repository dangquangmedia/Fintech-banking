# app.py — Demo chấm điểm tín dụng (Fintech & Banking)
import os, hashlib, json
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Chấm điểm tín dụng – Fintech & Banking", page_icon="💳", layout="centered")

# ====== cấu hình ======
MODEL_PATHS = ["models/xgboost.pkl", "models/logistic_regression.pkl", "models/random_forest.pkl"]
FAKE_DB_PATH = Path("data/processed/fake_db.csv")         # DB giả lập khách hàng (đã hash)
SALT_PATH = Path("data/processed/salt.json")               # muối để hash
TOPK_SHAP = 6

# ====== utils ======
def load_model():
    for p in MODEL_PATHS:
        if Path(p).exists():
            return joblib.load(p), p
    return None, None

def sha256_with_salt(raw: str, salt: str) -> str:
    return hashlib.sha256((salt + str(raw)).encode("utf-8")).hexdigest()

def fake_lookup(identifier: str) -> dict | None:
    """Tra hồ sơ giả lập theo SĐT/CCCD đã băm; trả về feature row nếu khớp."""
    if not (FAKE_DB_PATH.exists() and SALT_PATH.exists()):
        return None
    try:
        salt = json.loads(SALT_PATH.read_text(encoding="utf-8"))["salt"]
    except Exception:
        return None
    token = sha256_with_salt(identifier, salt)
    df = pd.read_csv(FAKE_DB_PATH)
    hit = df[df["token"] == token]
    if hit.empty:
        return None
    row = hit.iloc[0].drop(labels=["token"]).to_dict()
    return row

def ensure_feature_frame(d: dict) -> pd.DataFrame:
    cols = ['Age','Annual_Income','Monthly_Inhand_Salary','Num_Bank_Accounts',
            'Num_Credit_Card','Interest_Rate','Num_of_Loan',
            'Occupation','Type_of_Loan','Delay_from_due_date']
    for c in cols:
        d.setdefault(c, 0 if c not in ['Occupation','Type_of_Loan'] else 'Unknown')
    return pd.DataFrame([d])[cols]

def score_with_model(model, X: pd.DataFrame):
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
    classes = model.classes_ if hasattr(model, "classes_") else []
    return pred, proba, classes

def try_shap_plot(model, X: pd.DataFrame):
    try:
        import shap, numpy as np, matplotlib.pyplot as plt
        st.subheader("Giải thích mô hình (SHAP)")
        # Tương thích pipeline: lấy model gốc nếu có
        clf = getattr(model, "named_steps", {}).get("clf", None) or getattr(model, "named_steps", {}).get("classifier", None)
        pre = getattr(model, "named_steps", {}).get("pre", None) or getattr(model, "named_steps", {}).get("preprocessor", None)
        data_proc = pre.transform(X) if pre is not None else X.values
        explainer = shap.Explainer(clf) if clf is not None else shap.Explainer(model)
        sv = explainer(data_proc)
        shap.plots.bar(sv.abs.mean(0), max_display=TOPK_SHAP, show=False)
        st.pyplot(plt.gcf()); plt.clf()
    except Exception as e:
        st.caption(f"Không thể vẽ SHAP (môi trường hạn chế / mô hình không hỗ trợ): {e}")

# ====== sidebar ======
st.sidebar.header("Chế độ")
mode = st.sidebar.radio("Chọn chế độ demo", ["Tra nhanh (SĐT/CCCD)", "Chấm điểm chi tiết", "Giải thích & Đạo đức", "Quản trị (demo)"])
model, model_path = load_model()
st.sidebar.success(f"Model: {Path(model_path).name}" if model_path else "Chưa có model – đang dùng mô phỏng")

# ====== 1) Tra nhanh ======
if mode == "Tra nhanh (SĐT/CCCD)":
    st.title("Tra nhanh điểm tín dụng")
    st.write("Nhập **SĐT** hoặc **CCCD** (demo dùng DB **giả lập** đã băm/ẩn danh, không chứa dữ liệu thật).")
    identifier = st.text_input("SĐT / CCCD")
    if st.button("Kiểm tra"):
        if not identifier.strip():
            st.warning("Vui lòng nhập SĐT hoặc CCCD.")
        else:
            row = fake_lookup(identifier.strip())
            if row is None:
                st.error("Không tìm thấy hồ sơ (demo). Vui lòng thử 'Chấm điểm chi tiết' hoặc dùng ID khác.")
            else:
                X = ensure_feature_frame(row)
                if model is None:
                    # mô phỏng đơn giản
                    s = (X["Annual_Income"]/10000 + X["Monthly_Inhand_Salary"]/1000 - X["Num_of_Loan"]*0.5
                         - X["Delay_from_due_date"]/20 - X["Interest_Rate"]/30).values[0]
                    label = "Tốt" if s >= 1.5 else ("Trung bình" if s >= 0.9 else "Kém")
                    st.info(f"Kết quả (mô phỏng): **{label}**")
                else:
                    pred, proba, classes = score_with_model(model, X)
                    st.success(f"Kết quả: **{pred}**")
                    if proba is not None and len(classes):
                        st.subheader("Xác suất")
                        for c, p in zip(classes, proba):
                            st.write(f"- {c}: {p:.2%}")
                    with st.expander("Giải thích (SHAP)"):
                        try_shap_plot(model, X)

# ====== 2) Chấm điểm chi tiết ======
elif mode == "Chấm điểm chi tiết":
    st.title("Chấm điểm từ tiêu chí chi tiết")
    with st.form("credit_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input('Tuổi', 18, 100, 30)
            income = st.number_input('Thu nhập năm ($)', 0, 1_000_000, 12000, step=500)
            inhand = st.number_input('Lương thực nhận/tháng ($)', 0, 50_000, 1000, step=50)
            num_acc = st.number_input('Số tài khoản NH', 0, 20, 2)
            num_card = st.number_input('Số thẻ tín dụng', 0, 20, 1)
        with col2:
            rate = st.number_input('Lãi suất hiện tại (%)', 0, 100, 12)
            num_loan = st.number_input('Số khoản vay hiện tại', 0, 30, 1)
            occupation = st.selectbox('Nghề nghiệp', ['Software Engineer','Doctor','Teacher','Sales','Khác'])
            loan_type = st.selectbox('Loại hình khoản vay', ['Personal','Auto','Mortgage','BNPL','P2P'])
            delay = st.number_input('Số ngày trễ hạn TB', 0, 365, 0)

        submitted = st.form_submit_button("Chấm điểm")

    if submitted:
        X = ensure_feature_frame({
            'Age': age, 'Annual_Income': income, 'Monthly_Inhand_Salary': inhand,
            'Num_Bank_Accounts': num_acc, 'Num_Credit_Card': num_card,
            'Interest_Rate': rate, 'Num_of_Loan': num_loan,
            'Occupation': occupation, 'Type_of_Loan': loan_type, 'Delay_from_due_date': delay
        })
        if model is None:
            s = (income/10000 + inhand/1000 - num_loan*0.5 - delay/20 - rate/30)
            label = "Tốt" if s >= 1.5 else ("Trung bình" if s >= 0.9 else "Kém")
            st.info(f"Kết quả (mô phỏng): **{label}**")
        else:
            pred, proba, classes = score_with_model(model, X)
            st.success(f"Kết quả: **{pred}**")
            if proba is not None and len(classes):
                st.subheader("Xác suất")
                for c, p in zip(classes, proba):
                    st.write(f"- {c}: {p:.2%}")
            with st.expander("Giải thích (SHAP)"):
                try_shap_plot(model, X)

# ====== 3) Giải thích & Đạo đức ======
elif mode == "Giải thích & Đạo đức":
    st.title("Giải thích & Đạo đức")
    st.markdown("""
**Minh bạch**: mô hình có giải thích bằng SHAP; người dùng có quyền yêu cầu xem xét lại.  
**Quyền riêng tư**: demo dùng **DB giả lập** (đã băm SHA‑256 + muối), không lưu dữ liệu thật.  
**Công bằng**: áp dụng cân bằng lớp, kiểm tra fairness theo nhóm; điều chỉnh ngưỡng theo mục tiêu xã hội.  
**Bảo mật**: tách API scoring; giới hạn truy cập; ghi log ẩn danh.
    """)

# ====== 4) Quản trị (demo) ======
else:
    st.title("Quản trị (demo)")
    st.write("Tải lên mô hình `.pkl` tương thích (pipeline scikit‑learn).")
    f = st.file_uploader("Upload model (.pkl)", type=["pkl"])
    if f:
        Path("models").mkdir(exist_ok=True, parents=True)
        out = Path("models/uploaded.pkl")
        out.write_bytes(f.read())
        st.success(f"Đã lưu: {out}")
    st.write("Tải log dự đoán (CSV) – demo không bật ghi log thực để đảm bảo riêng tư.")
    st.caption("© Nhóm 10 – Fintech & Banking credit scoring demo")
