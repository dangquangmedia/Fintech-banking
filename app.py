# app.py — Credit Scoring (Streamlit) — VN form 2 cột + CIC 300–850
# - Giao diện “Nhập thông tin khách hàng” (2 cột), vẫn giữ các trường: Existing Loans, Delay days, Num Credit Cards,...
# - Kết quả hiển thị: PD và điểm tín dụng theo CIC (300–850) + phân loại (Kém/Khá/Tốt/Rất tốt/Xuất sắc)
# - Giữ nguyên luồng train nhanh CSV, upload model, SHAP (giới hạn mẫu), và mọi fix trước (no-lambda pickling, ép dtype/sparse cho SHAP…)
# - Có thể dùng Logistic Regression hoặc Random Forest (tối ưu hiệu năng)

import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

import shap
import matplotlib.pyplot as plt

# ============== CẤU HÌNH ==============
# Các trường demo khớp form + ưu tiên những trường bạn yêu cầu
FEATURE_SUBSET = [
    # Numeric
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",          # Số thẻ tín dụng
    "Interest_Rate",
    "Num_of_Loan",              # Số khoản vay hiện có
    "Delay_from_due_date",      # Số ngày trễ hạn
    # Categorical
    "Occupation",
    "Type_of_Loan",
]

MODEL_PATHS = [
    "models/random_forest.pkl",
    "models/logistic_regression.pkl",
    "models/uploaded.pkl",
    "models/xgboost.pkl",
]

# ============== HÀM HỖ TRỢ ==============
def to_str(X): return X.astype(str)

def is_fitted_estimator(est)->bool:
    try:
        check_is_fitted(est); return True
    except Exception:
        pass
    for a in ("classes_","n_features_in_"):
        if hasattr(est,a): return True
    return False

def load_any_model():
    for p in MODEL_PATHS:
        if os.path.exists(p):
            try:
                mdl = joblib.load(p)
                if isinstance(mdl, Pipeline) and "clf" in mdl.named_steps:
                    if not is_fitted_estimator(mdl.named_steps["clf"]): continue
                else:
                    if not is_fitted_estimator(mdl): continue
                return mdl, p
            except Exception as e:
                st.warning(f"Lỗi nạp model {p}: {e}")
    return None, None

def save_model(model,out_path:str):
    Path("models").mkdir(parents=True,exist_ok=True)
    joblib.dump(model,out_path); return out_path

def split_features_target(df,target_col):
    return df.drop(columns=[target_col]), df[target_col]

def infer_cols(X):
    num=[c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat=[c for c in X.columns if c not in num]
    return num,cat

def build_pipeline(model_type, X_sample, rf_params:dict|None=None)->Pipeline:
    rf_params = rf_params or {}
    num_cols, cat_cols = infer_cols(X_sample)
    num_tf = Pipeline([("imputer",SimpleImputer(strategy="median"))])
    cat_tf = Pipeline([
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("to_str",FunctionTransformer(lambda x: x.astype(str))),  # an toàn pickling vì FunctionTransformer сериал hoá được
        ("ohe",OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([("num",num_tf,num_cols),("cat",cat_tf,cat_cols)])
    if model_type=="Logistic Regression":
        clf = LogisticRegression(max_iter=200)
    else:
        clf = RandomForestClassifier(
            n_estimators=rf_params.get("n_estimators",150),
            max_depth=rf_params.get("max_depth",None),
            max_features=rf_params.get("max_features","sqrt"),
            min_samples_leaf=rf_params.get("min_samples_leaf",1),
            random_state=42, n_jobs=-1
        )
    return Pipeline([("pre",pre),("clf",clf)])

def get_feature_names_from_pre(pre:ColumnTransformer)->list[str]:
    names=[]
    for name,trans,cols in pre.transformers_:
        if name=='remainder' and trans=='drop': continue
        if hasattr(trans,'named_steps'):
            last=list(trans.named_steps.values())[-1]
            if hasattr(last,'get_feature_names_out'):
                base=cols if isinstance(cols,list) else [cols]
                try: names+=last.get_feature_names_out(base).tolist()
                except Exception: names+= base
            else: names+= cols if isinstance(cols,list) else [cols]
        else: names+= cols if isinstance(cols,list) else [cols]
    return names

def ensure_subset_and_types(df):
    X=df.copy()
    for c in FEATURE_SUBSET:
        if c not in X.columns: X[c]=pd.NA
    X=X[FEATURE_SUBSET]
    for c in FEATURE_SUBSET:
        conv=pd.to_numeric(X[c],errors="coerce")
        X[c]=conv if conv.notna().any() else X[c].astype(str)
    return X

def densify_float64(X):
    if hasattr(X,"toarray"): X=X.toarray()
    return np.asarray(X,dtype=np.float64)

def try_predict(model,X_df):
    try:
        X_df=ensure_subset_and_types(X_df)
        if isinstance(model,Pipeline):
            pre,clf = model.named_steps["pre"], model.named_steps["clf"]
            X_tr=pre.transform(X_df)
            pred=clf.predict(X_tr)[0]
            proba=classes=None
            if hasattr(clf,"predict_proba"):
                pp=clf.predict_proba(X_tr)[0]; classes=getattr(clf,"classes_",[str(i) for i in range(len(pp))]); proba=pp
            return pred,proba,classes,X_tr,get_feature_names_from_pre(pre)
        else:
            pred=model.predict(X_df)[0]
            proba=classes=None
            if hasattr(model,"predict_proba"):
                pp=model.predict_proba(X_df)[0]; classes=getattr(model,"classes_",[str(i) for i in range(len(pp))]); proba=pp
            return pred,proba,classes,X_df.to_numpy(),list(X_df.columns)
    except NotFittedError:
        return "Lỗi: Mô hình chưa được huấn luyện.",None,None,None,None
    except Exception as e:
        return f"Lỗi dự đoán: {e}",None,None,None,None

def plot_confusion_matrix_cm(y_true,y_pred,labels=None):
    cm=confusion_matrix(y_true,y_pred,labels=labels)
    fig,ax=plt.subplots(); im=ax.imshow(cm,interpolation='nearest'); ax.figure.colorbar(im,ax=ax)
    ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ticks=np.arange(len(labels)) if labels is not None else np.arange(cm.shape[0])
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(labels if labels is not None else ticks,rotation=45,ha="right")
    ax.set_yticklabels(labels if labels is not None else ticks)
    thresh=cm.max()/2.0 if cm.size else .5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,format(cm[i,j],"d"),ha="center",va="center",color="white" if cm[i,j]>thresh else "black")
    fig.tight_layout(); st.pyplot(fig)

def try_build_tree_explainer(pipe:Pipeline,X_bg):
    try:
        pre,clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
        X_bg_tr = densify_float64(pre.transform(ensure_subset_and_types(X_bg)))
        if isinstance(clf,RandomForestClassifier):
            return shap.TreeExplainer(clf,feature_names=get_feature_names_from_pre(pre)), get_feature_names_from_pre(pre), X_bg_tr
    except Exception as e:
        st.info(f"Không tạo được TreeExplainer: {e}")
    return None,None,None

def global_shap_bar(explainer,X_tr,feature_names,topk=15):
    X_tr=densify_float64(X_tr)
    shap_vals=explainer.shap_values(X_tr)
    if isinstance(shap_vals,list):
        abs_means=np.stack([np.mean(np.abs(sv),axis=0) for sv in shap_vals],axis=0); imp=abs_means.max(axis=0)
    else:
        imp=np.mean(np.abs(shap_vals),axis=0)
    idx=np.argsort(imp)[::-1][:topk]
    fig,ax=plt.subplots(); ax.barh([feature_names[i] for i in idx][::-1],imp[idx][::-1])
    ax.set_xlabel("|SHAP value| (mean)"); ax.set_title("Global feature importance (SHAP)")
    fig.tight_layout(); st.pyplot(fig)

def local_shap_table(explainer,x1_tr,feature_names,topk=10):
    x1=densify_float64(x1_tr).reshape(1,-1)
    shap_vals=explainer.shap_values(x1)
    if isinstance(shap_vals,list):
        totals=[np.sum(np.abs(sv)) for sv in shap_vals]; sv=shap_vals[int(np.argmax(totals))][0]
    else: sv=shap_vals[0]
    order=np.argsort(np.abs(sv))[::-1][:topk]
    st.subheader("Đóng góp theo đặc trưng (SHAP) – bản ghi hiện tại")
    st.dataframe(pd.DataFrame({"feature":[feature_names[i] for i in order],
                               "shap_value":sv[order],
                               "contribution":np.where(sv[order]>=0,"↑ tăng rủi ro","↓ giảm rủi ro")}),
                 use_container_width=True)

def fake_lookup(identifier:str)->pd.DataFrame:
    return pd.DataFrame([{
        "Age":30,"Annual_Income":30000,"Monthly_Inhand_Salary":2000,"Num_Bank_Accounts":2,
        "Num_Credit_Card":1,"Interest_Rate":12,"Num_of_Loan":1,"Occupation":"Employee",
        "Type_of_Loan":"Personal","Delay_from_due_date":0
    }])

def ensure_frame(x): return pd.DataFrame([x]) if isinstance(x,dict) else x

# ----- Điểm CIC 300–850 -----
def pd_to_cic_score(pd_hat: float) -> int:
    """Map PD (0..1) -> điểm CIC 300..850 (cao = rủi ro thấp)."""
    pd_hat = float(np.clip(pd_hat, 0.0, 1.0))
    return int(round(300 + (1.0 - pd_hat) * 550))  # span 550

def classify_cic(score: int) -> str:
    if score >= 800: return "Xuất sắc (800–850)"
    if score >= 740: return "Rất tốt (740–799)"
    if score >= 670: return "Tốt (670–739)"
    if score >= 580: return "Khá (580–669)"
    return "Kém (<580)"

def estimate_pd_from_proba(classes, proba) -> float:
    """Ưu tiên xác suất lớp rủi ro cao: Poor/Bad. Nếu có 'Standard' xem như rủi ro trung bình 0.5."""
    if classes is None or proba is None: return 0.5
    label_to_w = {"poor":1.0, "bad":1.0, "standard":0.5, "good":0.0}
    pd_est = 0.0
    for c,p in zip(classes, proba):
        w = label_to_w.get(str(c).lower(), 0.0)
        pd_est += w * float(p)
    # nếu các lớp chỉ có 2 (good/poor) thì giá trị trên là hợp lý;
    # nếu đủ 3 lớp (good/standard/poor) thì coi standard ~0.5 * p
    return float(np.clip(pd_est, 0.0, 1.0))

# ============== UI ==============
st.set_page_config(page_title="Đánh giá điểm tín dụng", layout="wide")

st.sidebar.header("Chế độ")
mode=st.sidebar.radio("Chọn chế độ",["Tra nhanh (SĐT/CCCD/ID)","Chấm điểm chi tiết","Giải thích & Đạo đức","Quản trị (train & upload)"])
model,model_path=load_any_model()
if "explainer" not in st.session_state: st.session_state.explainer=None
if "feature_names" not in st.session_state: st.session_state.feature_names=None
st.sidebar.success(f"Model: {os.path.basename(model_path)}" if model_path else "Chưa có model – mô phỏng")

# ============== 1) TRA NHANH ==============
if mode=="Tra nhanh (SĐT/CCCD/ID)":
    st.title("Tra nhanh điểm tín dụng")
    st.write("- Dataset Kaggle thường **không có SĐT** → hãy nhập **ID/Customer_ID**. Nếu không upload CSV, hệ thống dùng demo giả lập.")
    identifier=st.text_input("Nhập SĐT/CCCD/ID:")
    st.markdown("**(Tuỳ chọn)** Tải dataset CSV để tra cứu bản ghi thật:")
    search_csv=st.file_uploader("Upload CSV (có cột ID/SĐT/CCCD/Customer_ID)",type=["csv"],key="search_csv")
    id_column=df_search=None
    if search_csv is not None:
        df_search=pd.read_csv(search_csv)
        st.dataframe(df_search.head())
        guess=[c for c in df_search.columns if c.lower() in ("id","customer_id","phone","cccd","sdt")]
        id_column=st.selectbox("Chọn cột định danh để tra",options=df_search.columns,
                               index=0 if not guess else df_search.columns.get_loc(guess[0]))
    if st.button("Kiểm tra"):
        if not identifier.strip():
            st.warning("Vui lòng nhập ID/Customer_ID.")
        else:
            if df_search is not None and id_column:
                row=df_search[df_search[id_column].astype(str)==str(identifier)].drop(columns=[id_column],errors="ignore")
                if row.empty:
                    st.error("Không tìm thấy hồ sơ trong CSV."); st.stop()
                X=row
            else:
                X=ensure_frame(fake_lookup(identifier))
            if model is None:
                X=ensure_subset_and_types(X)
                s=(X.get("Annual_Income",pd.Series([30000])).iloc[0]/10000 +
                   X.get("Monthly_Inhand_Salary",pd.Series([2000])).iloc[0]/1000 -
                   X.get("Num_of_Loan",pd.Series([1])).iloc[0]*0.5 -
                   X.get("Delay_from_due_date",pd.Series([0])).iloc[0]/20 -
                   X.get("Interest_Rate",pd.Series([12])).iloc[0]/30)
                st.info(f"Kết quả (mô phỏng): **{'Tốt' if s>=1.5 else ('Trung bình' if s>=0.9 else 'Kém')}**")
            else:
                pred,proba,classes,X_tr,feat_names=try_predict(model,X)
                if isinstance(pred,str) and pred.startswith("Lỗi"): st.error(pred)
                else:
                    pd_hat = estimate_pd_from_proba(classes, proba)
                    score = pd_to_cic_score(pd_hat)
                    st.success("Kết quả")
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Dự đoán lớp", str(pred))
                    c2.metric("PD (xác suất vỡ nợ)", f"{pd_hat:.3f}")
                    c3.metric("Điểm CIC (300–850)", f"{score} · {classify_cic(score)}")
                    if st.session_state.explainer is not None and X_tr is not None:
                        with st.expander("Giải thích (SHAP) – bản ghi hiện tại"):
                            local_shap_table(st.session_state.explainer,X_tr,st.session_state.feature_names,topk=10)
                    else: st.caption("Gợi ý: Train Random Forest ở tab Quản trị để bật SHAP.")

# ============== 2) CHẤM ĐIỂM CHI TIẾT — GIAO DIỆN 2 CỘT ==============
elif mode=="Chấm điểm chi tiết":
    st.title("Nhập thông tin khách hàng")  # giữ tiêu đề theo yêu cầu
    st.caption("Điền thông tin theo tiêu chí. Kết quả trả PD và điểm CIC 300–850.")

    with st.form("detail_form"):
        left, right = st.columns(2, gap="large")

        # Nhóm numeric (bên trái)
        age         = left.number_input("Tuổi", 18, 100, 30, help="Độ tuổi khách hàng.")
        income      = left.number_input("Annual Income", 1000, 10_000_000, 30000, help="Thu nhập năm.")
        inhand      = left.number_input("Monthly Inhand Salary", 100, 100_000, 2000, help="Thu nhập thực lĩnh/tháng.")
        num_acc     = left.number_input("Số tài khoản ngân hàng", 0, 50, 2)
        num_card    = left.number_input("Số thẻ tín dụng", 0, 50, 1)  # GIỮ
        rate        = left.number_input("Lãi suất (%)", 0, 100, 12)
        num_loan    = left.number_input("Số khoản vay hiện có", 0, 50, 1)  # GIỮ
        delay       = left.number_input("Số ngày trễ hạn", 0, 3650, 0)      # GIỮ

        # Nhóm categorical (bên phải)
        occupation  = right.text_input("Nghề nghiệp", "Employee")
        loan_type   = right.text_input("Loại khoản vay", "Personal")

        submitted = st.form_submit_button("Dự đoán")

    X = ensure_frame({
        "Age": age,
        "Annual_Income": income,
        "Monthly_Inhand_Salary": inhand,
        "Num_Bank_Accounts": num_acc,
        "Num_Credit_Card": num_card,
        "Interest_Rate": rate,
        "Num_of_Loan": num_loan,
        "Delay_from_due_date": delay,
        "Occupation": occupation,
        "Type_of_Loan": loan_type,
    })

    if submitted:
        if model is None:
            X2=ensure_subset_and_types(X)
            # mô phỏng đơn giản (có trọng số delay + num_card + num_loan)
            s = (income/10000 + inhand/1000 - num_loan*0.6 - delay/18 - rate/30 - num_card*0.1)
            pd_hat = float(1.0/(1.0+np.exp(s)))  # logistic giả lập
            score = pd_to_cic_score(pd_hat)
            c1,c2,c3 = st.columns(3)
            c1.metric("Dự đoán (mô phỏng)", "Không có model")
            c2.metric("PD (xác suất vỡ nợ)", f"{pd_hat:.3f}")
            c3.metric("Điểm CIC (300–850)", f"{score} · {classify_cic(score)}")
        else:
            pred,proba,classes,X_tr,feat_names=try_predict(model,X)
            if isinstance(pred,str) and pred.startswith("Lỗi"): st.error(pred)
            else:
                pd_hat = estimate_pd_from_proba(classes, proba)
                score = pd_to_cic_score(pd_hat)
                st.success("Kết quả")
                c1,c2,c3 = st.columns(3)
                c1.metric("Dự đoán lớp", str(pred))
                c2.metric("PD (xác suất vỡ nợ)", f"{pd_hat:.3f}")
                c3.metric("Điểm CIC (300–850)", f"{score} · {classify_cic(score)}")
                if st.session_state.explainer is not None and X_tr is not None:
                    with st.expander("Giải thích (SHAP) – bản ghi hiện tại"):
                        local_shap_table(st.session_state.explainer,X_tr,st.session_state.feature_names,topk=10)
                else:
                    st.caption("Gợi ý: Train Random Forest ở tab Quản trị để bật SHAP.")

# ============== 3) GIẢI THÍCH & ĐẠO ĐỨC ==============
elif mode=="Giải thích & Đạo đức":
    st.title("Giải thích & Đạo đức")
    st.markdown("""
**Minh bạch**: giải thích cục bộ (SHAP) cho từng dự đoán; cơ chế soát xét/kháng nghị.  
**Quyền riêng tư**: tách PII, ẩn danh, hạn chế truy cập.  
**Công bằng**: theo dõi bias theo nhóm; cân bằng dữ liệu; điều chỉnh ngưỡng.  
**Bảo mật**: tách dịch vụ scoring; kiểm soát truy cập; log ẩn danh.
    """)

# ============== 4) QUẢN TRỊ (TRAIN & UPLOAD) ==============
else:
    st.title("Quản trị (train & upload)")
    st.subheader("A) Upload model có sẵn (.pkl)")
    f=st.file_uploader("Upload model (.pkl)",type=["pkl"])
    if f:
        out=Path("models/uploaded.pkl"); out.parent.mkdir(parents=True,exist_ok=True); out.write_bytes(f.read())
        st.success(f"Đã lưu model: {out}")
        try:
            new_model,new_path=load_any_model()
            if new_model: st.info(f"Model đang dùng: {new_path}")
        except Exception as e: st.error(f"Lỗi nạp model upload: {e}")

    st.markdown("---")
    st.subheader("B) Train nhanh từ CSV (Kaggle/nguồn khác)")
    up=st.file_uploader("Upload CSV Train",type=["csv"],key="train_csv")
    if up is not None:
        df_train=pd.read_csv(up)
        st.write("Xem trước dữ liệu:"); st.dataframe(df_train.head())

        # chọn target & ID
        guess_t=next((c for c in ["Credit_Score","risk_flag","label","target"] if c in df_train.columns),None)
        target_col=st.selectbox("Chọn cột nhãn (target)",options=df_train.columns,
                                index=df_train.columns.get_loc(guess_t) if guess_t else 0)
        guess_id=next((c for c in ["ID","Customer_ID","customer_id"] if c in df_train.columns),None)
        id_col=st.selectbox("Chọn cột ID (tuỳ chọn, sẽ bỏ khi train)",
                            options=["(Không dùng)"]+list(df_train.columns),
                            index=(["(Không dùng)"]+list(df_train.columns)).index(guess_id) if guess_id else 0)

        df_work=df_train.drop(columns=[id_col]) if id_col!="(Không dùng)" else df_train.copy()

        missing=[c for c in FEATURE_SUBSET if c not in df_work.columns]
        if missing:
            st.error("Dataset thiếu các cột cho demo (khớp form): "+", ".join(missing)); st.stop()

        df_model=df_work[FEATURE_SUBSET+[target_col]].copy()
        for c in FEATURE_SUBSET:
            conv=pd.to_numeric(df_model[c],errors="coerce")
            df_model[c]=conv if conv.notna().any() else df_model[c].astype(str)

        # Tuỳ chọn giảm cỡ dữ liệu (để nhanh)
        reduce_frac = st.selectbox("Giảm kích thước tập train (để nhanh)", ["1.0 (không giảm)","0.5","0.25","0.1"], index=0)
        reduce_frac = float(reduce_frac.split()[0])
        X_all,y_all=split_features_target(df_model,target_col)
        if reduce_frac<1.0:
            tmp=X_all.copy(); tmp[target_col]=y_all.values
            X_all = tmp.groupby(target_col, group_keys=False).apply(lambda d: d.sample(max(1,int(len(d)*reduce_frac)), random_state=42)).drop(columns=[target_col])
            y_all = tmp.groupby(target_col, group_keys=False).apply(lambda d: d.sample(max(1,int(len(d)*reduce_frac)), random_state=42))[target_col]

        test_size=st.slider("Tỉ lệ test",0.1,0.4,0.2,step=0.05)
        model_type=st.selectbox("Chọn thuật toán",["Logistic Regression","Random Forest"])

        rf_params={}
        enable_shap=False
        if model_type=="Random Forest":
            rf_params["n_estimators"]=st.slider("Số cây Random Forest",50,400,150,step=50)
            md = st.selectbox("max_depth",["None","5","10","20"],index=0)
            rf_params["max_depth"]=None if md=="None" else int(md)
            mf = st.selectbox("max_features",["sqrt","log2","0.5"],index=0)
            rf_params["max_features"]=0.5 if mf=="0.5" else mf
            rf_params["min_samples_leaf"]=st.selectbox("min_samples_leaf",[1,2,4,8],index=1)
            enable_shap = st.checkbox("Bật SHAP (giới hạn mẫu, có thể chậm)", value=False)

        if st.button("Huấn luyện"):
            try:
                vc=y_all.value_counts(); strat=None if (vc<2).any() else y_all
                X_tr,X_te,y_tr,y_te=train_test_split(X_all,y_all,test_size=test_size,random_state=42,stratify=strat)
                pipe=build_pipeline(model_type,X_tr,rf_params=rf_params)

                with st.status("Đang huấn luyện...",expanded=True) as status:
                    pipe.fit(X_tr,y_tr); st.write("✅ Đã fit xong model.")
                    y_pred=pipe.predict(X_te)
                    acc=accuracy_score(y_te,y_pred); f1=f1_score(y_te,y_pred,average="weighted")
                    st.write("**Kết quả trên tập test:**"); st.write(f"- Accuracy: {acc:.4f}"); st.write(f"- F1 (weighted): {f1:.4f}")
                    st.code(classification_report(y_te,y_pred),language="text")
                    with st.expander("Confusion Matrix"):
                        labels=[str(c) for c in sorted(pd.unique(pd.concat([y_tr,y_te])))]
                        plot_confusion_matrix_cm(y_te,y_pred,labels=labels)

                    st.session_state.explainer=None; st.session_state.feature_names=None
                    if model_type=="Random Forest" and enable_shap:
                        st.write("🔎 Đang tạo SHAP (giới hạn mẫu)…")
                        bg_n=min(150,len(X_tr)); te_n=min(200,len(X_te))
                        X_bg=X_tr.sample(bg_n,random_state=42) if len(X_tr)>bg_n else X_tr
                        X_te_small=X_te.sample(te_n,random_state=42) if len(X_te)>te_n else X_te
                        explainer,feat_names,_=try_build_tree_explainer(pipe,X_bg)
                        if explainer is not None:
                            st.session_state.explainer=explainer; st.session_state.feature_names=feat_names
                            with st.expander(f"Global Feature Importance (SHAP) — {te_n} mẫu"):
                                pre=pipe.named_steps["pre"]
                                X_te_tr=densify_float64(pre.transform(ensure_subset_and_types(X_te_small)))
                                global_shap_bar(explainer,X_te_tr,feat_names,topk=15)
                        else: st.info("Không tạo được SHAP Explainer.")
                    status.update(label="Huấn luyện xong!",state="complete")

                out_name="models/logistic_regression.pkl" if model_type=="Logistic Regression" else "models/random_forest.pkl"
                save_model(pipe,out_name); st.success(f"💾 Đã lưu model: {out_name}")

            except Exception as e:
                st.error(f"Lỗi train: {e}")
