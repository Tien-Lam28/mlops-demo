'''
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from evidently import Dataset, DataDefinition, Report
from evidently.presets import (
    DataDriftPreset,
    ClassificationPreset
)

from evidently.core.datasets import BinaryClassification


# =========================================================
# 1. Generate reference dataset
# =========================================================

X, y = make_classification(
    n_samples=2000,
    n_features=5,
    n_informative=3,
    random_state=42
)

reference_data = pd.DataFrame(
    X,
    columns=[
        "age",
        "income",
        "score",
        "debt",
        "balance"
    ]
)

reference_data["target"] = y


# =========================================================
# 2. Train RandomForest model
# =========================================================

X_train = reference_data.drop(columns=["target"])
y_train = reference_data["target"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# =========================================================
# 3. Create drifted current dataset
# =========================================================

current_data = reference_data.copy()

# create obvious drift
current_data["income"] += 5
current_data["score"] *= 1.8
current_data["balance"] += (
    current_data["balance"] * 0.5
)


# =========================================================
# 4. Generate predictions
# =========================================================

reference_X = reference_data.drop(columns=["target"])
current_X = current_data.drop(columns=["target"])

reference_data["prediction"] = model.predict(reference_X)
current_data["prediction"] = model.predict(current_X)


# =========================================================
# 5. Evidently dataset configuration
# =========================================================

data_definition = DataDefinition(
    classification=[
        BinaryClassification(
            target="target",
            prediction_labels="prediction"
        )
    ]
)

reference_dataset = Dataset.from_pandas(
    reference_data,
    data_definition=data_definition
)

current_dataset = Dataset.from_pandas(
    current_data,
    data_definition=data_definition
)


# =========================================================
# 6. Build Evidently report
# =========================================================

def generate_report(reference_data, current_data):
    #tao va luu report
    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset()
    ])
    my_eval = report.run(reference_data,current_data)

    return my_eval

def save_report(report, save_path):
    #os.makedirs(save_path, exist_ok= True)
    report.save_html(str(save_path))

    print(f"✅ Report đã được lưu an toàn tại: {save_path}")

def analyze_drift(report, drift_threshold=0.3):
    result = report.dict()
    #print(json.dumps(result["metrics"][0], indent=2, default=str))
    drift_share = result["metrics"][0]["value"]["share"]

    if drift_share >= drift_threshold:
        print(f"\n🚨 ALERT! Tỷ lệ drift vượt ngưỡng ({drift_share:.2%} >= {drift_threshold:.0%})")
        
        # GỌI HÀM RETRAIN
        #proactive_retrain()
        #ham nay khong the dong nhat voi dev1 nen khong them vao

    else:
        print(f"\n✅ Model đang hoạt động ổn định ({drift_share:.2%}).")     



report = generate_report(reference_dataset, current_dataset)


# =========================================================
# 7. Save report
# =========================================================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent 
report_file = BASE_DIR / "reports" / "drift_report.html"

save_report(report, report_file)

#import json
analyze_drift(report)
'''

#phan tren la de demo co chay duoc hay khong
#phan duoi la de lien ket voi dev1 va dev2

''''''
import os
import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import (
    DataDriftPreset,
    ClassificationPreset
)

from sklearn.ensemble import RandomForestClassifier
from evidently.core.datasets import BinaryClassification
from pathlib import Path
import joblib

#declaring paths
BASE_DIR = Path(__file__).resolve().parent.parent
train_path = BASE_DIR / "data" / "train_data.csv"
logs_path = BASE_DIR / "logs" / "inferences_logs.csv"
model_path = BASE_DIR / "models" / "model.pkl"
report_path = BASE_DIR / "reports" / "drift_report.html"


#defining methods
def load_model(model_path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"❌ Không tìm thấy model: {model_path}"
        )
    model = joblib.load(model_path)
    print(f"✅ THÀNH CÔNG: Đã tải model từ {model_path}")
    return model

def load_data(reference_path, current_path):
    if not reference_path.exists():
        raise FileNotFoundError(
            f"❌ Không tìm thấy reference data: {reference_path}"
        )
    
    if not current_path.exists():
        raise FileNotFoundError(
            f"❌ Không tìm thấy inference logs: {current_path}"
        )
    
    reference_data = pd.read_csv(reference_path)
    current_data = pd.read_csv(current_path)
    print(f"Đã tải dữ liệu thành công từ {reference_path} và {current_path}")

    return reference_data, current_data

def preprocess(reference_data, current_data, model):
    feature_cols = [
        "thu_nhap",
        "so_tien_vay",
        "lich_su_no_xau"
    ]
    # Thêm .copy() và .dropna() để code an toàn tuyệt đối
    ref_df = reference_data[feature_cols].dropna().copy()
    
    # Với current_data (log), ta lấy thêm cột 'kết quả dự đoán' có sẵn
    # (Đảm bảo tên cột này khớp với tên Dev 2 đã ghi vào CSV)
    cur_cols = feature_cols + ["kết quả dự đoán"] 
    cur_df = current_data[cur_cols].dropna().copy()
    
    # Xử lý Prediction cho tập Reference (vì data train chưa có dự đoán)
    ref_X = ref_df.drop(columns=["lich_su_no_xau"])
    ref_df["kết quả dự đoán"] = model.predict(ref_X)

    #chuan bi data cho Evidently hieu duoc
    data_definition = DataDefinition(
        classification=[
            BinaryClassification(
                target="lich_su_no_xau",
                prediction_labels="kết quả dự đoán"
            )
        ]
    )

    reference_evidently = Dataset.from_pandas(
        cur_df,
        data_definition=data_definition
    )

    current_evidently = Dataset.from_pandas(
        cur_df,
        data_definition=data_definition
    )

    return reference_evidently, current_evidently

def generate_report(reference_data, current_data):
    #tao va luu report
    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset()
    ])
    my_eval = report.run(reference_data,current_data)

    return my_eval

def save_report(report, save_path):
    #os.makedirs(save_path, exist_ok= True)
    report.save_html(str(save_path))

    print(f"✅ Report đã được lưu an toàn tại: {save_path}")

def analyze_drift(report, drift_threshold=0.3):
    result = report.as_dict()
    drift_share = result["metrics"][0]["result"]["share_of_drifted_columns"]

    if drift_share >= drift_threshold:
        print(f"\n🚨 ALERT! Tỷ lệ drift vượt ngưỡng ({drift_share:.2%} >= {drift_threshold:.0%})")
        
        # GỌI HÀM RETRAIN
        #proactive_retrain()
        #ham nay khong the dong nhat voi dev1 nen khong them vao

    else:
        print(f"\n✅ Model đang hoạt động ổn định ({drift_share:.2%}).")     


#load 2 dataset tu dev1 va dev2
reference_data, current_data = load_data(logs_path, train_path)
#load model tu file train.py
model = load_model(model_path)

#chuan bi dinh dang cua data de tao report
reference_evidently, current_evidently = preprocess(reference_data, current_data, model)

#tao va luu report 
report = generate_report(reference_evidently, current_evidently)
save_report(report, report_path)

analyze_drift(report)


