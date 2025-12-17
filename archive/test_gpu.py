import xgboost as xgb
import numpy as np

print(f"XGBoost Version: {xgb.__version__}")
try:
    data = np.random.rand(100, 10)
    label = np.random.randint(0, 2, 100)
    clf = xgb.XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor')
    clf.fit(data, label)
    print("SUCCESS: XGBoost GPU training worked!")
except Exception as e:
    print(f"FAILURE: XGBoost GPU training failed. Error: {e}")
