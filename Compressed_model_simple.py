import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from helpers import load_dataset, save_model

# 1. 读取数据（NumPy格式）
X, y = load_dataset("data_students.txt")
X = np.asarray(X)
y = np.asarray(y)
print("Dataset shape:", X.shape, y.shape)

# 2. 最优模型参数（我的结果）
model = RandomForestRegressor(
    n_estimators=15,
    max_depth=20,
    min_samples_leaf=3,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

# 3. 训练模型
model.fit(X, y)

# 4. 训练集评估（可选但推荐）
pred = model.predict(X)
r2 = r2_score(y, pred)
mae = mean_absolute_error(y, pred)
rmse = np.sqrt(mean_squared_error(y, pred))
print("\nTraining finished.")
print("R²   :", r2)
print("MAE  :", mae)
print("RMSE :", rmse)

# 5. 保存模型
save_model(model, "JY_369.pkl")
print("\nModel saved as JY_369.pkl")