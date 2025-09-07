import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import joblib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Đọc dữ liệu
df = pd.read_csv('financial_data.csv')

# --- Bước 1: Feature Engineering ---
# Tạo các đặc trưng tỷ lệ chi tiêu trên thu nhập
df['rent_ratio'] = df['rent_spending'] / df['total_income']
df['food_ratio'] = df['food_spending'] / df['total_income']
df['transport_ratio'] = df['transport_spending'] / df['total_income']
df['shopping_ratio'] = df['shopping_spending'] / df['total_income']
df['entertainment_ratio'] = df['entertainment_spending'] / df['total_income']

# --- Bước 2: Chuẩn bị dữ liệu cho clustering ---
# ĐÃ LOẠI BỎ 'age' KHỎI DANH SÁCH FEATURE
features_for_clustering = [
    'total_income', 'rent_ratio', 'food_ratio', 
    'transport_ratio', 'shopping_ratio', 'entertainment_ratio', 'city'
]

# Định nghĩa các cột số và cột phân loại
numerical_features = features_for_clustering[:-1]
categorical_features = ['city']

# Tạo pipeline để xử lý dữ liệu: Scale dữ liệu số và One-Hot Encode dữ liệu phân loại
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough' # Giữ lại các cột không được biến đổi nếu có
)

X = df[features_for_clustering]
X_processed = preprocessor.fit_transform(X)

print("Đã xử lý và chuẩn hóa dữ liệu.")

# --- Bước 3: Tìm số K tối ưu bằng Elbow Method ---
inertia_values = []
k_range = range(2, 11)
print("Đang tìm số K tối ưu bằng Elbow Method...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_processed)
    inertia_values.append(kmeans.inertia_)

# Vẽ biểu đồ Elbow
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia_values, marker='o', linestyle='--')
plt.title('Elbow Method để tìm K tối ưu')
plt.xlabel('Số lượng cụm (K)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.savefig('elbow_plot.png')
print("Đã lưu biểu đồ Elbow Method vào file 'elbow_plot.png'.")

# DỰA TRÊN BIỂU ĐỒ, CHỌN K TỐI ƯU (ví dụ: K=5)
OPTIMAL_K = 5
print(f"Chọn K tối ưu là: {OPTIMAL_K}")

# --- Bước 4: Huấn luyện model cuối cùng với K tối ưu ---
kmeans_final = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
df['cluster'] = kmeans_final.fit_predict(X_processed)
print(f"Đã huấn luyện model cuối cùng với K={OPTIMAL_K}.")

# --- Bước 5: Tạo và lưu "Persona" cho mỗi cụm ---
# Đảm bảo chỉ lấy các cột số đã định nghĩa để tính trung bình
persona_features = numerical_features + ['rent_ratio', 'food_ratio', 'transport_ratio', 'shopping_ratio', 'entertainment_ratio']
# Loại bỏ các cột trùng lặp
persona_features = list(dict.fromkeys(persona_features)) 

cluster_personas = df.groupby('cluster')[persona_features].mean()

# Thêm lại các cột tỷ lệ vào persona để logic gợi ý hoạt động đúng
for ratio_col in ['rent_ratio', 'food_ratio', 'transport_ratio', 'shopping_ratio', 'entertainment_ratio']:
    if ratio_col not in cluster_personas.columns:
        cluster_personas[ratio_col] = df.groupby('cluster')[ratio_col].mean()

print("\n'Persona' cho mỗi cụm:")
print(cluster_personas)

# --- Bước 6: Xuất (Export) các đối tượng cần thiết ---
joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(kmeans_final, 'kmeans_model.joblib')
cluster_personas.to_csv('cluster_personas.csv')

print("\nĐã export thành công 3 file:")
print("1. preprocessor.joblib")
print("2. kmeans_model.joblib")
print("3. cluster_personas.csv")

