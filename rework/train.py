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

# --- Bước 1: Feature Engineering (Cải tiến) ---
print("Đang thực hiện Feature Engineering nâng cao...")

# 1.1. Tạo các đặc trưng tỷ lệ chi tiêu cơ bản
df['rent_ratio'] = df['rent_spending'] / df['total_income']
df['food_ratio'] = df['food_spending'] / df['total_income']
df['transport_ratio'] = df['transport_spending'] / df['total_income']
df['shopping_ratio'] = df['shopping_spending'] / df['total_income']
df['entertainment_ratio'] = df['entertainment_spending'] / df['total_income']

# 1.2. Tạo các đặc trưng tổng hợp mang nhiều ý nghĩa hơn
df['savings_ratio'] = df['current_savings'] / df['total_income']
df['essential_spending_ratio'] = df['rent_ratio'] + df['food_ratio'] + df['transport_ratio']
df['discretionary_spending_ratio'] = df['shopping_ratio'] + df['entertainment_ratio']

# Xử lý các trường hợp chia cho 0 hoặc giá trị vô hạn
df.replace([float('inf'), -float('inf')], 0, inplace=True)
df.fillna(0, inplace=True)


# --- Bước 2: Chuẩn bị dữ liệu cho clustering ---
# SỬ DỤNG BỘ FEATURE MỚI, MẠNH MẼ HƠN
features_for_clustering = [
    'total_income',
    'savings_ratio',                  # Đặc trưng mới
    'essential_spending_ratio',       # Đặc trưng mới
    'discretionary_spending_ratio',   # Đặc trưng mới
    'city'
]

# Định nghĩa các cột số và cột phân loại
numerical_features = features_for_clustering[:-1]
categorical_features = ['city']

# Tạo pipeline để xử lý dữ liệu
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'
)

X = df[features_for_clustering]
X_processed = preprocessor.fit_transform(X)
print("Đã xử lý và chuẩn hóa dữ liệu với bộ feature mới.")


# --- Bước 3: Tìm số K tối ưu bằng Elbow Method (không thay đổi) ---
inertia_values = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_processed)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia_values, marker='o', linestyle='--')
plt.title('Elbow Method để tìm K tối ưu (với Feature mới)')
plt.xlabel('Số lượng cụm (K)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.savefig('elbow_plot_new.png')
print("Đã lưu biểu đồ Elbow Method mới vào file 'elbow_plot_new.png'.")

OPTIMAL_K = 5 
print(f"Giữ nguyên K tối ưu là: {OPTIMAL_K}")

# --- Bước 4: Huấn luyện model cuối cùng ---
kmeans_final = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
df['cluster'] = kmeans_final.fit_predict(X_processed)
print(f"Đã huấn luyện model cuối cùng với K={OPTIMAL_K}.")

# --- Bước 5: Tạo và lưu "Persona" cho mỗi cụm ---
# Lấy tất cả các cột số và tỷ lệ để phân tích persona
persona_features = numerical_features + [
    'rent_ratio', 'food_ratio', 'transport_ratio', 
    'shopping_ratio', 'entertainment_ratio'
]
persona_features = list(dict.fromkeys(persona_features)) 

cluster_personas = df.groupby('cluster')[persona_features].mean()

print("\n'Persona' mới cho mỗi cụm (sẽ rõ ràng hơn):")
print(cluster_personas)


# --- Bước 6: Xuất (Export) các đối tượng cần thiết ---
joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(kmeans_final, 'kmeans_model.joblib')
cluster_personas.to_csv('cluster_personas.csv')

print("\nĐã export thành công 3 file model.")