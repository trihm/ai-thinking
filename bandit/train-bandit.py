import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

FEEDBACK_FILE = 'feedback.csv'
PREPROCESSOR_FILE = 'preprocessor.joblib'
PERSONA_FILE = 'cluster_personas.csv'
OUTPUT_MODEL_FILE = 'bandit_model.joblib'

try:
    # Đọc dữ liệu feedback đã thu thập
    df = pd.read_csv(FEEDBACK_FILE)
    print(f"Đã đọc {len(df)} hàng dữ liệu feedback.")
except FileNotFoundError:
    print(f"Không tìm thấy file {FEEDBACK_FILE}. Không thể huấn luyện bandit model.")
    exit()

# Lọc dữ liệu: Chỉ huấn luyện trên các feedback rõ ràng 'like' hoặc 'dislike'
df = df[df['feedback'].isin(['like', 'dislike'])].copy()
if len(df) < 20: # Cần đủ dữ liệu để huấn luyện (nên là 100+)
    print(f"Không đủ dữ liệu (chỉ có {len(df)} samples). Cần ít nhất 20 samples 'like'/'dislike'.")
    exit()

# --- 1. CHUẨN BỊ TARGET (Y) ---
# Đây là REWARD của chúng ta
df['reward'] = df['feedback'].map({'like': 1, 'dislike': 0})
y = df['reward'].values
print("Đã xử lý Reward (Y).")

# --- 2. CHUẨN BỊ FEATURES (X) ---
# X = (Vector Context của User) + (Vector One-Hot của Action)

# 2a. Xử lý Context (S)
# Chúng ta phải tái tạo chính xác các features như trong app.py
try:
    # Ép kiểu các cột số đã lưu từ CSV
    numeric_cols = ['total_income', 'rent_spending', 'food_spending', 'transport_spending', 'shopping_spending', 'entertainment_spending']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['total_income'] = df['total_income'].replace(0, 1) # Tránh chia cho 0

    df['rent_ratio'] = df['rent_spending'] / df['total_income']
    df['food_ratio'] = df['food_spending'] / df['total_income']
    df['transport_ratio'] = df['transport_spending'] / df['total_income']
    df['shopping_ratio'] = df['shopping_spending'] / df['total_income']
    df['entertainment_ratio'] = df['entertainment_spending'] / df['total_income']
    
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    # Tải preprocessor đã được huấn luyện (từ train.py)
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    
    features_for_clustering = ['total_income', 'rent_ratio', 'food_ratio', 'transport_ratio', 'shopping_ratio', 'entertainment_ratio', 'city']
    
    # Tạo vector context
    context_features = preprocessor.transform(df[features_for_clustering])
    print(f"Đã xử lý Context Features (S). Shape: {context_features.shape}")

except Exception as e:
    print(f"Lỗi khi xử lý context features: {e}")
    print("Hãy đảm bảo 'preprocessor.joblib' tồn tại và các cột trong feedback.csv khớp.")
    exit()


# 2b. Xử lý Action (A)
# Chúng ta cần One-Hot Encode cột 'chosen_action'
personas = pd.read_csv(PERSONA_FILE)
OPTIMAL_K = len(personas)
all_possible_actions = [list(range(OPTIMAL_K))] # [[0, 1, 2, 3, 4]]

action_encoder = OneHotEncoder(categories=all_possible_actions, sparse_output=False)
action_features = action_encoder.fit_transform(df[['chosen_action']])
print(f"Đã xử lý Action Features (A) cho K={OPTIMAL_K}. Shape: {action_features.shape}")

# 2c. Ghép Context và Action để tạo X cuối cùng
# Đây là "trạng thái-hành động" (state-action pair)
X = np.hstack([context_features, action_features])
print(f"Đã ghép X (S+A). Shape cuối cùng: {X.shape}")


# --- 3. HUẤN LUYỆN BANDIT MODEL ---
# Chia dữ liệu để kiểm tra độ chính xác (không bắt buộc nhưng nên làm)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Chúng ta dùng Logistic Regression vì nó nhanh, cho ra xác suất, và hoạt động tốt
# class_weight='balanced' rất quan trọng nếu bạn có nhiều 'dislike' hơn 'like' hoặc ngược lại
model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Đánh giá model (trên tập test)
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"\nĐã huấn luyện xong Logistic Regression Bandit.")
print(f"Độ chính xác (Accuracy) trên tập Test: {acc * 100:.2f}%")

# --- 4. LƯU MODEL ---
# Huấn luyện lại trên TOÀN BỘ dữ liệu để có model tốt nhất
final_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
final_model.fit(X, y) # Huấn luyện trên tất cả data

# Ghi đè file model cũ
joblib.dump(final_model, OUTPUT_MODEL_FILE)
print(f"\nĐã lưu model cuối cùng vào file: {OUTPUT_MODEL_FILE}")