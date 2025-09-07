from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime
from sklearn.linear_model import LogisticRegression # Chúng ta sẽ dùng Logistic Regression làm model bandit

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# --- TẢI CÁC ARTIFACTS CẦN THIẾT ---
try:
    # 1. Tải preprocessor (để xử lý Context của người dùng)
    preprocessor = joblib.load('preprocessor.joblib')
    
    # 2. Tải các Personas (đây là các "Actions" của chúng ta)
    cluster_personas = pd.read_csv('cluster_personas.csv', index_col='cluster')
    OPTIMAL_K = len(cluster_personas) # Tự động phát hiện số cụm (số Actions)
    print(f"Đã tải {OPTIMAL_K} personas (actions).")

    # 3. Tải mô hình Bandit (học từ feedback)
    #    Nếu file này không tồn tại, chúng ta sẽ tạo một mô hình "ngu ngốc" (dummy) chỉ để khám phá
    BANDIT_MODEL_PATH = 'bandit_model.joblib'
    if os.path.exists(BANDIT_MODEL_PATH):
        bandit_model = joblib.load(BANDIT_MODEL_PATH)
        print("Đã tải bandit model thành công.")
    else:
        # Nếu chưa có model, tạo một model giả lập. 
        # Chúng ta sẽ chỉ khám phá ngẫu nhiên cho đến khi có dữ liệu feedback.
        bandit_model = None 
        print("Không tìm thấy bandit_model.joblib. Chạy ở chế độ 'chỉ khám phá' (explore-only).")

except FileNotFoundError:
    print("Lỗi: Không tìm thấy 'preprocessor.joblib' hoặc 'cluster_personas.csv'.")
    print("Vui lòng chạy 'train.py' trước tiên để tạo các file này.")
    exit()

# --- CẤU HÌNH BANDIT ---
EPSILON = 0.1 # Tỷ lệ khám phá (10% thời gian sẽ chọn 1 action ngẫu nhiên)

def get_budget_from_persona(user_income, persona, user_input_df, user_flags):
    """
    Hàm logic helper để tính toán ngân sách từ một persona CỤ THỂ.
    Logic này được tách ra vì chúng ta cần gọi nó cho action được chọn.
    """
    suggested_budget = {}
    income_after_fixed_costs = user_income

    # 1. Trừ chi phí cố định
    if user_flags.get('rent_fixed'):
        rent_cost = user_input_df['rent_spending'].iloc[0]
        suggested_budget['rent'] = int(rent_cost)
        income_after_fixed_costs -= rent_cost

    if user_flags.get('transport_fixed'):
        transport_cost = user_input_df['transport_spending'].iloc[0]
        suggested_budget['transport'] = int(transport_cost)
        income_after_fixed_costs -= transport_cost
        
    if income_after_fixed_costs < 0:
        income_after_fixed_costs = 0 # Ngăn chặn ngân sách âm

    # 2. Phân bổ phần còn lại dựa trên tỷ lệ của PERSONA ĐƯỢC CHỌN
    adjustable_categories = ['food', 'shopping', 'entertainment']
    if not user_flags.get('rent_fixed'):
        adjustable_categories.append('rent')
    if not user_flags.get('transport_fixed'):
        adjustable_categories.append('transport')

    # Lấy tỷ lệ từ persona
    persona_ratios_for_pool = {}
    for cat in adjustable_categories:
        persona_ratios_for_pool[cat] = persona[f'{cat}_ratio']
    
    # Tính tỷ lệ tiết kiệm của persona
    all_persona_spending_ratios = persona[['rent_ratio', 'food_ratio', 'transport_ratio', 'shopping_ratio', 'entertainment_ratio']].sum()
    persona_ratios_for_pool['savings'] = max(0.05, 1 - all_persona_spending_ratios) # Đảm bảo tiết kiệm luôn > 0

    total_ratio_of_pool = sum(persona_ratios_for_pool.values())

    if total_ratio_of_pool > 0:
        for category, ratio in persona_ratios_for_pool.items():
            amount = income_after_fixed_costs * (ratio / total_ratio_of_pool)
            suggested_budget[category] = int(amount)
    else:
        # Trường hợp hiếm gặp nếu tất cả tỷ lệ = 0
        suggested_budget['savings'] = int(income_after_fixed_costs)
        
    return suggested_budget


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.get_json()
    user_flags = { # Tách riêng các cờ boolean
        'rent_fixed': user_input.pop('rent_fixed', False),
        'transport_fixed': user_input.pop('transport_fixed', False)
    }
    input_df = pd.DataFrame([user_input])
    
    # Ép kiểu dữ liệu
    numeric_cols = ['age', 'total_income', 'rent_spending', 'food_spending', 'transport_spending', 'shopping_spending', 'entertainment_spending', 'goal_amount', 'goal_months']
    for col in numeric_cols:
        input_df[col] = pd.to_numeric(input_df[col])

    # Feature Engineering (Context)
    input_df['rent_ratio'] = input_df['rent_spending'] / input_df['total_income']
    input_df['food_ratio'] = input_df['food_spending'] / input_df['total_income']
    input_df['transport_ratio'] = input_df['transport_spending'] / input_df['total_income']
    input_df['shopping_ratio'] = input_df['shopping_spending'] / input_df['total_income']
    input_df['entertainment_ratio'] = input_df['entertainment_spending'] / input_df['total_income']
    input_df.replace([np.inf, -np.inf], 0, inplace=True) # Xử lý chia cho 0
    input_df.fillna(0, inplace=True)
    
    features_for_clustering = ['total_income', 'rent_ratio', 'food_ratio', 'transport_ratio', 'shopping_ratio', 'entertainment_ratio', 'city']
    
    # --- LOGIC CONTEXTUAL BANDIT BẮT ĐẦU TỪ ĐÂY ---
    
    # 1. Lấy Context Vector (Xử lý input của người dùng)
    # Đây là vector đặc trưng của người dùng (Context)
    user_context_vector = preprocessor.transform(input_df[features_for_clustering]) # Shape: (1, num_features)

    chosen_action_id = -1

    # 2. Quyết định Khám phá (Explore) hay Khai thác (Exploit)
    if bandit_model is None or np.random.rand() < EPSILON:
        # EXPLORE: Chọn một action (persona) ngẫu nhiên
        chosen_action_id = np.random.randint(0, OPTIMAL_K)
        print(f"Chế độ: EXPLORE. Chọn ngẫu nhiên Action: {chosen_action_id}")
    else:
        # EXPLOIT: Sử dụng model bandit để dự đoán P(like) cho MỌI action
        print("Chế độ: EXPLOIT. Đang dự đoán điểm cho tất cả actions...")
        
        # Chúng ta cần tạo một batch dữ liệu để dự đoán:
        # [ (context, action_0), (context, action_1), ..., (context, action_K-1) ]
        
        # Lặp lại context vector K lần
        context_batch = np.repeat(user_context_vector, OPTIMAL_K, axis=0) # Shape: (K, num_features)
        
        # Tạo ma trận One-Hot cho K actions
        action_batch_identity = np.eye(OPTIMAL_K) # Shape: (K, K)
        
        # Ghép chúng lại để tạo batch X_test đầy đủ
        # Đây là các đặc trưng đầu vào mà bandit model của chúng ta mong đợi
        X_test_batch = np.hstack([context_batch, action_batch_identity])
        
        # Dự đoán xác suất "like" (class=1) cho mỗi action
        try:
            proba_like = bandit_model.predict_proba(X_test_batch)[:, 1] # Lấy xác suất của class 1
            
            # Chọn action có xác suất "like" cao nhất
            chosen_action_id = np.argmax(proba_like)
            
            print(f"Điểm dự đoán (P(like)): {proba_like}")
            print(f"Chọn Action tốt nhất: {chosen_action_id}")
            
        except Exception as e:
            print(f"Lỗi khi dự đoán bandit: {e}. Quay lại chế độ Explore.")
            chosen_action_id = np.random.randint(0, OPTIMAL_K)


    # 3. Tạo đề xuất dựa trên ACTION ĐÃ CHỌN
    chosen_persona = cluster_personas.loc[chosen_action_id]
    user_income = input_df['total_income'].iloc[0]
    
    # Gọi hàm helper để tính ngân sách
    suggested_budget = get_budget_from_persona(user_income, chosen_persona, input_df, user_flags)

    # --- Các logic khác giữ nguyên ---
    goal_amount = input_df['goal_amount'].iloc[0]
    goal_months = input_df['goal_months'].iloc[0]
    required_monthly_savings = goal_amount / goal_months if goal_months > 0 else 0
    current_spending_total = input_df[['rent_spending', 'food_spending', 'transport_spending', 'shopping_spending', 'entertainment_spending']].sum(axis=1).iloc[0]
    current_savings = user_income - current_spending_total

    goal_analysis = {
        'required_savings': int(required_monthly_savings),
        'current_savings': int(current_savings)
    }

    risk_profile = input_df['risk_profile'].iloc[0]
    allocations = {
        'An toàn': {'Tiết kiệm': 0.7, 'Quỹ Trái phiếu': 0.3, 'Quỹ Cổ phiếu': 0.0},
        'Cân bằng': {'Tiết kiệm': 0.3, 'Quỹ Trái phiếu': 0.4, 'Quỹ Cổ phiếu': 0.3},
        'Tăng trưởng': {'Tiết kiệm': 0.1, 'Quỹ Trái phiếu': 0.2, 'Quỹ Cổ phiếu': 0.7}
    }
    investment_plan = {asset: ratio for asset, ratio in allocations[risk_profile].items() if ratio > 0}

    return jsonify({
        # Quan trọng: Gửi lại 'chosen_action_id' để client gửi lại trong feedback
        'chosen_action_id': int(chosen_action_id), 
        'suggested_budget': suggested_budget,
        'goal_analysis': goal_analysis,
        'investment_plan': investment_plan
        # Lưu ý: 'predicted_cluster' (cụm gốc của user) không còn liên quan đến đề xuất nữa
    })

# --- CẬP NHẬT ENDPOINT FEEDBACK ---
@app.route('/feedback', methods=['POST'])
def handle_feedback():
    data = request.get_json()
    
    user_input = data.get('user_input', {})
    recommendation = data.get('recommendation', {})
    feedback_type = data.get('feedback_type') # Giả sử client gửi 'like' hoặc 'dislike'
    
    # Lấy action đã được thực hiện (mà chúng ta đã gửi cho client)
    chosen_action = recommendation.get('chosen_action_id')

    # Làm phẳng dữ liệu để ghi ra file CSV (để huấn luyện bandit)
    flat_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'feedback': feedback_type, # Đây là REWARD (Y)
        'chosen_action': chosen_action, # Đây là ACTION (A)
        
        # Tất cả dữ liệu sau đây là CONTEXT (S)
        **user_input,
    }

    file_path = 'feedback.csv'
    file_exists = os.path.isfile(file_path)

    try:
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            # Đảm bảo fieldnames bao gồm tất cả các key từ flat_data
            fieldnames = flat_data.keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            writer.writerow(flat_data)
        
        return jsonify({'status': 'success', 'message': 'Feedback received.'})
    except Exception as e:
        print(f"Error writing to CSV: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to save feedback.'}), 500


if __name__ == '__main__':
    app.run(debug=True)