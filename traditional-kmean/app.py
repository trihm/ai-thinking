from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Tải các model và dữ liệu cần thiết KHI ỨNG DỤNG KHỞI ĐỘNG
try:
    preprocessor = joblib.load('preprocessor.joblib')
    kmeans_model = joblib.load('kmeans_model.joblib')
    cluster_personas = pd.read_csv('cluster_personas.csv', index_col='cluster')
    print("Tải model và dữ liệu thành công!")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file model. Vui lòng chạy 'train_and_export_model.py' trước.")
    exit()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.get_json()
    input_df = pd.DataFrame([user_input])
    
    # Ép kiểu dữ liệu
    for col in ['age', 'total_income', 'rent_spending', 'food_spending', 'transport_spending', 'shopping_spending', 'entertainment_spending', 'goal_amount', 'goal_months']:
        input_df[col] = pd.to_numeric(input_df[col])

    # Feature Engineering cho dữ liệu mới
    input_df['rent_ratio'] = input_df['rent_spending'] / input_df['total_income']
    input_df['food_ratio'] = input_df['food_spending'] / input_df['total_income']
    input_df['transport_ratio'] = input_df['transport_spending'] / input_df['total_income']
    input_df['shopping_ratio'] = input_df['shopping_spending'] / input_df['total_income']
    input_df['entertainment_ratio'] = input_df['entertainment_spending'] / input_df['total_income']
    
    # ĐÃ LOẠI BỎ 'age' KHỎI DANH SÁCH FEATURE PHÂN CỤM
    features_for_clustering = ['total_income', 'rent_ratio', 'food_ratio', 'transport_ratio', 'shopping_ratio', 'entertainment_ratio', 'city']
    
    # Dự đoán cụm
    processed_input = preprocessor.transform(input_df[features_for_clustering])
    cluster_prediction = kmeans_model.predict(processed_input)[0]
    persona = cluster_personas.loc[cluster_prediction]

    # --- LOGIC GỢI Ý NGÂN SÁCH HYBRID MỚI ---
    user_income = input_df['total_income'].iloc[0]
    suggested_budget = {}
    income_after_fixed_costs = user_income

    # 1. Trừ đi các chi phí CỐ ĐỊNH do người dùng xác định
    if user_input.get('rent_fixed'):
        rent_cost = input_df['rent_spending'].iloc[0]
        suggested_budget['rent'] = int(rent_cost)
        income_after_fixed_costs -= rent_cost

    if user_input.get('transport_fixed'):
        transport_cost = input_df['transport_spending'].iloc[0]
        suggested_budget['transport'] = int(transport_cost)
        income_after_fixed_costs -= transport_cost
        
    if income_after_fixed_costs < 0:
        for cat in ['food', 'shopping', 'entertainment', 'savings']:
             suggested_budget[cat] = 0
        if not user_input.get('rent_fixed'): suggested_budget['rent'] = 0
        if not user_input.get('transport_fixed'): suggested_budget['transport'] = 0

    else:
        adjustable_categories = ['food', 'shopping', 'entertainment']
        if not user_input.get('rent_fixed'):
            adjustable_categories.append('rent')
        if not user_input.get('transport_fixed'):
            adjustable_categories.append('transport')

        persona_ratios_for_pool = {}
        for cat in adjustable_categories:
            persona_ratios_for_pool[cat] = persona[f'{cat}_ratio']
        
        all_persona_spending_ratios = persona[['rent_ratio', 'food_ratio', 'transport_ratio', 'shopping_ratio', 'entertainment_ratio']].sum()
        persona_ratios_for_pool['savings'] = 1 - all_persona_spending_ratios
        
        if persona_ratios_for_pool['savings'] < 0:
             persona_ratios_for_pool['savings'] = 0.05 

        total_ratio_of_pool = sum(persona_ratios_for_pool.values())

        if total_ratio_of_pool > 0:
            for category, ratio in persona_ratios_for_pool.items():
                amount = income_after_fixed_costs * (ratio / total_ratio_of_pool)
                suggested_budget[category] = int(amount)
        else:
            suggested_budget['savings'] = int(income_after_fixed_costs)


    # --- Tối ưu hóa mục tiêu ---
    goal_amount = input_df['goal_amount'].iloc[0]
    goal_months = input_df['goal_months'].iloc[0]
    required_monthly_savings = goal_amount / goal_months if goal_months > 0 else 0
    current_spending_total = input_df[['rent_spending', 'food_spending', 'transport_spending', 'shopping_spending', 'entertainment_spending']].sum(axis=1).iloc[0]
    current_savings = user_income - current_spending_total

    goal_analysis = {
        'required_savings': int(required_monthly_savings),
        'current_savings': int(current_savings)
    }

    # --- Gợi ý đầu tư ---
    risk_profile = input_df['risk_profile'].iloc[0]
    allocations = {
        'An toàn': {'Tiết kiệm': 0.7, 'Quỹ Trái phiếu': 0.3, 'Quỹ Cổ phiếu': 0.0},
        'Cân bằng': {'Tiết kiệm': 0.3, 'Quỹ Trái phiếu': 0.4, 'Quỹ Cổ phiếu': 0.3},
        'Tăng trưởng': {'Tiết kiệm': 0.1, 'Quỹ Trái phiếu': 0.2, 'Quỹ Cổ phiếu': 0.7}
    }
    investment_plan = {asset: ratio for asset, ratio in allocations[risk_profile].items() if ratio > 0}

    return jsonify({
        'predicted_cluster': int(cluster_prediction),
        'suggested_budget': suggested_budget,
        'goal_analysis': goal_analysis,
        'investment_plan': investment_plan
    })

# --- NEW: FEEDBACK ENDPOINT ---
@app.route('/feedback', methods=['POST'])
def handle_feedback():
    data = request.get_json()
    
    # Chuẩn bị dữ liệu để ghi ra file
    user_input = data.get('user_input', {})
    recommendation = data.get('recommendation', {})
    suggested_budget = recommendation.get('suggested_budget', {})
    
    # Làm phẳng dữ liệu thành một hàng
    flat_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'feedback': data.get('feedback_type'),
        
        # Dữ liệu người dùng nhập
        **user_input,
        
        # Kết quả từ model
        'predicted_cluster': recommendation.get('predicted_cluster'),
        'suggested_rent': suggested_budget.get('rent'),
        'suggested_food': suggested_budget.get('food'),
        'suggested_transport': suggested_budget.get('transport'),
        'suggested_shopping': suggested_budget.get('shopping'),
        'suggested_entertainment': suggested_budget.get('entertainment'),
        'suggested_savings': suggested_budget.get('savings'),
    }

    file_path = 'feedback.csv'
    # Kiểm tra file có tồn tại không để quyết định có ghi header hay không
    file_exists = os.path.isfile(file_path)

    try:
        # Mở file ở chế độ 'a' (append)
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            # Dùng DictWriter để dễ dàng ghi từ dictionary
            writer = csv.DictWriter(f, fieldnames=flat_data.keys())
            if not file_exists:
                writer.writeheader()  # Ghi header nếu file mới được tạo
            writer.writerow(flat_data)
        
        return jsonify({'status': 'success', 'message': 'Feedback received successfully.'})
    except Exception as e:
        print(f"Error writing to CSV: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to save feedback.'}), 500


if __name__ == '__main__':
    app.run(debug=True)

