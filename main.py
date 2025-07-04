import os
import json
import time
import math
import random
import threading
import logging
from collections import defaultdict, deque
from flask import Flask, jsonify
from flask_cors import CORS
import requests # Import the requests library for HTTP requests
from waitress import serve # Ensure waitress is imported for production server

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Helper Functions ---
def _get_history_strings(history_list):
    """Hàm trợ giúp để lấy danh sách chuỗi 'Tài'/'Xỉu' từ danh sách dict."""
    return [item['ket_qua'] for item in history_list]

def _normalize_features(features):
    """Chuẩn hóa các đặc trưng để chúng có cùng khoảng giá trị."""
    normalized_features = []
    # Giả định giá trị max cho các đặc trưng để chuẩn hóa bằng cách chia
    # Các giá trị này có thể cần được tinh chỉnh dựa trên dữ liệu thực tế
    max_current_streak = 500 # MAX_HISTORY_LEN
    max_previous_streak = 500 # MAX_HISTORY_LEN
    # balance_short và balance_long đã ở trong khoảng [-1, 1]
    # volatility đã ở trong khoảng [0, 1]
    max_alternations = 10 # Trong 10 kết quả gần nhất, max alternations là 9

    # Kiểm tra và chuẩn hóa từng feature
    # current_streak
    normalized_features.append(features[0] / max_current_streak)
    # previous_streak_len
    normalized_features.append(features[1] / max_previous_streak)
    # balance_short (giữ nguyên, đã chuẩn hóa)
    normalized_features.append(features[2])
    # balance_long (giữ nguyên, đã chuẩn hóa)
    normalized_features.append(features[3])
    # volatility (giữ nguyên, đã chuẩn hóa)
    normalized_features.append(features[4])
    # alternations
    normalized_features.append(features[5] / max_alternations)

    return normalized_features

# 1. Định nghĩa các Patterns (Mở rộng với các pattern phức tạp và siêu phức tạp)
def define_patterns():
    """
    Định nghĩa một bộ sưu tập lớn các patterns từ đơn giản đến siêu phức tạp.
    Mỗi pattern là một hàm lambda nhận lịch sử (dạng chuỗi) và trả về True nếu khớp.
    """
    patterns = {
        # --- Cầu Bệt (Streaks) ---
        "Bệt": lambda h: len(h) >= 3 and h[-1] == h[-2] == h[-3],
        "Bệt siêu dài": lambda h: len(h) >= 5 and all(x == h[-1] for x in h[-5:]),
        "Bệt gãy nhẹ": lambda h: len(h) >= 4 and h[-1] != h[-2] and h[-2] == h[-3] == h[-4],
        "Bệt gãy sâu": lambda h: len(h) >= 5 and h[-1] != h[-2] and all(x == h[-2] for x in h[-5:-1]),
        "Bệt xen kẽ ngắn": lambda h: len(h) >= 4 and h[-4:-2] == [h[-4]]*2 and h[-2:] == [h[-2]]*2 and h[-4] != h[-2],
        "Bệt ngược": lambda h: len(h) >= 4 and h[-1] == h[-2] and h[-3] == h[-4] and h[-1] != h[-3],
        "Xỉu kép": lambda h: len(h) >= 2 and h[-1] == 'Xỉu' and h[-2] == 'Xỉu',
        "Tài kép": lambda h: len(h) >= 2 and h[-1] == 'Tài' and h[-2] == 'Tài',
        "Ngẫu nhiên bệt": lambda h: len(h) > 8 and 0.4 < (h[-8:].count('Tài') / 8) < 0.6 and h[-1] == h[-2],

        # --- Cầu Đảo (Alternating) ---
        "Đảo 1-1": lambda h: len(h) >= 4 and h[-1] != h[-2] and h[-2] != h[-3] and h[-3] != h[-4],
        "Xen kẽ dài": lambda h: len(h) >= 5 and all(h[i] != h[i+1] for i in range(-5, -1)),
        "Xen kẽ": lambda h: len(h) >= 3 and h[-1] != h[-2] and h[-2] != h[-3],
        "Xỉu lắc": lambda h: len(h) >= 4 and h[-4:] == ['Xỉu', 'Tài', 'Xỉu', 'Tài'],
        "Tài lắc": lambda h: len(h) >= 4 and h[-4:] == ['Tài', 'Xỉu', 'Tài', 'Xỉu'],
        
        # --- Cầu theo nhịp (Rhythmic) ---
        "Kép 2-2": lambda h: len(h) >= 4 and h[-4:] == [h[-4], h[-4], h[-2], h[-2]] and h[-4] != h[-2],
        "Nhịp 3-3": lambda h: len(h) >= 6 and all(x == h[-6] for x in h[-6:-3]) and all(x == h[-3] for x in h[-3:]),
        "Nhịp 4-4": lambda h: len(h) >= 8 and h[-8:-4] == [h[-8]]*4 and h[-4:] == [h[-4]]*4 and h[-8] != h[-4],
        "Lặp 2-1": lambda h: len(h) >= 3 and h[-3:-1] == [h[-3], h[-3]] and h[-1] != h[-3],
        "Lặp 3-2": lambda h: len(h) >= 5 and h[-5:-2] == [h[-5]]*3 and h[-2:] == [h[-2]]*2 and h[-5] != h[-2],
        "Cầu 3-1": lambda h: len(h) >= 4 and all(x == h[-4] for x in h[-4:-1]) and h[-1] != h[-4],
        "Cầu 4-1": lambda h: len(h) >= 5 and h[-5:-1] == [h[-5]]*4 and h[-1] != h[-5],
        "Cầu 1-2-1": lambda h: len(h) >= 4 and h[-4] != h[-3] and h[-3]==h[-2] and h[-2] != h[-1] and h[-4]==h[-1],
        "Cầu 2-1-2": lambda h: len(h) >= 5 and h[-5:-3] == [h[-5]]*2 and h[-2] != h[-5] and h[-1] == h[-5],
        "Cầu 3-1-2": lambda h: len(h) >= 6 and h[-6:-3]==[h[-6]]*3 and h[-3]!=h[-2] and h[-2:]==[h[-2]]*2 and len(set(h[-6:])) == 2,
        "Cầu 1-2-3": lambda h: len(h) >= 6 and h[-6:-5]==[h[-6]] and h[-5:-3]==[h[-5]]*2 and h[-3:]==[h[-3]]*3 and len(set(h[-6:])) == 2,
        "Dài ngắn đảo": lambda h: len(h) >= 5 and h[-5:-2] == [h[-5]] * 3 and h[-2] != h[-1] and h[-2] != h[-5],

        # --- Cầu Chu Kỳ & Đối Xứng (Cyclic & Symmetric) ---
        "Chu kỳ 2": lambda h: len(h) >= 4 and h[-1] == h[-3] and h[-2] == h[-4],
        "Chu kỳ 3": lambda h: len(h) >= 6 and h[-1] == h[-4] and h[-2] == h[-5] and h[-3] == h[-6],
        "Chu kỳ 4": lambda h: len(h) >= 8 and h[-1] == h[-5] and h[-2] == h[-6] and h[-3] == h[-7] and h[-4] == h[-8], # Sửa lại để đảm bảo đúng chu kỳ
        "Đối xứng (Gương)": lambda h: len(h) >= 5 and h[-1] == h[-5] and h[-2] == h[-4],
        "Bán đối xứng": lambda h: len(h) >= 5 and h[-1] == h[-4] and h[-2] == h[-5],
        "Ngược chu kỳ": lambda h: len(h) >= 4 and h[-1] == h[-4] and h[-2] == h[-3] and h[-1] != h[-2],
        "Chu kỳ biến đổi": lambda h: len(h) >= 5 and h[-5:] == [h[-5], h[-4], h[-5], h[-4], h[-5]],
        "Cầu linh hoạt": lambda h: len(h) >= 6 and h[-1]==h[-3]==h[-5] and h[-2]==h[-4]==h[-6],
        "Chu kỳ tăng": lambda h: len(h) >= 6 and h[-6:] == [h[-6], h[-5], h[-6], h[-5], h[-6], h[-5]] and h[-6] != h[-5],
        "Chu kỳ giảm": lambda h: len(h) >= 6 and h[-6:] == [h[-6], h[-6], h[-5], h[-5], h[-4], h[-4]] and len(set(h[-6:])) == 3,
        "Cầu lặp": lambda h: len(h) >= 6 and h[-6:-3] == h[-3:],
        "Gãy ngang": lambda h: len(h) >= 4 and h[-1] == h[-3] and h[-2] == h[-4] and h[-1] != h[-2],

        # --- Cầu Phức Tạp & Tổng Hợp ---
        "Gập ghềnh": lambda h: len(h) >= 5 and h[-5:] == [h[-5], h[-5], h[-3], h[-3], h[-5]],
        "Bậc thang": lambda h: len(h) >= 3 and h[-3:] == [h[-3], h[-3], h[-1]] and h[-3] != h[-1],
        "Cầu đôi": lambda h: len(h) >= 4 and h[-1] == h[-2] and h[-3] != h[-4] and h[-3] != h[-1],
        "Đối ngược": lambda h: len(h) >= 4 and h[-1] == ('Xỉu' if h[-2]=='Tài' else 'Tài') and h[-3] == ('Xỉu' if h[-4]=='Tài' else 'Tài'),
        "Cầu gập": lambda h: len(h) >= 5 and h[-5:] == [h[-5], h[-4], h[-4], h[-2], h[-2]],
        "Phối hợp 1": lambda h: len(h) >= 5 and h[-1] == h[-2] and h[-3] != h[-4],
        "Phối hợp 2": lambda h: len(h) >= 4 and h[-4:] == ['Tài', 'Tài', 'Xỉu', 'Tài'],
        "Phối hợp 3": lambda h: len(h) >= 4 and h[-4:] == ['Xỉu', 'Xỉu', 'Tài', 'Xỉu'],
        "Chẵn lẻ lặp": lambda h: len(h) >= 4 and len(set(h[-4:-2])) == 1 and len(set(h[-2:])) == 1 and h[-1] != h[-3],
        "Cầu dài ngẫu": lambda h: len(h) >= 7 and all(x == h[-7] for x in h[-7:-3]) and len(set(h[-3:])) > 1,
        
        # --- Cầu Dựa Trên Phân Bố (Statistical) ---
        "Ngẫu nhiên": lambda h: len(h) > 10 and 0.4 < (h[-10:].count('Tài') / 10) < 0.6,
        "Đa dạng": lambda h: len(h) >= 5 and len(set(h[-5:])) == 2,
        "Phân cụm": lambda h: len(h) >= 6 and (all(x == 'Tài' for x in h[-6:-3]) or all(x == 'Xỉu' for x in h[-6:-3])),
        "Lệch ngẫu nhiên": lambda h: len(h) > 10 and (h[-10:].count('Tài') / 10 > 0.7 or h[-10:].count('Xỉu') / 10 > 0.7),

        # --- Siêu Cầu (Super Patterns) ---
        "Cầu Tiến 1-1-2-2": lambda h: len(h) >= 6 and h[-6:] == [h[-6], h[-5], h[-4], h[-4], h[-2], h[-2]] and len(set(h[-6:])) == 2,
        "Cầu Lùi 3-2-1": lambda h: len(h) >= 6 and h[-6:-3]==[h[-6]]*3 and h[-3:-1]==[h[-3]]*2 and h[-1]!=h[-3] and len(set(h[-6:])) == 2,
        "Cầu Sandwich": lambda h: len(h) >= 5 and h[-1] == h[-5] and h[-2] == h[-3] == h[-4] and h[-1] != h[-2],
        "Cầu Thang máy": lambda h: len(h) >= 7 and h[-7:] == [h[-7],h[-7],h[-5],h[-5],h[-3],h[-3],h[-1]] and len(set(h[-7:]))==4, # T-T-X-X-T-T-X
        "Cầu Sóng vỗ": lambda h: len(h) >= 8 and h[-8:] == [h[-8],h[-8],h[-6],h[-8],h[-8],h[-6],h[-8],h[-8]],
        "Cầu Đỉnh-Đáy": lambda h: len(h) >= 7 and h[-7:] == [h[-7], h[-7], h[-6], h[-6], h[-5], h[-5], h[-4]] and len(set(h[-7:])) == 3, # T-T-X-X-T-T-X
        "Cầu Lưới": lambda h: len(h) >= 6 and h[-6:] == [h[-6], h[-5], h[-5], h[-4], h[-4], h[-3]] and len(set(h[-6:])) == 3, # T-X-X-T-T-X
    }
    return patterns

# 2. Các hàm cập nhật và huấn luyện mô hình
def update_transition_matrix(app, prev_result, current_result):
    if not prev_result: return
    prev_idx = 0 if prev_result == 'Tài' else 1
    curr_idx = 0 if current_result == 'Tài' else 1
    app.transition_counts[prev_idx][curr_idx] += 1
    total_transitions = sum(app.transition_counts[prev_idx])
    alpha = 1 # Laplace smoothing để tránh xác suất bằng 0
    num_outcomes = 2
    app.transition_matrix[prev_idx][0] = (app.transition_counts[prev_idx][0] + alpha) / (total_transitions + alpha * num_outcomes)
    app.transition_matrix[prev_idx][1] = (app.transition_counts[prev_idx][1] + alpha) / (total_transitions + alpha * num_outcomes)

def update_pattern_accuracy(app, detected_pattern_name, actual_result, history_str_used_for_detection):
    """
    Cập nhật độ chính xác của pattern VÀ thống kê kết quả tiếp theo của pattern đó.
    history_str_used_for_detection là lịch sử tại thời điểm pattern được phát hiện.
    """
    if not detected_pattern_name: return

    # Cập nhật tổng số và số lần thành công
    stats = app.pattern_accuracy[detected_pattern_name]
    stats['total'] += 1
    
    # Dự đoán của pattern là dựa trên thống kê, nên chúng ta cần biết pattern đó đã "dự đoán" gì
    # Giả sử pattern dự đoán kết quả có xác suất cao hơn sau nó
    p_stats = app.pattern_outcome_stats[detected_pattern_name]
    total_p_stats_outcomes = p_stats['Tài'] + p_stats['Xỉu']
    
    predicted_by_pattern = None
    # Nếu có đủ thống kê cho pattern, sử dụng thống kê để xác định dự đoán của pattern đó
    if total_p_stats_outcomes >= 5: # Đủ dữ liệu để tin cậy vào thống kê
        prob_tai = p_stats['Tài'] / total_p_stats_outcomes
        prob_xiu = p_stats['Xỉu'] / total_p_stats_outcomes
        if prob_tai > prob_xiu:
            predicted_by_pattern = 'Tài'
        elif prob_xiu > prob_tai:
            predicted_by_pattern = 'Xỉu'
        else: # Xác suất bằng nhau, chọn ngẫu nhiên để không ưu tiên mặc định
            predicted_by_pattern = random.choice(['Tài', 'Xỉu'])
    else: # Nếu chưa đủ thống kê, dùng logic ban đầu của pattern (theo cầu/bẻ cầu)
        if history_str_used_for_detection and len(history_str_used_for_detection) > 0:
            last_h = history_str_used_for_detection[-1]
            anti_last_h = 'Xỉu' if last_h == 'Tài' else 'Tài'
            
            # Quy tắc tạm thời để đánh giá ban đầu hiệu suất của pattern
            if any(p in detected_pattern_name for p in ["Bệt", "kép", "Nhịp", "Sóng vỗ", "Cầu Tiến", "Cầu Lùi", "Cầu 3-1", "Cầu 4-1", "Lặp"]):
                predicted_by_pattern = last_h
            elif any(p in detected_pattern_name for p in ["Đảo", "Xen kẽ", "lắc", "Đối ngược", "gãy", "Bậc thang", "Chu kỳ", "Gương", "Bán đối xứng"]):
                predicted_by_pattern = anti_last_h
            # Các pattern khác có thể cần quy tắc riêng hoặc mặc định là không dự đoán mạnh
            else:
                predicted_by_pattern = random.choice(['Tài', 'Xỉu']) # Coi như không có ưu tiên rõ ràng

    if predicted_by_pattern and predicted_by_pattern == actual_result:
        stats['success'] += 1

    # Cập nhật thống kê kết quả tiếp theo cho pattern
    p_stats = app.pattern_outcome_stats[detected_pattern_name]
    if actual_result == 'Tài':
        p_stats['Tài'] += 1
    else:
        p_stats['Xỉu'] += 1
    p_stats['total'] += 1


def train_logistic_regression(app, features, actual_result):
    if not features: return
    
    # Chuẩn hóa features trước khi huấn luyện
    normalized_features = _normalize_features(features)

    y = 1.0 if actual_result == 'Tài' else 0.0 # 1.0 cho Tài, 0.0 cho Xỉu
    z = app.logistic_bias + sum(w * f for w, f in zip(app.logistic_weights, normalized_features))
    
    p = 0.0 # Default value in case of error
    try:
        p = 1.0 / (1.0 + math.exp(-z)) # Sigmoid function
    except OverflowError: # Xử lý trường hợp z quá lớn hoặc quá nhỏ
        p = 0.0 if z < 0 else 1.0
        
    error = y - p # Sai số
    
    # Cập nhật bias
    app.logistic_bias += app.learning_rate * error
    # Cập nhật trọng số
    for i in range(len(app.logistic_weights)):
        gradient = error * normalized_features[i]
        regularization_term = app.regularization * app.logistic_weights[i]
        app.logistic_weights[i] += app.learning_rate * (gradient - regularization_term)

def update_model_weights(app):
    """Cập nhật trọng số của các mô hình trong ensemble dựa trên hiệu suất động."""
    total_weighted_accuracy = 0
    accuracies = {}
    
    # Tính toán độ chính xác thực tế của từng mô hình
    for name, perf in app.model_performance.items():
        if perf['total'] >= 10: # Cần ít nhất 10 điểm dữ liệu để có độ chính xác ý nghĩa
            accuracy = perf['success'] / perf['total']
            accuracies[name] = accuracy
        else: # Nếu chưa đủ dữ liệu, dùng độ chính xác mặc định (hoặc khởi tạo hợp lý)
            accuracies[name] = app.default_model_weights[name] # Sử dụng trọng số mặc định làm đại diện cho "độ chính xác" ban đầu
    
    # Tính tổng weighted accuracy để chuẩn hóa
    for name in app.model_weights:
        # Trọng số mới tỉ lệ thuận với độ chính xác hiện tại của mô hình đó
        # Thêm một hệ số "làm mượt" để tránh quá nhạy cảm với biến động nhỏ
        app.model_weights[name] = (app.model_weights[name] * 0.8) + (accuracies[name] * 0.2) # Ví dụ: 80% trọng số cũ, 20% độ chính xác mới
        total_weighted_accuracy += app.model_weights[name]

    # Chuẩn hóa lại để tổng bằng 1
    if total_weighted_accuracy > 0:
        for name in app.model_weights:
            app.model_weights[name] /= total_weighted_accuracy
    else: # Trường hợp tổng bằng 0, khởi tạo lại trọng số mặc định
        app.model_weights = app.default_model_weights.copy()
        
    logging.info(f"Updated model weights: {app.model_weights}")

# 3. Các hàm dự đoán cốt lõi
def detect_pattern(app, history_str):
    detected_patterns = []
    if len(history_str) < 2: return None
    
    # Tổng số lần xuất hiện của tất cả các pattern (để tính điểm recency)
    total_occurrences_all_patterns = sum(s['total'] for s in app.pattern_accuracy.values())
    total_occurrences_all_patterns = max(1, total_occurrences_all_patterns) # Tránh chia cho 0

    for name, func in app.patterns.items():
        try:
            if func(history_str):
                stats = app.pattern_accuracy[name]
                
                # Tính toán accuracy của pattern (dựa trên pattern_accuracy)
                accuracy = (stats['success'] / stats['total']) if stats['total'] >= 5 else 0.55 # Ưu tiên nhẹ nếu chưa đủ dữ liệu
                
                # Điểm Recency (tần suất xuất hiện)
                recency_score = (stats['total'] / total_occurrences_all_patterns) if total_occurrences_all_patterns > 0 else 0
                
                # Trọng số kết hợp độ chính xác lịch sử (70%) và tần suất xuất hiện (30%)
                # Pattern được nhìn thấy nhiều và có độ chính xác cao sẽ được ưu tiên
                weight = (0.7 * accuracy) + (0.3 * recency_score)
                
                detected_patterns.append({'name': name, 'weight': weight})
        except IndexError:
            continue
        except Exception as e:
            logging.error(f"Error detecting pattern {name}: {e}")
            continue

    if not detected_patterns:
        return None
    
    # Trả về pattern có trọng số cao nhất
    return max(detected_patterns, key=lambda x: x['weight'])

def predict_with_pattern(app, history_str, detected_pattern_info):
    """
    Dự đoán dựa trên pattern, sử dụng thống kê kết quả tiếp theo của pattern đó.
    """
    if not detected_pattern_info or len(history_str) < 2:
        return 'Tài', 50.0, "Không có Pattern mạnh" # Mặc định hoặc không dự đoán được

    name = detected_pattern_info['name']
    
    # Lấy thống kê kết quả tiếp theo cho pattern này
    p_stats = app.pattern_outcome_stats[name]

    if p_stats['total'] >= 5: # Chỉ sử dụng thống kê nếu có đủ dữ liệu (ít nhất 5 lần xuất hiện sau pattern)
        prob_tai = p_stats['Tài'] / p_stats['total']
        prob_xiu = p_stats['Xỉu'] / p_stats['total']

        if prob_tai > prob_xiu:
            prediction = 'Tài'
            confidence = prob_tai * 100
        elif prob_xiu > prob_tai:
            prediction = 'Xỉu'
            confidence = prob_xiu * 100
        else: # Bằng nhau, chọn ngẫu nhiên hoặc mặc định
            prediction = random.choice(['Tài', 'Xỉu'])
            confidence = 50.0
            
        reason = f"{name} (Tài:{round(prob_tai*100,1)}%/Xỉu:{round(prob_xiu*100,1)}%)"
        return prediction, confidence, reason
    else:
        # Nếu chưa đủ dữ liệu thống kê cho pattern, quay lại logic dự đoán dựa trên quy tắc cứng (hoặc giảm độ tin cậy)
        last = history_str[-1]
        anti_last = 'Xỉu' if last == 'Tài' else 'Tài'
        
        prediction_rules = {
            "Bệt": last, "Xỉu kép": last, "Tài kép": last, "Bệt siêu dài": last, "Ngẫu nhiên bệt": last,
            "Kép 2-2": last, "Nhịp 3-3": last, "Nhịp 4-4": last, "Lặp 2-1": last, "Lặp 3-2": last,
            "Cầu 3-1": last, "Cầu 4-1": last, "Cầu Tiến 1-1-2-2": last, "Cầu Lùi 3-2-1": last,
            "Cầu Sóng vỗ": last, "Cầu Đỉnh-Đáy": last, "Cầu Lưới": last, "Cầu lặp": last,
            "Phân cụm": last,
            
            "Đảo 1-1": anti_last, "Xen kẽ dài": anti_last, "Xen kẽ": anti_last,
            "Xỉu lắc": anti_last, "Tài lắc": anti_last, "Gãy ngang": anti_last,
            "Cầu 1-2-1": anti_last, "Cầu 2-1-2": anti_last, "Đối xứng (Gương)": anti_last,
            "Ngược chu kỳ": anti_last, "Chu kỳ biến đổi": anti_last,
            "Gập ghềnh": anti_last, "Bậc thang": anti_last, "Đối ngược": anti_last,
            "Cầu đôi": anti_last, "Cầu gập": anti_last, "Phối hợp 1": anti_last,
            "Phối hợp 2": 'Xỉu' if last == 'Tài' else 'Tài', # Ví dụ cho các phối hợp đặc biệt
            "Phối hợp 3": 'Tài' if last == 'Xỉu' else 'Xỉu',
        }
        
        prediction = prediction_rules.get(name, anti_last) # Mặc định là bẻ cầu nếu không khớp quy tắc cứng
        confidence = detected_pattern_info['weight'] * 100 # Độ tin cậy từ trọng số phát hiện pattern
        reason = f"{name} (Logic tạm thời - chưa đủ dữ liệu thống kê)"
        
        return prediction, confidence, reason


def get_logistic_features(history_str):
    if not history_str: return [0.0] * 6
        
    # Feature 1: Current streak length (Độ dài chuỗi hiện tại)
    current_streak = 0
    if len(history_str) > 0:
        last = history_str[-1]
        current_streak = 1
        for i in range(len(history_str) - 2, -1, -1):
            if history_str[i] == last: current_streak += 1
            else: break
    
    # Feature 2: Previous streak length (Độ dài chuỗi trước đó)
    previous_streak_len = 0
    if len(history_str) > current_streak:
        prev_streak_start_idx = len(history_str) - current_streak -1
        if prev_streak_start_idx >= 0:
            prev_streak_val = history_str[prev_streak_start_idx]
            previous_streak_len = 1
            for i in range(prev_streak_start_idx -1, -1, -1):
                if history_str[i] == prev_streak_val: previous_streak_len += 1
                else: break

    # Feature 3 & 4: Balance (Tài-Xỉu) short-term and long-term
    recent_history = history_str[-20:] # 20 kết quả gần nhất
    balance_short = (recent_history.count('Tài') - recent_history.count('Xỉu')) / max(1, len(recent_history))
    
    long_history = history_str[-100:] # 100 kết quả gần nhất
    balance_long = (long_history.count('Tài') - long_history.count('Xỉu')) / max(1, len(long_history))
    
    # Feature 5: Volatility (tần suất thay đổi)
    changes = sum(1 for i in range(len(recent_history)-1) if recent_history[i] != recent_history[i+1])
    volatility = changes / max(1, len(recent_history) -1) if len(recent_history) > 1 else 0.0

    # Feature 6: Alternation count in last 10 results (Số lần xen kẽ trong 10 kết quả cuối)
    last_10 = history_str[-10:]
    alternations = sum(1 for i in range(len(last_10) - 1) if last_10[i] != last_10[i+1])
    
    return [float(current_streak), float(previous_streak_len), balance_short, balance_long, volatility, float(alternations)]

def apply_meta_logic(prediction, confidence, history_str, suggested_pattern_name):
    """
    Áp dụng logic cấp cao để điều chỉnh dự đoán cuối cùng.
    Ví dụ: Logic "bẻ cầu" khi cầu quá dài.
    """
    final_prediction, final_confidence, reason = prediction, confidence, suggested_pattern_name

    # Logic 1: Bẻ cầu khi cầu bệt quá dài (Anti-Streak)
    streak_len = 0
    if len(history_str) > 0:
        last = history_str[-1]
        for x in reversed(history_str):
            if x == last: streak_len += 1
            else: break
    
    # Điều chỉnh độ tin cậy bẻ cầu linh hoạt hơn
    if streak_len >= 7: # Bắt đầu cân nhắc bẻ cầu từ 7
        if prediction == history_str[-1]: # Nếu dự đoán vẫn là theo cầu
            # Tính độ tin cậy bẻ cầu theo hàm sigmoid để nó tăng dần
            # Hàm này sẽ cho độ tin cậy bẻ cầu tăng nhanh khi streak dài
            # Ví dụ: streak 7: 50%, streak 8: ~62%, streak 9: ~73%, streak 10: ~82%
            confidence_break = 50 + (50 / (1 + math.exp(-(streak_len - 7) * 0.7))) # Tăng từ 50 lên gần 100 khi streak dài
            
            # Chỉ bẻ cầu nếu độ tin cậy của việc bẻ cầu cao hơn độ tin cậy ban đầu của dự đoán theo pattern
            if confidence_break > confidence:
                final_prediction = 'Xỉu' if history_str[-1] == 'Tài' else 'Tài'
                final_confidence = min(99.0, confidence_break) # Giới hạn 99%
                reason = f"META-BẺ CẦU: Cầu bệt siêu dài ({streak_len} {history_str[-1]}), độ tin cậy bẻ cầu cao."
                logging.warning(f"META-LOGIC: Activated Anti-Streak. Streak of {streak_len} detected. Forcing prediction to {final_prediction}.")
                return final_prediction, final_confidence, reason
            else:
                reason = f"{suggested_pattern_name} (Cầu bệt dài {streak_len}, nhưng độ tin cậy bẻ cầu chưa đủ cao để ghi đè)."
    elif streak_len >= 5: # Giảm nhẹ độ tin cậy nếu cầu khá dài nhưng chưa quá
        final_confidence = max(50.0, confidence - (streak_len - 4) * 2) # Giảm 2% cho mỗi kết quả thêm sau 4
        reason = f"{suggested_pattern_name} (Cầu bệt dài {streak_len}, độ tin cậy giảm nhẹ)."
        
    # Logic 2: Cân bằng lại nếu Tài/Xỉu quá lệch trong lịch sử dài hạn
    long_history_str = history_str[-100:]
    count_tai = long_history_str.count('Tài')
    count_xiu = long_history_str.count('Xỉu')

    if len(long_history_str) > 50: # Chỉ áp dụng nếu có đủ dữ liệu
        tai_ratio = count_tai / len(long_history_str)
        xiu_ratio = count_xiu / len(long_history_str)

        if tai_ratio > 0.65 and prediction == 'Tài': # Quá nhiều Tài, có thể sẽ về Xỉu
            # Chỉ ghi đè nếu độ tin cậy hiện tại không quá cao
            if final_confidence < 85.0: # Không ghi đè nếu pattern quá mạnh
                final_prediction = 'Xỉu'
                final_confidence = max(final_confidence, 75.0) # Tăng độ tin cậy nếu dự đoán này hợp lý
                reason = "META-CÂN BẰNG: Lịch sử Tài quá cao (>65%), có thể về Xỉu để cân bằng."
                logging.info(f"META-LOGIC: Tài ratio {tai_ratio*100:.1f}%. Forcing prediction to Xỉu.")
                return final_prediction, final_confidence, reason
        elif xiu_ratio > 0.65 and prediction == 'Xỉu': # Quá nhiều Xỉu, có thể về Tài
            if final_confidence < 85.0: # Không ghi đè nếu pattern quá mạnh
                final_prediction = 'Tài'
                final_confidence = max(final_confidence, 75.0)
                reason = "META-CÂN BẰNG: Lịch sử Xỉu quá cao (>65%), có thể về Tài để cân bằng."
                logging.info(f"META-LOGIC: Xỉu ratio {xiu_ratio*100:.1f}%. Forcing prediction to Tài.")
                return final_prediction, final_confidence, reason
    
    # Logic 3: Chống đảo quá dài (Anti-Alternating)
    # Nếu có chuỗi đảo quá dài (ví dụ T-X-T-X-T-X 7-8 lần), dự đoán sẽ bẻ cầu thành bệt
    alt_len = 0
    if len(history_str) >= 2:
        for i in range(len(history_str) - 1, 0, -1):
            if history_str[i] != history_str[i-1]:
                alt_len += 1
            else:
                break
    if alt_len >= 7: # Ví dụ: T-X-T-X-T-X-T (7 lần đảo)
        if prediction == ('Xỉu' if history_str[-1]=='Tài' else 'Tài'): # Nếu dự đoán vẫn là đảo
            if final_confidence < 80.0: # Nếu độ tin cậy không quá cao
                final_prediction = history_str[-1] # Dự đoán theo bệt
                final_confidence = max(final_confidence, 70.0) # Tăng độ tin cậy
                reason = f"META-CHỐNG ĐẢO: Cầu đảo quá dài ({alt_len} lần), dự đoán bẻ thành bệt."
                logging.warning(f"META-LOGIC: Activated Anti-Alternating. Alternation of {alt_len} detected. Forcing prediction to {final_prediction}.")
                return final_prediction, final_confidence, reason

    return final_prediction, final_confidence, reason


def predict_advanced(app, history_str):
    """Hàm điều phối dự đoán nâng cao, kết hợp nhiều mô hình với trọng số động."""
    if len(history_str) < 1: # Cần ít nhất 1 kết quả để biết phiên hiện tại
        return "Chờ dữ liệu", "Đang tải dữ liệu...", 0.0, {}, {}

    # Initial features (để dùng trong trường hợp chưa đủ lịch sử cho LR)
    initial_features = get_logistic_features(history_str)

    last_result = history_str[-1] if history_str else None

    # --- Model 1: Pattern Matching ---
    detected_pattern_info = detect_pattern(app, history_str)
    # pattern_reason sẽ giải thích lý do cụ thể của pattern
    patt_pred, patt_conf, pattern_reason = predict_with_pattern(app, history_str, detected_pattern_info)
    
    # --- Model 2: Markov Chain ---
    markov_pred = 'Tài'
    markov_conf = 50.0
    if last_result:
        last_result_idx = 0 if last_result == 'Tài' else 1
        prob_tai_markov = app.transition_matrix[last_result_idx][0]
        markov_pred = 'Tài' if prob_tai_markov > 0.5 else 'Xỉu'
        markov_conf = max(prob_tai_markov, 1 - prob_tai_markov) * 100

    # --- Model 3: Logistic Regression ---
    logistic_pred = 'Tài'
    logistic_conf = 50.0
    # Chỉ chạy Logistic Regression nếu có đủ lịch sử (ví dụ 10 phiên trở lên)
    if len(history_str) >= 10:
        features = get_logistic_features(history_str)
        # Phải chuẩn hóa features trước khi dùng với Logistic Regression
        normalized_features = _normalize_features(features)
        
        z = app.logistic_bias + sum(w * f for w, f in zip(app.logistic_weights, normalized_features))
        prob_tai_logistic = 0.0
        try:
            prob_tai_logistic = 1.0 / (1.0 + math.exp(-z))
        except OverflowError:
            prob_tai_logistic = 0.0 if z < 0 else 1.0
            
        logistic_pred = 'Tài' if prob_tai_logistic > 0.5 else 'Xỉu'
        logistic_conf = max(prob_tai_logistic, 1 - prob_tai_logistic) * 100
    else:
        features = initial_features # Dùng features ban đầu để lưu lại cho LR học sau này
        
    # Lưu lại dự đoán của từng mô hình để học
    individual_predictions = {
        'pattern': patt_pred,
        'markov': markov_pred,
        'logistic': logistic_pred
    }

    # --- Ensemble Prediction (Kết hợp các mô hình với trọng số động) ---
    predictions = {
        'pattern': {'pred': patt_pred, 'conf': patt_conf / 100, 'weight': app.model_weights['pattern']}, # Chia 100 để đưa conf về 0-1
        'markov': {'pred': markov_pred, 'conf': markov_conf / 100, 'weight': app.model_weights['markov']},
        'logistic': {'pred': logistic_pred, 'conf': logistic_conf / 100, 'weight': app.model_weights['logistic']},
    }
    
    tai_score, xiu_score = 0.0, 0.0
    total_effective_weight = 0.0 # Tổng trọng số thực tế của các mô hình có độ tin cậy trên 50%
    
    for model_name, model_data in predictions.items():
        # Chỉ tính điểm cho các mô hình có độ tin cậy cao hơn 50%
        if model_data['conf'] > 0.5:
            score = (model_data['conf'] - 0.5) * model_data['weight'] # Chỉ xét phần vượt quá 50%
            if model_data['pred'] == 'Tài': tai_score += score
            else: xiu_score += score
            total_effective_weight += model_data['weight'] # Cộng trọng số của mô hình này vào tổng
            
    final_prediction = 'Tài' # Mặc định Tài nếu không có mô hình nào tự tin
    final_confidence_raw = 0.0

    if total_effective_weight > 0:
        if tai_score > xiu_score:
            final_prediction = 'Tài'
            final_confidence_raw = (tai_score / total_effective_weight) * 100 # Tỷ lệ điểm Tài / tổng điểm
        elif xiu_score > tai_score:
            final_prediction = 'Xỉu'
            final_confidence_raw = (xiu_score / total_effective_weight) * 100 # Tỷ lệ điểm Xỉu / tổng điểm
        else: # Điểm bằng nhau, chọn ngẫu nhiên
            final_prediction = random.choice(['Tài', 'Xỉu'])
            final_confidence_raw = 50.0
    else: # Không có mô hình nào tự tin trên 50%, hoặc tất cả đều 50%
        # Fallback về một dự đoán ngẫu nhiên với độ tin cậy thấp
        final_prediction = random.choice(['Tài', 'Xỉu'])
        final_confidence_raw = 50.0

    # Chuyển đổi confidence_raw sang thang 50-100 để hiển thị dễ hiểu hơn
    # Nếu final_confidence_raw = 0.1 (tức 10%), sẽ được map về 50%
    # Nếu final_confidence_raw = 1.0 (tức 100%), sẽ được map về 100%
    final_confidence = 50 + (final_confidence_raw / 2) # Example mapping, can be adjusted
    final_confidence = min(99.9, max(50.0, final_confidence)) # Giới hạn trong 50-99.9

    # Tăng độ tin cậy nếu pattern mạnh nhất trùng với dự đoán cuối cùng VÀ pattern đó có độ tin cậy cao
    if detected_pattern_info and detected_pattern_info['weight'] > 0.6 and patt_pred == final_prediction:
        final_confidence = min(98.0, final_confidence + (patt_conf * 0.1)) # Tăng thêm 10% conf của pattern vào tổng

    # Áp dụng logic meta cuối cùng
    # pattern_reason (tên pattern + thống kê) sẽ là lý do được truyền vào ban đầu
    final_prediction, final_confidence, meta_reason = apply_meta_logic(final_prediction, final_confidence, history_str, pattern_reason)

    # Nếu meta_reason không có, sử dụng pattern_reason hoặc mặc định
    final_suggested_pattern = meta_reason if meta_reason else (pattern_reason if detected_pattern_info else "Phân tích Ensemble")

    # Thêm thông tin chi tiết về pattern và balance
    pattern_details = {
        "detected_pattern_name": detected_pattern_info['name'] if detected_pattern_info else "N/A",
        "pattern_weight": round(detected_pattern_info['weight'] * 100, 1) if detected_pattern_info else 0.0,
        "pattern_stats": app.pattern_outcome_stats.get(detected_pattern_info['name'], {}) if detected_pattern_info else {},
        "current_streak_length": int(features[0]),
        "previous_streak_length": int(features[1]),
        "balance_short_term_percent": round(features[2]*100,1),
        "balance_long_term_percent": round(features[3]*100,1),
        "volatility_score": round(features[4], 2),
        "alternation_count": int(features[5])
    }

    return final_prediction, final_suggested_pattern, final_confidence, individual_predictions, pattern_details

# --- END: Improved Logic ---

# --- Persistent Storage (Lưu/Tải trạng thái mô hình) ---
MODEL_STATE_FILE = 'model_state.json'

def save_model_state(app):
    state = {
        'history': list(app.history),
        'session_ids': list(app.session_ids),
        'transition_matrix': app.transition_matrix,
        'transition_counts': app.transition_counts,
        'logistic_weights': app.logistic_weights,
        'logistic_bias': app.logistic_bias,
        'model_weights': app.model_weights,
        'model_performance': dict(app.model_performance), # Convert defaultdict to dict
        'pattern_accuracy': dict(app.pattern_accuracy),
        'pattern_outcome_stats': dict(app.pattern_outcome_stats),
        'last_prediction': app.last_prediction
    }
    try:
        with open(MODEL_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4)
        logging.info("Model state saved successfully.")
    except Exception as e:
        logging.error(f"Error saving model state: {e}")

def load_model_state(app):
    if os.path.exists(MODEL_STATE_FILE):
        try:
            with open(MODEL_STATE_FILE, 'r') as f:
                state = json.load(f)
            app.history = deque(state.get('history', []), maxlen=app.MAX_HISTORY_LEN)
            app.session_ids = deque(state.get('session_ids', []), maxlen=app.MAX_HISTORY_LEN)
            app.transition_matrix = state.get('transition_matrix', [[0.5, 0.5], [0.5, 0.5]])
            app.transition_counts = state.get('transition_counts', [[0, 0], [0, 0]])
            app.logistic_weights = state.get('logistic_weights', [0.0] * 6)
            app.logistic_bias = state.get('logistic_bias', 0.0)
            app.model_weights = state.get('model_weights', app.default_model_weights.copy())
            
            # Convert back to defaultdict
            app.model_performance = defaultdict(lambda: {"success": 0, "total": 0}, state.get('model_performance', {}))
            app.pattern_accuracy = defaultdict(lambda: {"success": 0, "total": 0}, state.get('pattern_accuracy', {}))
            app.pattern_outcome_stats = defaultdict(lambda: {'Tài': 0, 'Xỉu': 0, 'total': 0}, state.get('pattern_outcome_stats', {}))
            app.last_prediction = state.get('last_prediction', None)
            logging.info("Model state loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model state: {e}. Starting with default state.")
            # Reset to default if loading fails
            app.history = deque(maxlen=app.MAX_HISTORY_LEN)
            app.session_ids = deque(maxlen=app.MAX_HISTORY_LEN)
            app.transition_matrix = [[0.5, 0.5], [0.5, 0.5]]
            app.transition_counts = [[0, 0], [0, 0]]
            app.logistic_weights = [0.0] * 6
            app.logistic_bias = 0.0
            app.model_weights = app.default_model_weights.copy()
            app.model_performance = {name: {"success": 0, "total": 0} for name in app.model_weights}
            app.pattern_accuracy = defaultdict(lambda: {"success": 0, "total": 0})
            app.pattern_outcome_stats = defaultdict(lambda: {'Tài': 0, 'Xỉu': 0, 'total': 0})
            app.last_prediction = None
    else:
        logging.info("No saved model state found. Starting with default state.")


# --- Flask App Factory ---
def create_app():
    app = Flask(__name__)
    CORS(app)

    # --- Khởi tạo State ---
    app.lock = threading.Lock()
    app.MAX_HISTORY_LEN = 1000 # Tăng độ dài lịch sử
    
    app.history = deque(maxlen=app.MAX_HISTORY_LEN)
    app.session_ids = deque(maxlen=app.MAX_HISTORY_LEN)
    
    # State cho các thuật toán
    app.patterns = define_patterns()
    app.transition_matrix = [[0.5, 0.5], [0.5, 0.5]] # Tài -> Tài, Tài -> Xỉu; Xỉu -> Tài, Xỉu -> Xỉu
    app.transition_counts = [[0, 0], [0, 0]]
    app.logistic_weights = [0.0] * 6 # Mở rộng cho 6 features
    app.logistic_bias = 0.0
    app.learning_rate = 0.005 # Tinh chỉnh tốc độ học
    app.regularization = 0.005 # Tinh chỉnh hệ số chính quy hóa
    
    # State cho ensemble model động
    app.default_model_weights = {'pattern': 0.4, 'markov': 0.3, 'logistic': 0.3} # Tùy chỉnh trọng số mặc định
    app.model_weights = app.default_model_weights.copy()
    app.model_performance = defaultdict(lambda: {"success": 0, "total": 0}, {name: {"success": 0, "total": 0} for name in app.model_weights})

    app.last_prediction = None
    app.pattern_accuracy = defaultdict(lambda: {"success": 0, "total": 0})
    # Thêm state để lưu thống kê kết quả tiếp theo cho mỗi pattern
    app.pattern_outcome_stats = defaultdict(lambda: {'Tài': 0, 'Xỉu': 0, 'total': 0})

    # Tải trạng thái mô hình khi khởi động
    load_model_state(app)
    
    # --- Xử lý API HTTP ---
    app.API_URL = os.getenv("API_URL", "https://sunwinwanglin.up.railway.app/api/sunwin?key=WangLinID99")

    def fetch_data_from_api():
        try:
            response = requests.get(app.API_URL, timeout=8) # Tăng timeout
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data from API: {e}")
            return None
        except json.JSONDecodeError as e:
            # Ghi log response content nếu có lỗi JSON để debug
            logging.error(f"Error decoding JSON from API: {e}. Response content: {response.text if 'response' in locals() else 'No response'}")
            return None


    def data_fetch_loop():
        last_processed_session = None
        while True:
            data = fetch_data_from_api()
            if data:
                try:
                    phien_truoc = data.get("phien_truoc")
                    ket_qua = data.get("ket_qua")

                    if phien_truoc is None or ket_qua not in ["Tài", "Xỉu"]:
                        logging.warning(f"Invalid data received from API: {data}")
                        time.sleep(2) # Wait a bit before retrying
                        continue

                    # phien_hien_tai là phiên kết quả vừa nhận được
                    phien_hien_tai = phien_truoc

                    with app.lock:
                        # Kiểm tra xem phiên này đã được xử lý chưa
                        if not app.session_ids or phien_hien_tai > app.session_ids[-1]:
                            # Chỉ thêm nếu là phiên mới nhất
                            app.session_ids.append(phien_hien_tai)
                            app.history.append({'ket_qua': ket_qua, 'phien': phien_hien_tai})
                            logging.info(f"New result for session {phien_hien_tai}: {ket_qua}. History length: {len(app.history)}")
                            last_processed_session = phien_hien_tai
                            
                            # Lưu trạng thái sau mỗi khi có dữ liệu mới quan trọng
                            save_model_state(app)

                        else:
                            # Nếu phiên đã có hoặc cũ hơn, không làm gì
                            if phien_hien_tai == app.session_ids[-1]:
                                logging.debug(f"Session {phien_hien_tai} already processed. Waiting for next.")
                            else:
                                logging.debug(f"Received older session {phien_hien_tai}. Current latest is {app.session_ids[-1]}.")
                except (json.JSONDecodeError, TypeError) as e:
                    logging.error(f"Error decoding or processing API data: {e} - Data: {data}")
                except Exception as e:
                    logging.error(f"Unexpected error in data_fetch_loop: {e}")
            
            time.sleep(1) # Poll the API every 1 second

    # --- API Endpoints ---
    @app.route("/api/taixiu_ws", methods=["GET"])
    def get_taixiu_prediction():
        with app.lock:
            # Lấy bản sao của lịch sử và các biến state để xử lý mà không bị khóa quá lâu
            history_copy = list(app.history)
            session_ids_copy = list(app.session_ids)
            last_prediction_data = app.last_prediction # Lấy bản sao của last_prediction

        # Lấy kết quả thực tế của phiên cuối cùng để học
        actual_result = None
        if history_copy:
            actual_result = history_copy[-1]['ket_qua']
        
        # --- Bước học Online (Online Learning) ---
        # Chỉ học nếu có dự đoán trước đó và phiên của dự đoán đó khớp với phiên vừa có kết quả
        # Đảm bảo có đủ lịch sử để lấy features_at_prediction_time chính xác (history_copy[:-1])
        if last_prediction_data and actual_result and session_ids_copy and \
           last_prediction_data['session'] == session_ids_copy[-1] and \
           last_prediction_data['features'] is not None: # Đảm bảo features đã được lưu
            
            # Lịch sử tại thời điểm dự đoán (trừ đi kết quả cuối cùng vừa nhận)
            history_at_prediction_time_str = _get_history_strings(history_copy[:-1])

            with app.lock: # Khóa lại khi cập nhật state của các mô hình
                # Học cho Logistic Regression (sử dụng features tại thời điểm dự đoán)
                train_logistic_regression(app, last_prediction_data['features'], actual_result)
                
                # Học cho Markov Chain
                if len(history_at_prediction_time_str) > 0:
                     update_transition_matrix(app, history_at_prediction_time_str[-1], actual_result)
                
                # Học cho Pattern (phải truyền lịch sử tại thời điểm phát hiện pattern)
                # pattern_at_prediction_time là tên pattern đã được phát hiện ở lần dự đoán trước
                update_pattern_accuracy(app, last_prediction_data['pattern_name'], actual_result, history_at_prediction_time_str)
                
                # Cập nhật hiệu suất của từng mô hình con để điều chỉnh trọng số ensemble
                for model_name, model_pred in last_prediction_data['individual_predictions'].items():
                    app.model_performance[model_name]['total'] += 1
                    if model_pred == actual_result:
                        app.model_performance[model_name]['success'] += 1
                
                # Cập nhật lại trọng số của ensemble model
                update_model_weights(app)

            logging.info(f"Learned from session {session_ids_copy[-1]}. Prediction was {last_prediction_data['prediction']}, actual was {actual_result}. Pattern: {last_prediction_data['pattern_name']}")
            
        # --- Bước Dự đoán (Prediction) cho phiên TIẾP THEO ---
        if len(history_copy) < 1: # Ít nhất phải có 1 kết quả để biết phiên hiện tại
             return jsonify({
                "error": "Chưa có đủ dữ liệu lịch sử để dự đoán.",
                "current_session": None,
                "current_result": None,
                "next_session": None,
                "prediction": "Chờ dữ liệu",
                "confidence_percent": 0.0,
                "suggested_pattern": "Đang tải dữ liệu...",
            }), 200 # Trả về 200 vì đây không phải lỗi server mà là thiếu dữ liệu

        history_str_for_prediction = _get_history_strings(history_copy)
        
        # current_session là phiên cuối cùng đã có kết quả
        current_session = session_ids_copy[-1] if session_ids_copy else None 
        # next_session là phiên sẽ dự đoán
        next_session = current_session + 1 if current_session is not None else None

        # Thực hiện dự đoán
        # Đảm bảo rằng hàm predict_advanced luôn trả về 5 giá trị
        # và dòng này nhận đúng 5 giá trị.
        prediction_str, suggested_pattern, confidence, individual_preds, pattern_details = \
            predict_advanced(app, history_str_for_prediction)
        
        # Lưu lại thông tin dự đoán để học ở lần tiếp theo
        with app.lock:
            # Lưu features hiện tại để dùng cho huấn luyện LR ở bước tiếp theo
            # Đây là features của history_str_for_prediction (lịch sử ĐẦY ĐỦ tại thời điểm dự đoán)
            current_features_for_lr = get_logistic_features(history_str_for_prediction)
            
            app.last_prediction = {
                'session': next_session, # Đây là phiên mà dự đoán này áp dụng
                'prediction': prediction_str,
                'pattern_name': pattern_details['detected_pattern_name'], # Lưu tên pattern chính để học
                'features': current_features_for_lr, # LƯU CÁC FEATURES HIỆN TẠI ĐỂ HUẤN LUYỆN
                'individual_predictions': individual_preds, # Dự đoán của từng mô hình
            }
        
        # Tinh chỉnh hiển thị độ tin cậy và lý do
        final_confidence_display = round(confidence, 1)

        # Quyết định hiển thị "Đang phân tích"
        display_prediction = prediction_str
        display_suggested_pattern = suggested_pattern
        if confidence < 60.0: # Ngưỡng để hiển thị "Đang phân tích"
            display_prediction = "Đang phân tích"
            display_suggested_pattern = "Chưa có cầu rõ rệt / Đang cân nhắc"
            final_confidence_display = round(confidence, 1) # Vẫn hiển thị % tin cậy thấp

        return jsonify({
            "current_session": current_session,
            "current_result": actual_result, # Kết quả của phiên hiện tại
            "next_session_id": next_session, # Phiên mà dự đoán này áp dụng
            "prediction": display_prediction, # Dự đoán cuối cùng
            "confidence_percent": final_confidence_display, # Độ tin cậy
            "suggested_pattern": display_suggested_pattern, # Lý do / Pattern gợi ý
            "detail_info": { # Thông tin chi tiết hơn giống ảnh
                "pattern_info": {
                    "name": pattern_details['detected_pattern_name'],
                    "weight_percent": pattern_details['pattern_weight'],
                    "outcome_stats": pattern_details['pattern_stats']
                },
                "current_streak_length": pattern_details['current_streak_length'],
                "previous_streak_length": pattern_details['previous_streak_length'],
                "balance_short_term_percent": pattern_details['balance_short_term_percent'],
                "balance_long_term_percent": pattern_details['balance_long_term_percent'],
                "volatility_score": pattern_details['volatility_score'],
                "alternation_count": pattern_details['alternation_count'],
                "individual_model_predictions": {k: f"{v} ({round(app.model_performance[k]['success']/max(1, app.model_performance[k]['total'])*100, 1)}% acc)" 
                                                 for k,v in individual_preds.items()}, # Thêm độ chính xác của từng model
                "ensemble_model_weights": {k: round(v, 3) for k, v in app.model_weights.items()} # Thêm trọng số ensemble
            }
        })

    @app.route("/api/history", methods=["GET"])
    def get_history_api():
        with app.lock:
            hist_copy = list(app.history)
            # Tạo danh sách các chuỗi "Tài" hoặc "Xỉu" cho lịch sử gần nhất
            recent_history_strings = _get_history_strings(hist_copy[-50:]) # Lấy 50 kết quả gần nhất để hiển thị
        return jsonify({"history": hist_copy, "recent_history_string": recent_history_strings, "length": len(hist_copy)})

    @app.route("/api/performance", methods=["GET"])
    def get_performance():
        with app.lock:
            # Sắp xếp pattern theo tổng số lần xuất hiện và độ chính xác
            seen_patterns = {k: v for k, v in app.pattern_accuracy.items() if v['total'] > 0}
            sorted_patterns = sorted(
                seen_patterns.items(), 
                key=lambda item: (item[1]['total'], (item[1]['success'] / item[1]['total'] if item[1]['total'] > 0 else 0)),
                reverse=True
            )
            pattern_result = {}
            for p_type, data in sorted_patterns[:50]: # Lấy 50 pattern hàng đầu
                accuracy = round(data["success"] / data["total"] * 100, 2) if data["total"] > 0 else 0
                outcome_stats = app.pattern_outcome_stats.get(p_type, {'Tài': 0, 'Xỉu': 0, 'total': 0})
                
                # Tránh chia cho 0 nếu total_next_outcomes là 0
                prob_tai_next = round(outcome_stats['Tài'] / outcome_stats['total'] * 100, 1) if outcome_stats['total'] > 0 else 0.0
                prob_xiu_next = round(outcome_stats['Xỉu'] / outcome_stats['total'] * 100, 1) if outcome_stats['total'] > 0 else 0.0

                pattern_result[p_type] = { 
                    "total_matches": data["total"], 
                    "successful_predictions": data["success"], 
                    "accuracy_percent": accuracy,
                    "next_outcome_stats": {
                        "Tài": outcome_stats['Tài'],
                        "Xỉu": outcome_stats['Xỉu'],
                        "total_next_outcomes": outcome_stats['total'],
                        "prob_tai_next_percent": prob_tai_next,
                        "prob_xiu_next_percent": prob_xiu_next
                    }
                }
            
            # Lấy hiệu suất của các mô hình con
            model_perf_result = {}
            for name, perf in app.model_performance.items():
                 accuracy = round(perf["success"] / perf["total"] * 100, 2) if perf["total"] > 0 else 0
                 model_perf_result[name] = {**perf, "accuracy_percent": accuracy}

            # Thông tin thêm về Logistic Regression
            logistic_info = {
                "weights": [round(w, 4) for w in app.logistic_weights],
                "bias": round(app.logistic_bias, 4),
                "learning_rate": app.learning_rate,
                "regularization": app.regularization
            }

            # Thông tin Markov Chain
            markov_info = {
                "transition_matrix": [[round(val, 4) for val in row] for row in app.transition_matrix],
                "transition_counts": app.transition_counts
            }


        return jsonify({
            "pattern_performance": pattern_result,
            "model_performance": model_perf_result,
            "ensemble_model_weights": app.model_weights,
            "logistic_regression_details": logistic_info,
            "markov_chain_details": markov_info,
            "total_history_length": len(app.history)
        })

    # Start the HTTP data fetching thread
    api_fetch_thread = threading.Thread(target=data_fetch_loop, daemon=True)
    api_fetch_thread.start()
    return app

# --- Thực thi chính ---
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logging.info(f"Flask app ready. Serving on http://0.0.0.0:{port}")
    # Sử dụng Waitress cho môi trường production
    serve(app, host="0.0.0.0", port=port, threads=10)
