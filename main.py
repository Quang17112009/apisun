import logging
import json
import os
import time
from collections import deque
from flask import Flask, jsonify, request
import requests
import numpy as np
from datetime import datetime, timedelta
import threading # Đảm bảo threading được import

# --- Cấu hình logging ---
# Đặt cấu hình logging ở đầu file để nó hoạt động ngay lập tức
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Khởi tạo ứng dụng Flask ---
app = Flask(__name__)

# --- Cấu hình toàn cục ---
app.MAX_HISTORY_LEN = 1000 # Giới hạn độ dài lịch sử
app.API_URL = "https://sunwinwanglin.up.railway.app/api/sunwin?key=WangLinID99"
app.FETCH_INTERVAL = 30 # Thời gian chờ giữa các lần fetch dữ liệu (giây)

# Tên file lưu trạng thái mô hình.
# Nếu bạn đã cấu hình Render Disks, hãy thay đổi thành '/mnt/data/model_state.json'
# Ví dụ: app.MODEL_STATE_FILE = '/mnt/data/model_state.json'
app.MODEL_STATE_FILE = 'model_state.json' 

app.lock = threading.Lock() # Khóa để đảm bảo an toàn luồng khi cập nhật trạng thái

# --- Khởi tạo các biến trạng thái mô hình ---
# Các biến này sẽ được gán lại khi tải trạng thái, nhưng cần có giá trị khởi tạo
app.history = deque(maxlen=app.MAX_HISTORY_LEN)
app.session_ids = deque(maxlen=app.MAX_HISTORY_LEN) 
app.last_fetched_session = None

# Trạng thái học máy
app.logistic_weights = np.zeros(6) # 6 đặc trưng
app.logistic_bias = 0.0
app.learning_rate = 0.005
app.regularization = 0.005 # L2 regularization

app.markov_transition_counts = np.zeros((2, 2)) # [[TT, TX], [XT, XX]]
app.markov_transition_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])

app.pattern_outcome_stats = {} # {pattern_name: {'Tài': count, 'Xỉu': count, 'total_next_outcomes': count}}
app.pattern_performance = {} # {pattern_name: {'successful_predictions': count, 'total_matches': count}}

app.model_performance = { # Để đánh giá và điều chỉnh trọng số ensemble
    'logistic': {'success': 0, 'total': 0},
    'markov': {'success': 0, 'total': 0},
    'pattern': {'success': 0, 'total': 0}
}
app.ensemble_model_weights = {'logistic': 1/3, 'markov': 1/3, 'pattern': 1/3}

app.last_prediction = None # Lưu dự đoán cuối cùng để học từ kết quả thực tế

# --- Hàm hỗ trợ ---
def _get_history_strings(history_data):
    return [item['ket_qua'] for item in history_data if 'ket_qua' in item]

def _map_result_to_int(result):
    return 1 if result == 'Tài' else 0

def _map_int_to_result(val):
    return 'Tài' if val == 1 else 'Xỉu'

def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500))) # Clip để tránh overflow

# --- Hàm tính đặc trưng cho Logistic Regression ---
def get_logistic_features(history_str):
    if len(history_str) < 100: # Cần ít nhất 100 phiên để tính đủ các features
        return None

    # Lấy các phần lịch sử cần thiết
    last_10 = history_str[-10:]
    last_20 = history_str[-20:]
    last_50 = history_str[-50:]
    last_100 = history_str[-100:]

    # 1. Chiều dài chuỗi hiện tại (streak length)
    current_result = history_str[-1]
    streak_length = 0
    for res in reversed(history_str):
        if res == current_result:
            streak_length += 1
        else:
            break
    
    # 2. Tỷ lệ Tài/Xỉu trong ngắn hạn (last 10)
    tai_count_10 = last_10.count('Tài')
    xiu_count_10 = last_10.count('Xỉu')
    balance_short_term_percent = (tai_count_10 - xiu_count_10) / len(last_10) # Từ -1 (toàn Xỉu) đến 1 (toàn Tài)

    # 3. Tỷ lệ Tài/Xỉu trong dài hạn (last 50)
    tai_count_50 = last_50.count('Tài')
    xiu_count_50 = last_50.count('Xỉu')
    balance_long_term_percent = (tai_count_50 - xiu_count_50) / len(last_50)

    # 4. Độ biến động (volatility) - số lần thay đổi kết quả trong 20 phiên gần nhất
    volatility_score = 0
    for i in range(1, len(last_20)):
        if last_20[i] != last_20[i-1]:
            volatility_score += 1
    volatility_score /= len(last_20) # Chuẩn hóa về 0-1

    # 5. Số lần xen kẽ (alternation) - số lần đổi Tài Xỉu liên tục trong 10 phiên
    alternation_count = 0
    for i in range(1, len(last_10)):
        if last_10[i] != last_10[i-1]:
            alternation_count += 1
    alternation_count /= len(last_10) # Chuẩn hóa về 0-1

    # 6. Tỷ lệ xuất hiện của kết quả hiện tại trong 100 phiên gần nhất
    current_result_count_100 = last_100.count(current_result)
    current_result_ratio_100 = current_result_count_100 / len(last_100)

    features = [
        streak_length / 10.0, # Chuẩn hóa streak_length (giả sử max streak là 10, có thể cần điều chỉnh)
        balance_short_term_percent,
        balance_long_term_percent,
        volatility_score,
        alternation_count,
        current_result_ratio_100
    ]
    return features


# --- Các hàm học máy ---
def train_logistic_regression(app, features, actual_result):
    if features is None:
        logging.warning("Features are None for LR training. Skipping.")
        return

    x = np.array(features)
    y = _map_result_to_int(actual_result)

    # Tính lại xác suất dự đoán P tại thời điểm huấn luyện (sử dụng trọng số HIỆN TẠI của mô hình)
    p = _sigmoid(np.dot(x, app.logistic_weights) + app.logistic_bias)

    error = y - p

    # Cập nhật trọng số và bias
    app.logistic_weights += app.learning_rate * error * x - app.learning_rate * app.regularization * app.logistic_weights
    app.logistic_bias += app.learning_rate * error

    logging.debug(f"LR Trained: Actual={actual_result}, PredictedProb={p:.2f}, Error={error:.2f}")
    logging.debug(f"New LR Weights: {app.logistic_weights.round(4)}, Bias: {app.logistic_bias:.4f}")


def update_transition_matrix(app, prev_result, current_result):
    if prev_result not in ['Tài', 'Xỉu'] or current_result not in ['Tài', 'Xỉu']:
        logging.warning(f"Invalid results for Markov update: prev={prev_result}, curr={current_result}")
        return

    prev_idx = _map_result_to_int(prev_result)
    curr_idx = _map_result_to_int(current_result)

    app.markov_transition_counts[prev_idx, curr_idx] += 1

    # Cập nhật ma trận chuyển đổi
    for i in range(2):
        total_transitions_from_i = np.sum(app.markov_transition_counts[i, :])
        if total_transitions_from_i > 0:
            app.markov_transition_matrix[i, :] = app.markov_transition_counts[i, :] / total_transitions_from_i
        else:
            app.markov_transition_matrix[i, :] = [0.5, 0.5] # Nếu chưa có dữ liệu, giữ 50/50

    logging.debug(f"Markov Updated: Prev={prev_result}, Curr={current_result}")
    logging.debug(f"New Markov Matrix: {app.markov_transition_matrix.round(4)}")


def update_pattern_accuracy(app, pattern_name, actual_result):
    if pattern_name is None or pattern_name == "Không có pattern rõ ràng":
        return

    if pattern_name not in app.pattern_outcome_stats:
        app.pattern_outcome_stats[pattern_name] = {'Tài': 0, 'Xỉu': 0, 'total_next_outcomes': 0}
    
    if actual_result == 'Tài':
        app.pattern_outcome_stats[pattern_name]['Tài'] += 1
    else:
        app.pattern_outcome_stats[pattern_name]['Xỉu'] += 1
    app.pattern_outcome_stats[pattern_name]['total_next_outcomes'] += 1

    # Cập nhật performance
    if pattern_name not in app.pattern_performance:
        app.pattern_performance[pattern_name] = {'successful_predictions': 0, 'total_matches': 0}

    # Chỉ đánh giá khi pattern đã xuất hiện đủ số lần để có ý nghĩa thống kê
    if app.pattern_outcome_stats[pattern_name]['total_next_outcomes'] >= 5: 
        # Giả định rằng dự đoán của pattern là kết quả có xác suất cao hơn
        pattern_stats = app.pattern_outcome_stats[pattern_name]
        
        # Nếu tổng Tài + Xỉu là 0, không thể chia. Kiểm tra lại logic này.
        if pattern_stats['total_next_outcomes'] > 0:
            prob_tai = pattern_stats['Tài'] / pattern_stats['total_next_outcomes']
            prob_xiu = pattern_stats['Xỉu'] / pattern_stats['total_next_outcomes']
            
            predicted_by_pattern = None
            if prob_tai > prob_xiu:
                predicted_by_pattern = 'Tài'
            elif prob_xiu > prob_tai:
                predicted_by_pattern = 'Xỉu'

            if predicted_by_pattern is not None:
                app.pattern_performance[pattern_name]['total_matches'] += 1 # Tăng total_matches mỗi khi có dữ liệu để đánh giá
                if predicted_by_pattern == actual_result:
                    app.pattern_performance[pattern_name]['successful_predictions'] += 1

    logging.debug(f"Pattern '{pattern_name}' Updated: Actual={actual_result}")
    # logging.debug(f"New Pattern Stats: {app.pattern_outcome_stats[pattern_name]}") # Quá nhiều log


def update_model_weights(app):
    total_performance_score = 0
    model_scores = {}

    for model_name, perf in app.model_performance.items():
        if perf['total'] >= 10: # Chỉ xem xét mô hình khi có ít nhất 10 lần dự đoán để đánh giá
            accuracy = perf['success'] / perf['total']
            # Đánh giá cao hơn các mô hình có độ chính xác tốt hơn 50%
            model_scores[model_name] = max(0.0, accuracy - 0.5) # Chỉ lấy phần vượt quá 50%
        else:
            model_scores[model_name] = 0.0 # Chưa đủ dữ liệu để đánh giá

    total_score_sum = sum(model_scores.values())

    if total_score_sum > 0:
        for model_name, score in model_scores.items():
            app.ensemble_model_weights[model_name] = score / total_score_sum
    else:
        # Nếu không có mô hình nào có độ chính xác trên 50% hoặc chưa đủ dữ liệu, phân bổ đều
        app.ensemble_model_weights = {'logistic': 1/3, 'markov': 1/3, 'pattern': 1/3}

    logging.info(f"Ensemble Weights Updated: {app.ensemble_model_weights}")


# --- Hàm dự đoán chính ---
@app.route('/api/taixiu_ws')
def get_taixiu_prediction_route():
    # Sử dụng khóa để đảm bảo chỉ có một luồng xử lý dự đoán và học tại một thời điểm
    with app.lock:
        return jsonify(get_taixiu_prediction_logic())

def get_taixiu_prediction_logic():
    history_copy_full = list(app.history) # Bản sao đầy đủ của lịch sử
    history_strings_full = _get_history_strings(history_copy_full)

    # Lấy kết quả thực tế của phiên cuối cùng để học
    actual_result_for_learning = None
    session_id_for_learning = None
    if history_copy_full:
        actual_result_for_learning = history_copy_full[0].get('ket_qua') # Lịch sử API mới nhất ở đầu deque
        session_id_for_learning = history_copy_full[0].get('phien')

    # --- Bước học Online (Online Learning) ---
    # Học DỰA TRÊN KẾT QUẢ THỰC TẾ của phiên vừa xong và DỰ ĐOÁN ĐÃ ĐƯA RA cho phiên đó.
    # app.last_prediction chứa thông tin dự đoán của phiên TRƯỚC (phiên mà giờ ta biết kết quả)
    
    logging.info(f"DEBUG Learning Check: app.last_prediction exists: {app.last_prediction is not None}")
    if app.last_prediction:
        logging.info(f"DEBUG Last Prediction Session: {app.last_prediction.get('session', 'N/A')}")
        logging.info(f"DEBUG Last Prediction Features set: {app.last_prediction.get('features') is not None}")
    
    logging.info(f"DEBUG Actual Result for Learning: {actual_result_for_learning}")
    logging.info(f"DEBUG Session ID for Learning: {session_id_for_learning}")

    # Điều kiện để học:
    # 1. Có dữ liệu của lần dự đoán trước đó (app.last_prediction không phải None)
    # 2. Có kết quả thực tế cho phiên đó
    # 3. ID phiên của kết quả thực tế khớp với ID phiên mà ta đã dự đoán trước đó
    if app.last_prediction and actual_result_for_learning and \
       app.last_prediction.get('session') == session_id_for_learning:
        
        logging.info("DEBUG Learning condition MET. Proceeding with learning.")

        # Lịch sử TẠI THỜI ĐIỂM DỰ ĐOÁN cho phiên vừa học (tức là lịch sử hiện tại bỏ đi phiên mới nhất vừa fetch)
        # Vì history_copy_full có dữ liệu mới nhất ở đầu (index 0), thì lịch sử tại thời điểm dự đoán là history_copy_full[1:]
        history_at_prediction_time_str = _get_history_strings(list(history_copy_full)[1:]) 

        # Học cho Logistic Regression (sử dụng features tại thời điểm dự đoán)
        train_logistic_regression(app, app.last_prediction.get('features'), actual_result_for_learning)
        
        # Học cho Markov Chain
        # Markov cần kết quả của 2 phiên liền kề (phiên trước và phiên vừa học)
        if len(history_strings_full) >= 2:
             prev_result_for_markov = history_strings_full[1] # Kết quả phiên trước phiên vừa học
             update_transition_matrix(app, prev_result_for_markov, actual_result_for_learning)
        
        # Học cho Pattern
        update_pattern_accuracy(app, app.last_prediction.get('pattern_name'), actual_result_for_learning)
        
        # Cập nhật hiệu suất của từng mô hình con để điều chỉnh trọng số ensemble
        # Chỉ cập nhật total nếu predict_advanced đã thực sự đưa ra dự đoán (tức là individual_predictions có dữ liệu)
        if app.last_prediction.get('individual_predictions'):
            for model_name, model_pred in app.last_prediction['individual_predictions'].items():
                app.model_performance[model_name]['total'] += 1
                if model_pred == actual_result_for_learning:
                    app.model_performance[model_name]['success'] += 1
        else:
             logging.warning("individual_predictions was not available in last_prediction for learning.")
        
        # Cập nhật lại trọng số của ensemble model
        update_model_weights(app)

        logging.info(f"Learned from session {session_id_for_learning}. Prediction was {app.last_prediction.get('prediction', 'N/A')}, actual was {actual_result_for_learning}. Pattern: {app.last_prediction.get('pattern_name', 'N/A')}")
        
        # Sau khi học xong, xóa last_prediction để nó không bị học lại cho cùng một phiên
        # Nó sẽ được gán lại khi có dự đoán cho phiên tiếp theo
        app.last_prediction = None 
        logging.info("app.last_prediction reset to None after learning.")

    else:
        logging.warning("DEBUG Learning condition NOT MET. Skipping learning.")
        if not app.last_prediction:
            logging.warning("  Reason: app.last_prediction is None.")
        elif not actual_result_for_learning:
            logging.warning("  Reason: actual_result_for_learning is None.")
        elif app.last_prediction and session_id_for_learning: 
            # Đảm bảo app.last_prediction.get('session') không bị lỗi nếu key không tồn tại
            logging.warning(f"  Reason: Session ID mismatch: last_pred_session={app.last_prediction.get('session', 'N/A')}, actual_session={session_id_for_learning}")


    # --- Bước Dự đoán (Prediction) cho phiên TIẾP THEO (phien_hien_tai + 1) ---
    # Lịch sử hiện tại để dự đoán phiên tiếp theo là toàn bộ history_strings_full
    if len(history_strings_full) < 100: # Cần ít nhất 100 phiên để có features LR
        return {"prediction": "Đang phân tích", "confidence": 50.0, "reason": "Chưa đủ lịch sử để phân tích."}

    current_history_str_for_prediction = history_strings_full # Lịch sử đầy đủ nhất để dự đoán

    # --- 1. Dự đoán bằng Logistic Regression ---
    logistic_features = get_logistic_features(current_history_str_for_prediction)
    logistic_prediction_score = 0.5
    logistic_prediction = "Đang phân tích"
    if logistic_features is not None:
        logistic_score = np.dot(np.array(logistic_features), app.logistic_weights) + app.logistic_bias
        logistic_prediction_score = _sigmoid(logistic_score)
        logistic_prediction = "Tài" if logistic_prediction_score >= 0.5 else "Xỉu"
    logging.debug(f"Logistic Prediction: {logistic_prediction} ({logistic_prediction_score:.2f})")

    # --- 2. Dự đoán bằng Markov Chain ---
    markov_prediction_score = 0.5
    markov_prediction = "Đang phân tích"
    if len(current_history_str_for_prediction) > 0:
        last_result_idx = _map_result_to_int(current_history_str_for_prediction[0]) # Lấy kết quả mới nhất
        tai_prob = app.markov_transition_matrix[last_result_idx, _map_result_to_int('Tài')]
        xiu_prob = app.markov_transition_matrix[last_result_idx, _map_result_to_int('Xỉu')]
        
        if tai_prob > xiu_prob:
            markov_prediction = "Tài"
            markov_prediction_score = tai_prob
        elif xiu_prob > tai_prob:
            markov_prediction = "Xỉu"
            markov_prediction_score = xiu_prob
        else:
            markov_prediction = "Đang phân tích" # 50/50
            markov_prediction_score = 0.5
    logging.debug(f"Markov Prediction: {markov_prediction} ({markov_prediction_score:.2f})")

    # --- 3. Dự đoán bằng Pattern Matching ---
    best_pattern_name = "Không có pattern rõ ràng"
    pattern_prediction_score = 0.5
    pattern_prediction = "Đang phân tích"
    
    # Các pattern cơ bản
    patterns = {
        "Bệt Tài": ["Tài", "Tài", "Tài"],
        "Bệt Xỉu": ["Xỉu", "Xỉu", "Xỉu"],
        "Đảo 1-1": ["Tài", "Xỉu", "Tài"],
        "Lặp 2-1": ["Tài", "Tài", "Xỉu"],
        "Lặp 1-2": ["Xỉu", "Xỉu", "Tài"],
        "Tài kép": ["Tài", "Xỉu", "Xỉu", "Tài"], # Ví dụ: Tài Xỉu Xỉu Tài
        "Xỉu kép": ["Xỉu", "Tài", "Tài", "Xỉu"], # Ví dụ: Xỉu Tài Tài Xỉu
        "Bệt gãy nhẹ": ["Tài", "Tài", "Tài", "Xỉu", "Tài"], # Bệt 3 Tài, gãy Xỉu, lại về Tài
        "Xen kẽ": ["Tài", "Tài", "Xỉu", "Xỉu"] # 2 Tài 2 Xỉu
    }

    # Tìm pattern dài nhất và có độ tin cậy cao nhất
    max_len = 0
    best_pattern_accuracy = 0.0
    
    for name, pattern_seq_rev in patterns.items():
        pattern_seq = list(reversed(pattern_seq_rev)) # Reverse pattern để so sánh với history_strings_full[0:len(pattern_seq)]

        if len(current_history_str_for_prediction) >= len(pattern_seq) and \
           list(current_history_str_for_prediction[0:len(pattern_seq)]) == pattern_seq:
            
            if name in app.pattern_outcome_stats and \
               app.pattern_outcome_stats[name]['total_next_outcomes'] >= 5: # Chỉ tin cậy pattern có đủ dữ liệu
                
                stats = app.pattern_outcome_stats[name]
                if stats['total_next_outcomes'] > 0:
                    prob_tai = stats['Tài'] / stats['total_next_outcomes']
                    prob_xiu = stats['Xỉu'] / stats['total_next_outcomes']

                    current_accuracy = max(prob_tai, prob_xiu)
                    
                    # Ưu tiên pattern dài hơn và có độ chính xác cao hơn
                    if len(pattern_seq) > max_len or \
                       (len(pattern_seq) == max_len and current_accuracy > best_pattern_accuracy):
                        max_len = len(pattern_seq)
                        best_pattern_name = name
                        best_pattern_accuracy = current_accuracy
                        pattern_prediction = 'Tài' if prob_tai > prob_xiu else 'Xỉu'
                        pattern_prediction_score = max(prob_tai, prob_xiu)
                
            else:
                # Nếu không đủ dữ liệu cho pattern, vẫn có thể ghi nhận tên pattern nhưng không dùng để dự đoán chính
                if len(pattern_seq) > max_len: # Vẫn ưu tiên pattern dài nhất nếu chưa có dữ liệu
                     max_len = len(pattern_seq)
                     best_pattern_name = name # Ghi nhận tên pattern dù chưa đủ data
    
    logging.debug(f"Pattern Prediction: {pattern_prediction} ({pattern_prediction_score:.2f}) from {best_pattern_name}")

    # --- 4. Ensemble Model (Kết hợp các mô hình) ---
    individual_predictions = {
        'logistic': logistic_prediction,
        'markov': markov_prediction,
        'pattern': pattern_prediction
    }
    
    # Tính điểm cho Tài và Xỉu từ mỗi mô hình
    tai_scores = {}
    xiu_scores = {}

    if logistic_prediction != "Đang phân tích":
        if logistic_prediction == "Tài":
            tai_scores['logistic'] = logistic_prediction_score
            xiu_scores['logistic'] = 1 - logistic_prediction_score
        else:
            tai_scores['logistic'] = 1 - logistic_prediction_score
            xiu_scores['logistic'] = logistic_prediction_score
    else: # Nếu LR đang phân tích, coi như 50/50
        tai_scores['logistic'] = 0.5
        xiu_scores['logistic'] = 0.5

    if markov_prediction != "Đang phân tích":
        if markov_prediction == "Tài":
            tai_scores['markov'] = markov_prediction_score
            xiu_scores['markov'] = 1 - markov_prediction_score
        else:
            tai_scores['markov'] = 1 - markov_prediction_score
            xiu_scores['markov'] = markov_prediction_score
    else: # Nếu Markov đang phân tích, coi như 50/50
        tai_scores['markov'] = 0.5
        xiu_scores['markov'] = 0.5

    if pattern_prediction != "Đang phân tích":
        if pattern_prediction == "Tài":
            tai_scores['pattern'] = pattern_prediction_score
            xiu_scores['pattern'] = 1 - pattern_prediction_score
        else:
            tai_scores['pattern'] = 1 - pattern_prediction_score
            xiu_scores['pattern'] = pattern_prediction_score
    else: # Nếu Pattern đang phân tích, coi như 50/50
        tai_scores['pattern'] = 0.5
        xiu_scores['pattern'] = 0.5

    # Tính điểm tổng hợp có trọng số
    tai_score = sum(tai_scores[m] * app.ensemble_model_weights[m] for m in app.ensemble_model_weights)
    xiu_score = sum(xiu_scores[m] * app.ensemble_model_weights[m] for m in app.ensemble_model_weights)

    final_prediction = "Đang phân tích"
    final_confidence_raw = 0.0

    if tai_score > xiu_score:
        final_prediction = "Tài"
        final_confidence_raw = tai_score
    elif xiu_score > tai_score:
        final_prediction = "Xỉu"
        final_confidence_raw = xiu_score
    else:
        final_prediction = "Đang phân tích"
        final_confidence_raw = 0.5 # Nếu bằng nhau, độ tin cậy 50%

    # Chuyển đổi điểm raw thành phần trăm độ tin cậy (0-100)
    # Scale từ 0.5-1.0 lên 50-100
    final_confidence = 50.0 + (final_confidence_raw - 0.5) * 100.0

    # --- Logic Meta (Anti-Streak / Balance) ---
    # Nếu bạn muốn thêm logic meta, hãy đặt ở đây.
    # Ví dụ: nếu độ tin cậy cao nhưng lịch sử hiện tại có chuỗi rất dài, có thể giảm tin cậy hoặc chuyển sang "Đang phân tích"
    
    # === LƯU THÔNG TIN DỰ ĐOÁN HIỆN TẠI VÀO app.last_prediction ===
    # Đây là bước cực kỳ quan trọng để chuẩn bị cho quá trình học ở phiên TIẾP THEO
    # session_id_for_prediction sẽ là ID của phiên MÀ TA ĐANG DỰ ĐOÁN
    # nó sẽ là session_id_for_learning + 1
    session_id_for_prediction = session_id_for_learning + 1 if session_id_for_learning is not None else None
    
    if session_id_for_prediction is not None:
        app.last_prediction = {
            'session': session_id_for_prediction,
            'prediction': final_prediction,
            'confidence': final_confidence,
            'pattern_name': best_pattern_name,
            'features': logistic_features, # Rất quan trọng cho LR
            'individual_predictions': individual_predictions # Rất quan trọng cho cập nhật hiệu suất mô hình con
        }
        logging.info(f"Saved last_prediction for session {session_id_for_prediction}")
    else:
        logging.warning("Could not set app.last_prediction: session_id_for_prediction is None.")

    # --- Trả về kết quả ---
    if final_confidence < 60.0: # Ngưỡng để hiển thị "Đang phân tích"
        return {"prediction": "Đang phân tích", "confidence": final_confidence, "reason": "Độ tin cậy thấp."}
    else:
        return {"prediction": final_prediction, "confidence": final_confidence, "reason": "Dự đoán dựa trên mô hình."}


# --- Các route API ---
# Route này gọi hàm logic dự đoán
@app.route('/api/taixiu_prediction') # Đổi tên route để không nhầm với websocket
def taixiu_prediction_endpoint():
    return get_taixiu_prediction_route() # Gọi hàm route bên trên

@app.route('/api/history')
def get_history_api():
    with app.lock:
        # Trả về lịch sử theo thứ tự cũ nhất đến mới nhất cho dễ đọc
        # Mặc định deque app.history có mới nhất ở index 0, cũ nhất ở cuối
        return jsonify(list(app.history)[::-1]) 

@app.route('/api/performance')
def get_performance_api():
    with app.lock:
        # Cập nhật accuracy_percent cho pattern_performance
        for pattern_name, perf_data in app.pattern_performance.items():
            if perf_data['total_matches'] > 0:
                perf_data['accuracy_percent'] = round((perf_data['successful_predictions'] / perf_data['total_matches']) * 100, 2)
            else:
                perf_data['accuracy_percent'] = 0.0
        
        # Cập nhật accuracy_percent cho model_performance
        for model_name, perf_data in app.model_performance.items():
            if perf_data['total'] > 0:
                perf_data['accuracy_percent'] = round((perf_data['success'] / perf_data['total']) * 100, 2)
            else:
                perf_data['accuracy_percent'] = 0.0

        return jsonify({
            "total_history_length": len(app.history),
            "model_performance": app.model_performance,
            "ensemble_model_weights": app.ensemble_model_weights,
            "logistic_regression_details": {
                "weights": app.logistic_weights.tolist(),
                "bias": app.logistic_bias,
                "learning_rate": app.learning_rate,
                "regularization": app.regularization
            },
            "markov_chain_details": {
                "transition_counts": app.markov_transition_counts.tolist(),
                "transition_matrix": app.markov_transition_matrix.tolist()
            },
            "pattern_performance": app.pattern_performance,
            "current_last_prediction_state": app.last_prediction # Thêm trạng thái last_prediction vào đây để debug
        })

# --- Hàm fetch dữ liệu và vòng lặp ---
def fetch_data_and_update_history():
    while True:
        try:
            response = requests.get(app.API_URL)
            response.raise_for_status() # Ném lỗi cho các mã trạng thái HTTP xấu
            data = response.json()

            if data and isinstance(data, list) and data[0].get('phien') is not None:
                latest_session = data[0]
                session_id = latest_session['phien']

                with app.lock:
                    if session_id != app.last_fetched_session:
                        app.history.appendleft(latest_session) # Thêm vào đầu deque (phiên mới nhất)
                        app.session_ids.appendleft(session_id)
                        app.last_fetched_session = session_id
                        logging.info(f"Fetched new session: {session_id}, Result: {latest_session.get('ket_qua')}. Current History length: {len(app.history)}")
                        
                        # Lưu trạng thái sau mỗi khi có dữ liệu mới quan trọng
                        save_model_state(app)
                    else:
                        logging.debug(f"Session {session_id} already fetched. Waiting for new data.")
            else:
                logging.warning("API returned empty or invalid data from external source.")

        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data: {e}")
        except json.JSONDecodeError:
            logging.error("Error decoding JSON from API response.")
        except Exception as e:
            logging.error(f"An unexpected error occurred in fetch_data_and_update_history: {e}")

        time.sleep(app.FETCH_INTERVAL)

# --- Hàm lưu/tải trạng thái mô hình ---
def save_model_state(app):
    state = {
        'history': list(app.history),
        'session_ids': list(app.session_ids),
        'last_fetched_session': app.last_fetched_session,
        'logistic_weights': app.logistic_weights.tolist(),
        'logistic_bias': app.logistic_bias,
        'markov_transition_counts': app.markov_transition_counts.tolist(),
        'markov_transition_matrix': app.markov_transition_matrix.tolist(),
        'pattern_outcome_stats': app.pattern_outcome_stats,
        'pattern_performance': app.pattern_performance,
        'model_performance': app.model_performance,
        'ensemble_model_weights': app.ensemble_model_weights,
        'last_prediction': app.last_prediction # Lưu cả last_prediction
    }
    try:
        with open(app.MODEL_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4) # indent=4 để dễ đọc file JSON
        logging.info(f"Model state saved successfully to {app.MODEL_STATE_FILE}.")
    except Exception as e:
        logging.error(f"Error saving model state to {app.MODEL_STATE_FILE}: {e}")

def load_model_state(app):
    loaded_from_model_state_file = False
    
    # 1. Ưu tiên tải từ model_state.json (nếu có)
    if os.path.exists(app.MODEL_STATE_FILE):
        try:
            with open(app.MODEL_STATE_FILE, 'r') as f:
                state = json.load(f)
            
            app.history = deque(state.get('history', []), maxlen=app.MAX_HISTORY_LEN)
            app.session_ids = deque(state.get('session_ids', []), maxlen=app.MAX_HISTORY_LEN)
            app.last_fetched_session = state.get('last_fetched_session')
            app.logistic_weights = np.array(state.get('logistic_weights', np.zeros(6).tolist()))
            app.logistic_bias = state.get('logistic_bias', 0.0)
            app.markov_transition_counts = np.array(state.get('markov_transition_counts', np.zeros((2, 2)).tolist()))
            app.markov_transition_matrix = np.array(state.get('markov_transition_matrix', np.array([[0.5, 0.5], [0.5, 0.5]]).tolist()))
            app.pattern_outcome_stats = state.get('pattern_outcome_stats', {})
            app.pattern_performance = state.get('pattern_performance', {})
            app.model_performance = state.get('model_performance', {'logistic': {'success': 0, 'total': 0}, 'markov': {'success': 0, 'total': 0}, 'pattern': {'success': 0, 'total': 0}})
            app.ensemble_model_weights = state.get('ensemble_model_weights', {'logistic': 1/3, 'markov': 1/3, 'pattern': 1/3})
            app.last_prediction = state.get('last_prediction')
            
            logging.info(f"Model state loaded successfully from {app.MODEL_STATE_FILE}.")
            loaded_from_model_state_file = True
        except Exception as e:
            logging.error(f"Error loading model state from {app.MODEL_STATE_FILE}: {e}. Will attempt to load from lichsu.txt.")
            # Nếu model_state.json bị lỗi, reset các biến về mặc định trước khi thử lichsu.txt
            reset_model_state_defaults(app) 

    # 2. Nếu không tải được từ model_state.json, thử tải từ lichsu.txt
    if not loaded_from_model_state_file:
        if os.path.exists('lichsu.txt'): # Giả định lichsu.txt nằm trong cùng thư mục
            try:
                with open('lichsu.txt', 'r') as f:
                    initial_history_data = json.load(f)
                
                # Sắp xếp lịch sử theo ID phiên tăng dần để thêm vào deque một cách đúng thứ tự
                # (để dữ liệu cũ nhất ở bên phải (sẽ là head của deque nếu appendleft dữ liệu mới sau này))
                # API fetch mới nhất ở index 0, cũ nhất ở cuối.
                # Khi nạp từ file, ta muốn file này bổ sung các phiên cũ hơn nếu cần.
                # Do đó, sắp xếp theo ID phiên tăng dần rồi append.
                initial_history_data.sort(key=lambda x: x.get('phien', 0))

                # Thêm vào app.history. Deque sẽ tự động cắt bớt nếu vượt quá maxlen.
                for session_data in initial_history_data:
                    # Kiểm tra trùng lặp với app.session_ids để tránh thêm lại các phiên đã có
                    if session_data.get('phien') not in app.session_ids:
                        app.history.append(session_data) # Thêm vào bên phải (phía cũ hơn của lịch sử)
                        app.session_ids.append(session_data.get('phien')) # Thêm ID phiên

                # Cập nhật app.last_fetched_session dựa trên phiên mới nhất từ lichsu.txt
                # Quan trọng: app.last_fetched_session phải là ID của phiên mới nhất ĐÃ CÓ trong history.
                # Khi luồng fetch chạy, nó sẽ tìm phiên > app.last_fetched_session
                if app.history:
                    # Vì lichsu.txt đã sắp xếp tăng dần, phần tử cuối cùng là mới nhất
                    app.last_fetched_session = app.history[-1]['phien'] 

                logging.info(f"Initial history loaded successfully from lichsu.txt. Current history length: {len(app.history)}")
                
            except Exception as e:
                logging.error(f"Error loading initial history from lichsu.txt: {e}. Starting with default state.")
                reset_model_state_defaults(app) # Nếu lichsu.txt cũng lỗi, reset hoàn toàn
        else:
            logging.info("No saved model state found and no lichsu.txt. Starting with default state.")
            reset_model_state_defaults(app)

def reset_model_state_defaults(app): # Đổi tên hàm để rõ ràng hơn
    app.history = deque(maxlen=app.MAX_HISTORY_LEN)
    app.session_ids = deque(maxlen=app.MAX_HISTORY_LEN)
    app.last_fetched_session = None
    app.logistic_weights = np.zeros(6)
    app.logistic_bias = 0.0
    app.markov_transition_counts = np.zeros((2, 2))
    app.markov_transition_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
    app.pattern_outcome_stats = {}
    app.pattern_performance = {}
    app.model_performance = {'logistic': {'success': 0, 'total': 0}, 'markov': {'success': 0, 'total': 0}, 'pattern': {'success': 0, 'total': 0}}
    app.ensemble_model_weights = {'logistic': 1/3, 'markov': 1/3, 'pattern': 1/3}
    app.last_prediction = None
    logging.info("Model state reset to default.")

# --- Hàm tạo ứng dụng (cho Gunicorn/WSGI) ---
def create_app():
    load_model_state(app) # Tải trạng thái khi ứng dụng khởi động

    # Khởi chạy luồng fetch dữ liệu
    fetch_thread = threading.Thread(target=fetch_data_and_update_history, daemon=True)
    fetch_thread.start()
    
    return app

# Nếu chạy trực tiếp (ví dụ: python app.py)
if __name__ == '__main__':
    # Đảm bảo logging được cấu hình trước khi tạo app
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    my_app = create_app()
    my_app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)

