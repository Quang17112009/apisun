from flask import Flask, jsonify, request
import requests
import os
import collections
import copy
import random # Dùng cho trường hợp xác suất bằng nhau

app = Flask(__name__)

# --- Cấu hình API bên ngoài ---
# Đường dẫn API đã được thay thế theo yêu cầu của bạn
EXTERNAL_API_URL = "https://taixiu-qbwo.onrender.com/api/axocuto" 

# --- Mẫu dữ liệu ban đầu và trạng thái toàn cục ---
# Đây là trạng thái mà API sẽ trả về, được cập nhật sau mỗi phiên.
# LƯU Ý QUAN TRỌNG: Lịch sử và trạng thái này KHÔNG BỀN VỮNG
# và sẽ bị reset khi ứng dụng khởi động lại. Đối với môi trường sản xuất,
# bạn NÊN sử dụng cơ sở dữ liệu.
initial_api_data_template = {
    "Phien_moi": None, # Sẽ được cập nhật từ new_session_data['Expect']
    "pattern_length": 8,
    "pattern": "xxxxxxxx",
    "matches": ["x"],
    "pattern_tai": 0,
    "pattern_xiu": 0,
    "pattern_percent_tai": 0,
    "pattern_percent_xiu": 0,
    "phan_tram_tai": 50,
    "phan_tram_xiu": 50,
    "tong_tai": 0.0,
    "tong_xiu": 0.0,
    "du_doan": "Không có",
    "ly_do": "Chưa có dữ liệu dự đoán.",
    "phien_du_doan": None, # Sẽ được tính toán từ Expect của phiên hiện tại
    "admin_info": "@heheviptool",
    "Tong_phien_du_doan": 0,   # NEW: Tổng số lần dự đoán đã được ghi nhận
    "So_lan_du_doan_sai": 0    # NEW: Số lần dự đoán sai
}

# Lịch sử các kết quả thực tế (t='Tài', x='Xỉu')
# Kích thước tối thiểu là pattern_length + số lượng phiên cần theo dõi cho logic.
# Tăng kích thước để phân tích mẫu và xác suất có điều kiện tốt hơn.
history_results = collections.deque(maxlen=100) # Ví dụ: 100 phiên gần nhất

# Lưu trữ trạng thái dự đoán gần nhất để kiểm tra 'consecutive_losses'
last_prediction_info = {
    "predicted_expect": None, # Expect code của phiên đã được dự đoán trong lần chạy trước
    "predicted_result": None, # "Tài" hoặc "Xỉu"
    "consecutive_losses": 0, # Số lần dự đoán sai liên tiếp
    "last_actual_result": None # Kết quả thực tế của phiên vừa rồi
}

# --- Hàm hỗ trợ ---
def calculate_tai_xiu(open_code_str):
    """
    Tính tổng xúc xắc và xác định Tài/Xỉu.
    Trả về ('Tài'/'Xỉu', tổng, [xúc xắc 1, xúc xắc 2, xúc xắc 3]) hoặc ('Lỗi', 0, [])
    """
    try:
        dice_values = [int(x.strip()) for x in open_code_str.split(',')]
        total_sum = sum(dice_values)

        if total_sum >= 4 and total_sum <= 10:
            return "Xỉu", total_sum, dice_values
        elif total_sum >= 11 and total_sum <= 17:
            return "Tài", total_sum, dice_values
        else: # Tổng 3 hoặc 18 (Bộ ba) - Gộp vào Tài/Xỉu để đơn giản logic pattern
            if total_sum == 3: return "Xỉu", total_sum, dice_values
            if total_sum == 18: return "Tài", total_sum, dice_values
            return "Không xác định", total_sum, dice_values
    except (ValueError, TypeError) as e:
        print(f"Error calculating Tai/Xiu from OpenCode '{open_code_str}': {e}")
        return "Lỗi", 0, []

def get_next_expect_code(current_expect_code):
    """
    Tính toán Expect code của phiên tiếp theo bằng cách tăng phần số cuối cùng.
    Giả định Expect code có dạng 'YYYYMMDDXXXX' với XXXX là số tăng dần 4 chữ số.
    """
    if len(current_expect_code) < 4 or not current_expect_code[-4:].isdigit():
        print(f"Warning: Expect code '{current_expect_code}' does not match expected format for incrementing.")
        return None # Hoặc xử lý lỗi tùy theo yêu cầu

    prefix = current_expect_code[:-4]
    suffix_str = current_expect_code[-4:]
    
    try:
        suffix_int = int(suffix_str)
        next_suffix_int = suffix_int + 1
        next_suffix_str = str(next_suffix_int).zfill(len(suffix_str)) # Đảm bảo giữ nguyên số chữ số
        return prefix + next_suffix_str
    except ValueError:
        print(f"Error: Could not convert suffix '{suffix_str}' to integer.")
        return None


def update_history_and_state(new_session_data):
    """
    Cập nhật lịch sử và trạng thái dự đoán toàn cục dựa trên dữ liệu phiên mới.
    """
    global history_results, initial_api_data_template, last_prediction_info

    # Lấy ID phiên và Expect code từ dữ liệu mới
    current_id = new_session_data['ID']
    current_expect_code = new_session_data['Expect'] # Đây là chuỗi "2507010521"
    current_open_code = new_session_data['OpenCode']
    actual_result_type, total_sum, dice_values = calculate_tai_xiu(current_open_code)
    actual_result_char = "t" if "Tài" in actual_result_type else "x"

    # Chỉ thêm vào lịch sử nếu đây là phiên mới (kiểm tra bằng ID)
    if not any(entry['ID'] == current_id for entry in history_results):
        history_results.append({
            "ID": current_id,
            "Expect": current_expect_code, # Lưu cả Expect code vào lịch sử
            "OpenCode": current_open_code,
            "Result": actual_result_char,
            "Ket_qua_text": actual_result_type, # Thêm kết quả dạng text
            "Tong": total_sum,                 # Thêm tổng xúc xắc
            "Xuc_xac_1": dice_values[0] if len(dice_values) > 0 else None,
            "Xuc_xac_2": dice_values[1] if len(dice_values) > 1 else None,
            "Xuc_xac_3": dice_values[2] if len(dice_values) > 2 else None,
        })
        print(f"Added new session to history: ID {current_id}, Expect {current_expect_code} - Result: {actual_result_type}")

        # --- Cập nhật Consecutive Losses và Prediction Accuracy ---
        # Logic này kiểm tra xem dự đoán của phiên TRƯỚC ĐÓ (predicted_expect)
        # có khớp với Expect code của phiên HIỆN TẠI mà ta vừa nhận được kết quả hay không.
        # last_prediction_info["predicted_expect"] lưu Expect code của phiên mà chúng ta đã dự đoán cho nó.
        # current_expect_code là Expect code của phiên mà chúng ta vừa nhận được kết quả thực tế.
        if last_prediction_info["predicted_expect"] is not None and \
           last_prediction_info["predicted_expect"] == current_expect_code and \
           last_prediction_info["predicted_result"] is not None:
            
            predicted_res = last_prediction_info["predicted_result"]
            
            # Tăng tổng số phiên dự đoán đã được ghi nhận
            initial_api_data_template["Tong_phien_du_doan"] += 1

            if predicted_res.lower() != actual_result_char:
                last_prediction_info["consecutive_losses"] += 1
                initial_api_data_template["So_lan_du_doan_sai"] += 1 # Tăng số lần dự đoán sai
                print(f"Prediction '{predicted_res}' for session Expect {current_expect_code} MISSED. Consecutive losses: {last_prediction_info['consecutive_losses']}, Total incorrect: {initial_api_data_template['So_lan_du_doan_sai']}")
            else:
                last_prediction_info["consecutive_losses"] = 0
                print(f"Prediction '{predicted_res}' for session Expect {current_expect_code} CORRECT. Resetting losses.")
        else:
            # Nếu không có dự đoán trước đó hoặc phiên không khớp (ví dụ: khởi động lại app), reset loss
            last_prediction_info["consecutive_losses"] = 0
            print("No matching previous prediction to evaluate or app restarted. Resetting losses.")
        
        last_prediction_info["last_actual_result"] = actual_result_char # Cập nhật kết quả thực tế mới nhất

    # Cập nhật các trường chính trong initial_api_data_template
    initial_api_data_template["Phien_moi"] = current_expect_code # Hiển thị Expect code làm "phiên mới"
    
    # Tính toán Phien_du_doan bằng cách tăng Expect code của phiên hiện tại
    next_expect_code = get_next_expect_code(current_expect_code)
    initial_api_data_template["phien_du_doan"] = next_expect_code if next_expect_code else "Không xác định"

    # --- Cập nhật pattern và pattern percentages ---
    current_pattern_chars = "".join([entry['Result'] for entry in history_results])
    initial_api_data_template['pattern'] = current_pattern_chars[-initial_api_data_template['pattern_length']:]
    
    tai_count = initial_api_data_template['pattern'].count('t')
    xiu_count = initial_api_data_template['pattern'].count('x')
    
    initial_api_data_template['pattern_tai'] = tai_count
    initial_api_data_template['pattern_xiu'] = xiu_count

    total_pattern_chars = len(initial_api_data_template['pattern'])
    if total_pattern_chars > 0:
        initial_api_data_template['pattern_percent_tai'] = round((tai_count / total_pattern_chars) * 100, 2)
        initial_api_data_template['pattern_percent_xiu'] = round((xiu_count / total_pattern_chars) * 100, 2)
    else:
        initial_api_data_template['pattern_percent_tai'] = 0
        initial_api_data_template['pattern_percent_xiu'] = 0

    # Cập nhật 'matches' (giả định là kết quả của phiên mới nhất)
    if history_results:
        initial_api_data_template['matches'] = [history_results[-1]['Result']]
    else:
        initial_api_data_template['matches'] = []

    # Giả định phan_tram_tai/xiu và tong_tai/xiu dựa trên pattern_percent
    # Trong môi trường thực, các giá trị này thường đến từ dữ liệu cược hoặc hệ thống riêng.
    initial_api_data_template['phan_tram_tai'] = initial_api_data_template['pattern_percent_tai']
    initial_api_data_template['phan_tram_xiu'] = initial_api_data_template['pattern_percent_xiu']
    
    # Giả định tổng tiền theo tỷ lệ phần trăm (chỉ để điền vào mẫu JSON)
    initial_api_data_template['tong_tai'] = round(initial_api_data_template['phan_tram_tai'] * 1000 / 100, 2)
    initial_api_data_template['tong_xiu'] = round(initial_api_data_template['phan_tram_xiu'] * 1000 / 100, 2)

# --- Logic Dự Đoán Thông Minh Hơn ---
def analyze_streaks(history_deque):
    """Phân tích các chuỗi (streaks) Tài/Xỉu trong lịch sử gần đây."""
    if not history_deque:
        return 0, None # current_streak_length, current_streak_type

    current_streak_length = 0
    current_streak_type = None

    # Đi ngược từ kết quả gần nhất để tìm chuỗi
    for i in range(len(history_deque) - 1, -1, -1):
        result = history_deque[i]['Result']
        if current_streak_type is None:
            current_streak_type = result
            current_streak_length = 1
        elif result == current_streak_type:
            current_streak_length += 1
        else:
            break # Chuỗi bị phá vỡ

    return current_streak_length, current_streak_type

def calculate_conditional_probability(history_deque, lookback_length=3):
    """
    Tính xác suất có điều kiện của 't' hoặc 'x' dựa trên 'lookback_length' kết quả trước đó.
    Trả về dict: { 'prefix': {'t': probability_of_next_is_t, 'x': probability_of_next_is_x} }
    """
    if len(history_deque) < lookback_length + 1:
        return {} # Không đủ dữ liệu

    probabilities = {}
    
    # Lấy chuỗi các ký tự kết quả
    results_chars = "".join([entry['Result'] for entry in history_deque])

    for i in range(len(results_chars) - lookback_length):
        prefix = results_chars[i : i + lookback_length]
        next_char = results_chars[i + lookback_length]

        if prefix not in probabilities:
            probabilities[prefix] = {'t': 0, 'x': 0, 'total': 0}
        
        probabilities[prefix][next_char] += 1
        probabilities[prefix]['total'] += 1
    
    # Chuyển đổi số đếm thành xác suất
    final_probs = {}
    for prefix, counts in probabilities.items():
        if counts['total'] > 0:
            final_probs[prefix] = {
                't': counts['t'] / counts['total'],
                'x': counts['x'] / counts['total']
            }
        else:
            final_probs[prefix] = {'t': 0, 'x': 0}

    return final_probs


def perform_prediction_logic():
    """
    Thực hiện logic dự đoán thông minh cho phiên tiếp theo và cập nhật 'du_doan', 'ly_do'.
    """
    global initial_api_data_template, last_prediction_info, history_results

    du_doan_ket_qua = "Không có" # Khởi tạo với giá trị mặc định
    ly_do_du_doan = "Chưa có dữ liệu dự đoán."

    # --- Tín hiệu 1: Phân tích cầu (Streaks) ---
    min_streak_for_prediction = 3 # Ví dụ: Dự đoán theo cầu nếu cầu >= 3
    break_streak_threshold = 5 # Ví dụ: Cân nhắc bẻ cầu nếu cầu >= 5 (có thể điều chỉnh)

    current_streak_length, current_streak_type = analyze_streaks(history_results)

    if current_streak_type: # Đảm bảo có cầu để phân tích
        if current_streak_length >= min_streak_for_prediction:
            if current_streak_length < break_streak_threshold:
                # Nếu cầu chưa quá dài, tiếp tục theo cầu
                if current_streak_type == 't':
                    du_doan_ket_qua = "Tài"
                    ly_do_du_doan = f"Theo cầu Tài dài ({current_streak_length} lần)."
                else:
                    du_doan_ket_qua = "Xỉu"
                    ly_do_du_doan = f"Theo cầu Xỉu dài ({current_streak_length} lần)."
            else:
                # Nếu cầu quá dài, cân nhắc bẻ cầu (dự đoán ngược lại)
                if current_streak_type == 't':
                    du_doan_ket_qua = "Xỉu"
                    ly_do_du_doan = f"Bẻ cầu Tài dài ({current_streak_length} lần) có khả năng đảo chiều."
                else:
                    du_doan_ket_qua = "Tài"
                    ly_do_du_doan = f"Bẻ cầu Xỉu dài ({current_streak_length} lần) có khả năng đảo chiều."
        else:
            ly_do_du_doan = "Không có cầu rõ ràng."
    else:
        ly_do_du_doan = "Chưa đủ dữ liệu để phân tích cầu."


    # --- Tín hiệu 2: Xác suất có điều kiện (Conditional Probability) ---
    # Ưu tiên xác suất có điều kiện nếu có đủ dữ liệu và tín hiệu mạnh hơn
    lookback_prob = 3 # Nhìn vào N phiên trước đó để tính xác suất (có thể điều chỉnh)
    
    if len(history_results) >= lookback_prob:
        recent_prefix_chars = "".join([entry['Result'] for entry in history_results])[-lookback_prob:]
        conditional_probs = calculate_conditional_probability(history_results, lookback_prob)

        if recent_prefix_chars in conditional_probs:
            prob_t = conditional_probs[recent_prefix_chars]['t']
            prob_x = conditional_probs[recent_prefix_chars]['x']

            # Yêu cầu xác suất đủ cao để ghi đè (ví dụ: >60%)
            prob_threshold_strong = 0.6
            
            if prob_t > prob_x and prob_t >= prob_threshold_strong:
                # Ghi đè dự đoán nếu xác suất có điều kiện mạnh hơn hoặc nếu chưa có dự đoán
                if not du_doan_ket_qua or \
                   (du_doan_ket_qua == "Xỉu" and prob_t > prob_x * 1.2): # Ghi đè Xỉu nếu Tài mạnh hơn đáng kể
                    du_doan_ket_qua = "Tài"
                    ly_do_du_doan = f"Xác suất Tài cao ({round(prob_t*100, 2)}%) sau {recent_prefix_chars}."
            elif prob_x > prob_t and prob_x >= prob_threshold_strong:
                if not du_doan_ket_qua or \
                   (du_doan_ket_qua == "Tài" and prob_x > prob_t * 1.2): # Ghi đè Tài nếu Xỉu mạnh hơn đáng kể
                    du_doan_ket_qua = "Xỉu"
                    ly_do_du_doan = f"Xác suất Xỉu cao ({round(prob_x*100, 2)}%) sau {recent_prefix_chars}."
        
    # --- Tín hiệu 3: Logic "Đang trật X lần → Auto đảo ngược" ---
    # Đây là cơ chế quản lý rủi ro cuối cùng, sẽ ghi đè các dự đoán khác nếu ngưỡng bị đạt.
    reverse_threshold = 3 # Ngưỡng đảo ngược (có thể điều chỉnh)
    if last_prediction_info["consecutive_losses"] >= reverse_threshold:
        original_prediction = du_doan_ket_qua # Lưu lại dự đoán gốc để ghi log
        if du_doan_ket_qua == "Tài":
            du_doan_ket_qua = "Xỉu"
        elif du_doan_ket_qua == "Xỉu":
            du_doan_ket_qua = "Tài"
        else: # Nếu chưa có dự đoán nào từ các logic trên, và đang trật, thì cứ đảo ngược theo kết quả gần nhất
            if last_prediction_info["last_actual_result"] == 't':
                du_doan_ket_qua = "Xỉu"
            else:
                du_doan_ket_qua = "Tài"

        ly_do_du_doan += f" | Đang trật {last_prediction_info['consecutive_losses']} lần → Auto đảo ngược."
        print(f"Applied reversal logic: Original prediction was '{original_prediction}', changed to '{du_doan_ket_qua}'.")
    
    # --- Tín hiệu cuối cùng nếu không có tín hiệu mạnh nào ---
    # Nếu sau tất cả các logic trên mà vẫn chưa có dự đoán chắc chắn
    if du_doan_ket_qua == "Không có" or du_doan_ket_qua == "": # Xử lý trường hợp chuỗi rỗng
        # Dùng tỷ lệ pattern chung (ít thông minh hơn nhưng là fallback)
        if initial_api_data_template['pattern_percent_tai'] > initial_api_data_template['pattern_percent_xiu']:
            du_doan_ket_qua = "Tài"
            ly_do_du_doan = "Mặc định: Theo tỷ lệ pattern Tài lớn hơn (không có tín hiệu mạnh khác)."
        elif initial_api_data_template['pattern_percent_xiu'] > initial_api_data_template['pattern_percent_tai']:
            du_doan_ket_qua = "Xỉu"
            ly_do_du_doan = "Mặc định: Theo tỷ lệ pattern Xỉu lớn hơn (không có tín hiệu mạnh khác)."
        else:
            # Nếu tất cả các tín hiệu đều cân bằng, dự đoán ngẫu nhiên để tránh bị động
            du_doan_ket_qua = random.choice(["Tài", "Xỉu"])
            ly_do_du_doan = "Mặc định: Các tín hiệu cân bằng, dự đoán ngẫu nhiên."


    initial_api_data_template['du_doan'] = du_doan_ket_qua
    initial_api_data_template['ly_do'] = ly_do_du_doan

    # Lưu dự đoán này để kiểm tra ở phiên tiếp theo
    last_prediction_info["predicted_expect"] = initial_api_data_template["phien_du_doan"]
    last_prediction_info["predicted_result"] = du_doan_ket_qua


@app.route('/')
def home():
    return "Chào mừng đến với API dự đoán Tài Xỉu trên Render! Truy cập /predict để xem dự đoán."

@app.route('/predict', methods=['GET'])
def get_prediction():
    """
    Endpoint chính để lấy dữ liệu mới nhất từ API bên ngoài, cập nhật trạng thái
    và trả về dự đoán cho phiên tiếp theo theo định dạng JSON mẫu.
    """
    global initial_api_data_template, last_prediction_info

    try:
        print(f"Calling external API: {EXTERNAL_API_URL}")
        response = requests.get(EXTERNAL_API_URL)
        response.raise_for_status() # Ném lỗi nếu HTTP request không thành công (4xx hoặc 5xx)
        external_data = response.json()
        print(f"Data received from external API: {external_data}")

        if external_data.get("state") == 1 and "data" in external_data:
            new_session_data = external_data["data"]

            update_history_and_state(new_session_data)
            perform_prediction_logic()

            # Trích xuất dữ liệu phiên hiện tại để định dạng lại response
            current_session_data = history_results[-1] if history_results else {}
            
            output_response = {
                "Ket_qua_du_doan": initial_api_data_template["du_doan"],
                "Phien_du_doan": initial_api_data_template["phien_du_doan"],
                "Ly_do": initial_api_data_template["ly_do"],
                "Thong_ke_du_doan": {
                    "Tong_phien_du_doan": initial_api_data_template["Tong_phien_du_doan"],
                    "So_lan_du_doan_sai": initial_api_data_template["So_lan_du_doan_sai"]
                },
                "Phien_hien_tai": {
                    "Phien": current_session_data.get("Expect"),
                    "Ket_qua": current_session_data.get("Ket_qua_text"),
                    "Tong": current_session_data.get("Tong"),
                    "Xuc_xac_1": current_session_data.get("Xuc_xac_1"),
                    "Xuc_xac_2": current_session_data.get("Xuc_xac_2"),
                    "Xuc_xac_3": current_session_data.get("Xuc_xac_3")
                },
                "admin_info": initial_api_data_template["admin_info"]
            }
            
            return jsonify(output_response), 200
        else:
            error_message = "Invalid data or 'state' is not 1 from external API."
            print(f"Error: {error_message} - Raw response: {external_data}")
            return jsonify({"error": error_message, "raw_response": external_data}), 500

    except requests.exceptions.RequestException as e:
        error_message = f"Error connecting to external API: {e}. Vui lòng kiểm tra URL và kết nối."
        print(f"Error: {error_message}")
        return jsonify({"error": error_message}), 500
    except Exception as e:
        error_message = f"Internal server error: {e}"
        print(f"Error: {error_message}")
        return jsonify({"error": error_message}), 500

@app.route('/status', methods=['GET'])
def get_current_status():
    """
    Endpoint để lấy trạng thái dự đoán hiện tại mà không gọi API bên ngoài.
    """
    current_session_data = history_results[-1] if history_results else {}
    output_response = {
        "Ket_qua_du_doan": initial_api_data_template["du_doan"],
        "Phien_du_doan": initial_api_data_template["phien_du_doan"],
        "Ly_do": initial_api_data_template["ly_do"],
        "Thong_ke_du_doan": {
            "Tong_phien_du_doan": initial_api_data_template["Tong_phien_du_doan"],
            "So_lan_du_doan_sai": initial_api_data_template["So_lan_du_doan_sai"]
        },
        "Phien_hien_tai": {
            "Phien": current_session_data.get("Expect"),
            "Ket_qua": current_session_data.get("Ket_qua_text"),
            "Tong": current_session_data.get("Tong"),
            "Xuc_xac_1": current_session_data.get("Xuc_xac_1"),
            "Xuc_xac_2": current_session_data.get("Xuc_xac_2"),
            "Xuc_xac_3": current_session_data.get("Xuc_xac_3")
        },
        "admin_info": initial_api_data_template["admin_info"]
    }
    return jsonify(output_response), 200

@app.route('/history', methods=['GET'])
def get_history():
    """
    Endpoint để xem lịch sử các phiên đã được xử lý (trong bộ nhớ).
    """
    return jsonify(list(history_results)), 200

@app.route('/last_prediction_info', methods=['GET'])
def get_last_prediction_info_route(): 
    """
    Endpoint để xem thông tin về dự đoán gần nhất và số lần trật liên tiếp.
    """
    return jsonify(last_prediction_info), 200

# --- Chạy ứng dụng Flask ---
if __name__ == '__main__':
    # Flask sẽ lấy PORT từ biến môi trường của Render
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
