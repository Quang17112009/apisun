import os
import random
import httpx
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.exc import IntegrityError

# --- Machine Learning Imports ---
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

# --- Quản lý client httpx (TÙY CHỌN - Khuyến nghị cho hiệu suất) ---
# Sử dụng lifespan của FastAPI để tạo và đóng client HTTP một lần
import contextlib

# Khởi tạo client toàn cục
shared_httpx_client: httpx.AsyncClient | None = None

@contextlib.asynccontextmanager

async def lifespan(app: FastAPI):
    global shared_httpx_client
    shared_httpx_client = httpx.AsyncClient()
    print("FastAPI app startup: httpx.AsyncClient initialized.")
    yield
    if shared_httpx_client:
        await shared_httpx_client.aclose()
        print("FastAPI app shutdown: httpx.AsyncClient closed.")

app = FastAPI(lifespan=lifespan)

# --- Database Configuration (PostgreSQL) ---
# Đảm bảo biến môi trường DATABASE_URL được cấu hình trên Render hoặc sử dụng mặc định cho dev
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/taixiu_db")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Database Model Definition ---
class PhienTaiXiu(Base):
    __tablename__ = "phien_tai_xiu"

    id = Column(Integer, primary_key=True, index=True)
    expect_string = Column(String, unique=True, index=True, nullable=False)
    open_time = Column(DateTime)
    ket_qua = Column(String) # "Tài" or "Xỉu"
    tong = Column(Integer)
    xuc_xac_1 = Column(Integer)
    xuc_xac_2 = Column(Integer)
    xuc_xac_3 = Column(Integer)
    created_at = Column(DateTime, default=datetime.now)

# --- KHỞI TẠO BẢNG DATABASE (QUAN TRỌNG - BỎ COMMENT VÀ CHẠY 1 LẦN) ---
# Bỏ comment dòng dưới đây VÀ CHẠY ỨNG DỤNG CỦA BẠN MỘT LẦN.
# Ví dụ: uvicorn main:app --reload (nếu file của bạn tên là main.py)
# Sau khi bạn thấy bảng đã được tạo thành công trong PostgreSQL (kiểm tra bằng pgAdmin, DBeaver, hoặc psql),
# HÃY COMMENT LẠI DÒNG NÀY ĐỂ TRÁNH TẠO LẠI BẢNG MỖI KHI KHỞI ĐỘNG ỨNG DỤNG.
Base.metadata.create_all(bind=engine) # <--- DÒNG NÀY ĐÃ ĐƯỢC BỎ COMMENT

# Dependency để lấy Session DB cho mỗi request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Logic tính Tài Xỉu ---
def get_tai_xiu_result(xuc_xac_values: List[int]) -> Dict[str, any]:
    """Tính toán kết quả Tài/Xỉu từ 3 giá trị xúc xắc."""
    if len(xuc_xac_values) != 3:
        raise ValueError("Phải có đúng 3 giá trị xúc xắc.")

    x1, x2, x3 = xuc_xac_values
    tong = x1 + x2 + x3
    ket_qua = "Tài" if 11 <= tong <= 17 else "Xỉu" # Tài: 11-17, Xỉu: 4-10

    # Quy tắc cho "Bão" (bộ 3 đồng nhất) - thường được coi là Xỉu
    # (Nếu tổng là Tài nhưng có 3 xúc xắc giống nhau, vẫn là Xỉu)
    if x1 == x2 == x3:
        ket_qua = "Xỉu"

    return {"Tong": tong, "Xuc_xac_1": x1, "Xuc_xac_2": x2, "Xuc_xac_3": x3, "Ket_qua": ket_qua}

# --- Machine Learning Model for Prediction ---
def predict_with_ml_model(historical_results: List[str]) -> Dict[str, str]:
    """
    Sử dụng mô hình học máy để dự đoán kết quả Tài/Xỉu và độ tin cậy.
    """
    if len(historical_results) < 20:
        return {"Ket_qua_du_doan": "Không đủ dữ liệu để huấn luyện ML", "Do_tin_cay": "N/A"}

    le = LabelEncoder()
    # Đảm bảo fit với tất cả các nhãn có thể có
    le.fit(["Tài", "Xỉu"])
    encoded_results = le.transform(historical_results)

    window_size = 5
    X = []
    y = []

    # Tạo các cặp (đầu vào, nhãn) cho mô hình
    for i in range(len(encoded_results) - window_size):
        X.append(encoded_results[i : i + window_size])
        y.append(encoded_results[i + window_size])

    if not X: # Đảm bảo có dữ liệu sau khi tạo cửa sổ
        return {"Ket_qua_du_doan": "Không đủ dữ liệu sau khi tạo cửa sổ ML", "Do_tin_cay": "N/A"}

    X = np.array(X)
    y = np.array(y)

    try:
        model = LogisticRegression(solver='liblinear', random_state=42)
        model.fit(X, y)

        # Lấy N kết quả gần nhất để dự đoán
        last_n_results = encoded_results[-window_size:].reshape(1, -1)

        predicted_encoded = model.predict(last_n_results)[0]
        predicted_outcome = le.inverse_transform([predicted_encoded])[0]

        probabilities = model.predict_proba(last_n_results)[0]
        # Tìm độ tin cậy cho kết quả được dự đoán
        confidence_index = np.where(model.classes_ == predicted_encoded)[0][0]
        confidence = probabilities[confidence_index] * 100

        return {
            "Ket_qua_du_doan": predicted_outcome,
            "Do_tin_cay": f"{confidence:.2f}%"
        }
    except Exception as e:
        print(f"Error during ML prediction: {e}")
        return {"Ket_qua_du_doan": "Lỗi khi chạy mô hình ML", "Do_tin_cay": "N/A"}

# --- API Endpoint Cũ (Giả định) ---
@app.get("/api/taixiu")
async def get_taixiu_data_old_api(db: Session = Depends(get_db)):
    EXTERNAL_API_URL_OLD = "https://1.bot/GetNewLottery/LT_Taixiu" # URL giả định từ ví dụ trước

    if not shared_httpx_client:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi nội bộ: httpx client chưa được khởi tạo."
        )

    try:
        response = await shared_httpx_client.get(EXTERNAL_API_URL_OLD, timeout=10.0)
        response.raise_for_status()
        external_data = response.json()
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi kết nối đến API bên ngoài (cũ): {exc}. Vui lòng thử lại sau."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi xử lý phản hồi từ API bên ngoài (cũ): {e}"
        )

    if external_data.get("state") != 1 or not external_data.get("data"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Dữ liệu từ API bên ngoài (cũ) không hợp lệ hoặc không có kết quả."
        )

    data = external_data["data"]

    try:
        expect_str = str(data["Expect"])
        open_code_str = data["OpenCode"]
        xuc_xac_values = [int(x.strip()) for x in open_code_str.split(',')]
        open_time_str = data["OpenTime"]
        open_time_dt = datetime.strptime(open_time_str, "%Y-%m-%d %H:%M:%S")

        current_result_data = get_tai_xiu_result(xuc_xac_values)

        current_phien_record: Optional[PhienTaiXiu] = None

        existing_phien = db.query(PhienTaiXiu).filter(
            PhienTaiXiu.expect_string == expect_str
        ).first()

        if not existing_phien:
            new_phien = PhienTaiXiu(
                expect_string=expect_str,
                open_time=open_time_dt,
                ket_qua=current_result_data["Ket_qua"],
                tong=current_result_data["Tong"],
                xuc_xac_1=current_result_data["Xuc_xac_1"],
                xuc_xac_2=current_result_data["Xuc_xac_2"],
                xuc_xac_3=current_result_data["Xuc_xac_3"]
            )
            db.add(new_phien)
            try:
                db.commit()
                db.refresh(new_phien)
                current_phien_record = new_phien
            except IntegrityError: # Xử lý trường hợp trùng lặp expect_string
                db.rollback()
                current_phien_record = db.query(PhienTaiXiu).filter(
                    PhienTaiXiu.expect_string == expect_str
                ).first()
                if not current_phien_record: # Nếu vẫn không tìm thấy sau rollback, là lỗi nghiêm trọng
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Lỗi hệ thống: Không thể lưu hoặc truy xuất phiên mới sau lỗi trùng lặp."
                    )
        else:
            current_phien_record = existing_phien

        HISTORY_LIMIT_FOR_ANALYSIS = 200
        DISPLAY_HISTORY_LIMIT = 20

        # Lấy lịch sử và sắp xếp theo expect_string giảm dần (phiên mới nhất lên đầu)
        lich_su = db.query(PhienTaiXiu).order_by(PhienTaiXiu.expect_string.desc()).limit(HISTORY_LIMIT_FOR_ANALYSIS).all()

        lich_su_formatted_full = [
            {
                "Phien": p.expect_string,
                "Ket_qua": p.ket_qua,
                "Tong": p.tong,
                "Xuc_xac_1": p.xuc_xac_1,
                "Xuc_xac_2": p.xuc_xac_2,
                "Xuc_xac_3": p.xuc_xac_3,
                "OpenTime": p.open_time.strftime("%Y-%m-%d %H:%M:%S") if p.open_time else None
            } for p in lich_su
        ]
        lich_su_formatted_display = lich_su_formatted_full[:DISPLAY_HISTORY_LIMIT]

        historical_outcomes_for_analysis = [p["Ket_qua"] for p in lich_su_formatted_full]

        ml_prediction = predict_with_ml_model(historical_outcomes_for_analysis)

        return {
            "Ket_qua_phien_hien_tai": current_phien_record.ket_qua,
            "Ma_phien_hien_tai": current_phien_record.expect_string,
            "Tong_diem_hien_tai": current_phien_record.tong,
            "Xuc_xac_hien_tai": [
                current_phien_record.xuc_xac_1,
                current_phien_record.xuc_xac_2,
                current_phien_record.xuc_xac_3
            ],
            "admin_name": "Nhutquang",
            "Lich_su_gan_nhat": lich_su_formatted_display,
            "Du_doan_phien_tiep_theo_ML": ml_prediction
        }

    except (KeyError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Dữ liệu từ API bên ngoài (cũ) không đúng định dạng hoặc thiếu trường bắt buộc: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi không xác định trong quá trình xử lý yêu cầu (cũ): {e}"
        )

# --- API Endpoint Mới (Wanglin API) ---
@app.get("/api/taixiu/wanglin") # Đây là endpoint mới
async def get_taixiu_data_wanglin_api(db: Session = Depends(get_db)):
    EXTERNAL_API_URL_WANGLIN = "https://wanglinapiws.up.railway.app/api/taixiu"

    if not shared_httpx_client:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi nội bộ: httpx client chưa được khởi tạo."
        )

    try:
        response = await shared_httpx_client.get(EXTERNAL_API_URL_WANGLIN, timeout=10.0)
        response.raise_for_status()
        external_data = response.json()
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi kết nối đến API Wanglin: {exc}. Vui lòng thử lại sau."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi xử lý phản hồi từ API Wanglin: {e}"
        )

    if not external_data or not all(k in external_data for k in ["Ket_qua", "Phien", "Tong", "Xuc_xac_1", "Xuc_xac_2", "Xuc_xac_3", "id"]):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Dữ liệu từ API Wanglin không hợp lệ hoặc thiếu trường bắt buộc."
        )

    try:
        expect_str = str(external_data["Phien"])
        ket_qua_str = external_data["Ket_qua"]
        tong_val = external_data["Tong"]
        xuc_xac_1_val = external_data["Xuc_xac_1"]
        xuc_xac_2_val = external_data["Xuc_xac_2"]
        xuc_xac_3_val = external_data["Xuc_xac_3"]
        open_time_dt = datetime.now() # API Wanglin không có OpenTime, sử dụng thời gian hiện tại
        admin_id_from_api = external_data["id"]

        current_phien_record: Optional[PhienTaiXiu] = None

        existing_phien = db.query(PhienTaiXiu).filter(
            PhienTaiXiu.expect_string == expect_str
        ).first()

        if not existing_phien:
            new_phien = PhienTaiXiu(
                expect_string=expect_str,
                open_time=open_time_dt,
                ket_qua=ket_qua_str,
                tong=tong_val,
                xuc_xac_1=xuc_xac_1_val,
                xuc_xac_2=xuc_xac_2_val,
                xuc_xac_3=xuc_xac_3_val
            )
            db.add(new_phien)
            try:
                db.commit()
                db.refresh(new_phien)
                current_phien_record = new_phien
            except IntegrityError: # Xử lý trường hợp trùng lặp expect_string
                db.rollback()
                current_phien_record = db.query(PhienTaiXiu).filter(
                    PhienTaiXiu.expect_string == expect_str
                ).first()
                if not current_phien_record: # Nếu vẫn không tìm thấy sau rollback, là lỗi nghiêm trọng
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Lỗi hệ thống: Không thể lưu hoặc truy xuất phiên mới sau lỗi trùng lặp."
                    )
        else:
            current_phien_record = existing_phien

        HISTORY_LIMIT_FOR_ANALYSIS = 200
        DISPLAY_HISTORY_LIMIT = 20

        # Lấy lịch sử và sắp xếp theo expect_string giảm dần (phiên mới nhất lên đầu)
        lich_su = db.query(PhienTaiXiu).order_by(PhienTaiXiu.expect_string.desc()).limit(HISTORY_LIMIT_FOR_ANALYSIS).all()

        lich_su_formatted_full = [
            {
                "Phien": p.expect_string,
                "Ket_qua": p.ket_qua,
                "Tong": p.tong,
                "Xuc_xac_1": p.xuc_xac_1,
                "Xuc_xac_2": p.xuc_xac_2,
                "Xuc_xac_3": p.xuc_xac_3,
                "OpenTime": p.open_time.strftime("%Y-%m-%d %H:%M:%S") if p.open_time else None
            } for p in lich_su
        ]
        lich_su_formatted_display = lich_su_formatted_full[:DISPLAY_HISTORY_LIMIT]

        historical_outcomes_for_analysis = [p["Ket_qua"] for p in lich_su_formatted_full]

        ml_prediction = predict_with_ml_model(historical_outcomes_for_analysis)

        return {
            "Ket_qua_phien_hien_tai": current_phien_record.ket_qua,
            "Ma_phien_hien_tai": current_phien_record.expect_string,
            "Tong_diem_hien_tai": current_phien_record.tong,
            "Xuc_xac_hien_tai": [
                current_phien_record.xuc_xac_1,
                current_phien_record.xuc_xac_2,
                current_phien_record.xuc_xac_3
            ],
            "admin_name": admin_id_from_api,
            "Lich_su_gan_nhat": lich_su_formatted_display,
            "Du_doan_phien_tiep_theo_ML": ml_prediction
        }

    except (KeyError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Dữ liệu từ API Wanglin không đúng định dạng hoặc thiếu trường bắt buộc: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi không xác định trong quá trình xử lý yêu cầu (Wanglin): {e}"
        )
