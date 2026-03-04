# File: src/modules/data_loader.py
import pandas as pd
import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "student_academic_performance_1M.csv"

@st.cache_data(show_spinner=False)
def load_raw_dataset(sample_size=None):
    """
    Tải dữ liệu thô với kỹ thuật tối ưu hóa bộ nhớ.
    - sample_size: Nếu truyền vào số lượng, sẽ lấy mẫu để tăng tốc.
    """
    if not DATA_PATH.exists():
        return pd.DataFrame()

    try:
        # Tối ưu: Chỉ đọc các cột cần thiết nếu bạn biết trước, 
        # ở đây tôi đọc hết nhưng sẽ downcast sau.
        df = pd.read_csv(DATA_PATH)

        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)

        # --- TỐI ƯU HÓA KIỂU DỮ LIỆU ---
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
            if df[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
            if 'flag' in col or 'gender' in col:
                df[col] = df[col].astype('category')
        
        return df
    except Exception as e:
        st.error(f"Lỗi hệ thống khi đọc dữ liệu: {e}")
        return pd.DataFrame()

def add_ai_features(df_input):
    if df_input.empty: return df_input
    df = df_input.copy()
    if 'ai_tool_usage' in df.columns:
        df['ai_tool_usage_label'] = df['ai_tool_usage'].map({
            1.0: 'Có sử dụng AI', 
            0.0: 'Không sử dụng AI'
        }).fillna('Không rõ').astype('category')
    return df

def add_tuition_features(df_input):
    if df_input.empty: return df_input
    df = df_input.copy()
    if 'private_tuition' in df.columns:
        df['private_tuition_label'] = df['private_tuition'].map({
            1.0: 'Có học thêm', 
            0.0: 'Không học thêm'
        }).fillna('Không rõ').astype('category')
    return df

def add_parental_education_features(df_input):
    if df_input.empty: return df_input
    df = df_input.copy()
    if 'parent_education' in df.columns:
        bins = [-1, 0, 1, 2, 3, 4, 5]
        labels = ['Không bằng cấp', 'Tiểu học', 'THCS', 'THPT', 'Đại học', 'Sau đại học']
        df['parent_education_group'] = pd.cut(
            df['parent_education'], 
            bins=bins, 
            labels=labels
        ).astype('category')
    return df

def add_family_income_features(df_input):
    if df_input.empty: return df_input
    df = df_input.copy()
    if 'family_income' in df.columns:
        df['family_income_rounded'] = df['family_income'].round(2)
    return df

def add_mental_stress_features(df_input):
    if df_input.empty: return df_input
    df = df_input.copy()
    if 'mental_stress' in df.columns:
        bins = [-0.1, 3, 7, 10]
        labels = ['Nhẹ','Đáng kể', 'Cực độ']
        df['mental_stress_group'] = pd.cut(
            df['mental_stress'], 
            bins=bins, 
            labels=labels,
            include_lowest=True,
            right=True
        ).astype('category')
    return df

def add_study_hours_daily_features(df_input):
    if df_input.empty: return df_input
    df = df_input.copy()
    if 'study_hours_daily' in df.columns:
        df['study_hours_daily_rounded'] = df['study_hours_daily'].round().astype('int8')
        bins = [-0.1, 1, 3, 5, 10]
        labels = ['Rất ít (0-1h)', 'Ít (1-3h)', 'Trung bình (3-5h)', 'Nhiều (>5h)']
        df['study_hours_group'] = pd.cut(
            df['study_hours_daily'], 
            bins=bins, 
            labels=labels,
            include_lowest=True,
            right=True
        ).astype('category')
    return df

def add_coding_features(df_input):
    if df_input.empty: return df_input
    df = df_input.copy()
    if 'coding_practice_hours' in df.columns:
        df['coding_hours_rounded'] = df['coding_practice_hours'].round().astype('int8')
        
        bins = [-0.1, 2, 5, 10, 100]
        labels = ['Ít (0-2h)', 'Trung bình (2-5h)', 'Nhiều (5-10h)', 'Rất nhiều (>10h)']
        
        df['coding_group'] = pd.cut(
            df['coding_practice_hours'], 
            bins=bins, 
            labels=labels,
            right=True
        ).astype('category')
            
    return df

def add_lms_forum_features(df_input):
    if df_input.empty: return df_input
    df = df_input.copy()
    
    if 'lms_login_frequency' in df.columns:
        df['lms_login_int'] = df['lms_login_frequency'].round().astype('int16')
    
    if 'forum_participation' in df.columns:
        bins = [-1, 0, 5, 10, 1000]
        labels = ['Không tham gia', 'Thấp (1-5)', 'Trung bình (6-10)', 'Cao (>10)']
        df['forum_activity_group'] = pd.cut(
            df['forum_participation'], 
            bins=bins, 
            labels=labels
        ).astype('category')
        
    return df

def add_risk_and_habit_features(df_input):
    """
    Tạo các nhãn và mask dùng cho phân tích rủi ro & thói quen.
    Giữ nguyên các cột số để tính toán Radar Chart, chỉ thêm cột phân loại mới.
    """
    if df_input.empty: return df_input
    df = df_input.copy()

    # Tạo cờ boolean trực tiếp trong DataFrame 
    if 'dropout_risk_score' in df.columns:
        df['is_high_risk'] = df['dropout_risk_score'] >= 0.9

    if 'top_performer_flag' in df.columns:
        # Chuyển từ category/int sang boolean để filter dễ dàng hơn
        df['is_top_performer'] = df['top_performer_flag'].astype(int) == 1

    # [TÙY CHỌN]: Tạo thêm các nhãn tĩnh cho thói quen nếu sau này muốn dùng Boxplot
    # Ví dụ: Nhãn giấc ngủ (Giữ nguyên cột sleep_hours bằng số)
    if 'sleep_hours' in df.columns:
        bins = [-0.1, 4, 7, 9, 24]
        labels = ['Thiếu ngủ trầm trọng', 'Thiếu ngủ', 'Đủ giấc', 'Ngủ nhiều']
        df['sleep_quality_label'] = pd.cut(df['sleep_hours'], bins=bins, labels=labels).astype('category')

    return df