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