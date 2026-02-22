import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# --- 1. SETUP ĐƯỜNG DẪN MODULE ---
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from modules.data_loader import load_raw_dataset, add_ai_features

# --- 2. CẤU HÌNH TRANG (Đã bỏ icon) ---
st.set_page_config(
    page_title="Student Analytics Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. SIDEBAR CHUNG (Đã bỏ ảnh và icon) ---
with st.sidebar:
    st.title("Education Dashboard")
    st.info(
        """
        Dashboard phân tích dữ liệu giáo dục 
        với tập dữ liệu 1 triệu bản ghi.
        """
    )
    st.markdown("---")
    st.markdown("**Version:** 1.0.0")
    st.markdown("**Author:** Data Team")

# --- 4. NỘI DUNG TRANG CHỦ ---

# Load dữ liệu tổng quan
df = load_raw_dataset()

# Header (Đã bỏ icon)
st.title("Student Performance Analytics Hub")
st.markdown("""
Chào mừng đến với hệ thống phân tích dữ liệu học tập. 
Hệ thống cung cấp cái nhìn toàn diện về các yếu tố ảnh hưởng đến kết quả học tập của sinh viên.
""")
st.markdown("---")

# --- KHỐI KPI TỔNG QUAN ---
if not df.empty:
    total_students = len(df)
    avg_gpa = df['final_gpa'].mean()
    
    # Tính tỷ lệ dùng AI
    df_ai = add_ai_features(df)
    ai_usage_rate = (df_ai['ai_tool_usage'] == 1).mean() * 100
    
    avg_cs = df['computer_score'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Tổng số học sinh", value=f"{total_students:,.0f}")
    
    with col2:
        st.metric(label="GPA Trung bình", value=f"{avg_gpa:.2f}")
        
    with col3:
        st.metric(label="Tỷ lệ sử dụng AI", value=f"{ai_usage_rate:.1f}%")
        
    with col4:
        st.metric(label="Điểm Kỹ thuật TB", value=f"{avg_cs:.1f}")
else:
    st.error("Không thể tải dữ liệu tổng quan.")

st.markdown("---")

# --- KHỐI ĐIỀU HƯỚNG THEO 4 NHÓM ---
st.subheader("Danh mục phân tích chi tiết")
st.markdown("Chọn chủ đề phân tích bên dưới để xem báo cáo chi tiết:")

# Hàng 1: Nhóm 1 và Nhóm 2
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    with st.container(border=True):
        st.subheader("Tác động của Công nghệ và Thói quen học tập")
        st.markdown("""
        **Nội dung phân tích:**
        * Tương quan giữa AI, Lập trình và Điểm số.
        * Ảnh hưởng của LMS và Diễn đàn đến quá trình học.
        """)
        # Link đến file 01 hiện có
        if st.button("Xem báo cáo", key="btn_grp1"):
            st.switch_page("pages/01_education_analyze.py")

with row1_col2:
    with st.container(border=True):
        st.subheader("Sức khỏe Thể chất và Tinh thần")
        st.markdown("""
        **Nội dung phân tích:**
        * Giấc ngủ, Top Performance và Nguy cơ rớt môn.
        * Thời gian màn hình (Screen time) và Căng thẳng (Stress).
        """)
        # Cần tạo file pages/02_health_lifestyle.py
        if st.button("Xem báo cáo", key="btn_grp2"):
            try:
                st.switch_page("pages/02_health_lifestyle.py")
            except Exception:
                st.warning("Vui lòng tạo file 'pages/02_health_lifestyle.py'")

# Hàng 2: Nhóm 3 và Nhóm 4
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    with st.container(border=True):
        st.subheader("Yếu tố Kinh tế và Hỗ trợ Xã hội")
        st.markdown("""
        **Nội dung phân tích:**
        * Ảnh hưởng của Thu nhập và Học vấn cha mẹ.
        * So sánh kết quả nhóm Học thêm vs Tự học.
        """)
        # Cần tạo file pages/03_socioeconomic.py
        if st.button("Xem báo cáo", key="btn_grp3"):
            try:
                st.switch_page("pages/03_socioeconomic.py")
            except Exception:
                st.warning("Vui lòng tạo file 'pages/03_socioeconomic.py'")

with row2_col2:
    with st.container(border=True):
        st.subheader("Phân tích Hiệu suất & Cảnh báo Rủi ro")
        st.markdown("""
        **Nội dung phân tích:**
        * Tổ hợp hành vi ảnh hưởng đến Trượt môn (Pass/Fail).
        * Cấu trúc thói quen nhóm Nguy cơ bỏ học vs Top đầu.
        """)
        # Cần tạo file pages/04_performance_risk.py
        if st.button("Xem báo cáo", key="btn_grp4"):
            try:
                st.switch_page("pages/04_performance_risk.py")
            except Exception:
                st.warning("Vui lòng tạo file 'pages/04_performance_risk.py'")