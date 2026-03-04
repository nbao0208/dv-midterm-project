import streamlit as st
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path

# --- SETUP HỆ THỐNG ---
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from modules.data_loader import (
    add_study_hours_daily_features,
    load_raw_dataset, 
    add_family_income_features,
    add_parental_education_features,
    add_tuition_features,
    add_mental_stress_features
)

st.set_page_config(page_title="Phân tích Học tập", layout="wide")

# --- XỬ LÝ DỮ LIỆU (Tách biệt Full và Sample) ---
# Dùng st.spinner tùy chỉnh thay vì spinner mặc định của cache
with st.spinner("Đang tải và xử lý cơ sở dữ liệu học tập..."):
    # 1. Load FULL dữ liệu (sample_size=None)
    raw_df_full = load_raw_dataset(sample_size=None)
    
    if raw_df_full.empty:
        st.warning("Không thể tải dữ liệu. Vui lòng kiểm tra đường dẫn.")
        st.stop()

    # 2. Xử lý feature trên tập FULL (để đếm số lượng Pie/Bar chính xác)
    df_full = (
        raw_df_full
        .pipe(add_family_income_features)
        .pipe(add_parental_education_features)
        .pipe(add_tuition_features)
        .pipe(add_mental_stress_features)
        .pipe(add_study_hours_daily_features)
    )

    # 3. Tạo tập SAMPLE từ tập full đã xử lý (Dùng cho Box/Line/Heatmap để nhẹ)
    SAMPLE_N = 50000
    if len(df_full) > SAMPLE_N:
        df_sampled = df_full.sample(n=SAMPLE_N, random_state=42)
    else:
        df_sampled = df_full

# GIAO DIỆN
st.title("Phân tích chuyên sâu: Yếu tố Kinh tế và Hỗ trợ Xã hội")
st.divider()

# THÔNG TIN TỔNG QUAN
st.header("Thông tin tổng quan về các yếu tố phân tích")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    avg_study_hours = df_full['study_hours_daily'].mean()
    st.metric(
        label="📚 Thời lượng tự học trung bình",
        value=f"{avg_study_hours:.2f} giờ"
    )
    
with kpi2:
    higher_edu = df_full['parent_education_group'].isin(['Không bằng cấp'])
    edu_rate = higher_edu.mean() * 100
    st.metric(label="🎓 Tỉ lệ Phụ huynh (Các bậc học)", value=f"{edu_rate:.1f}%")

with kpi3:
    median_income = df_full['family_income'].median()
    st.metric(label="💰 Thu nhập TB của gia đình", value=f"{median_income:.2f}")
    
with kpi4:
    avg_stress = df_full['mental_stress'].mean()
    st.metric(label="🤯 Mức Stress trung bình", value=f"{avg_stress:.1f} / 10")
    

st.divider()

# LIÊN HỆ TỔNG HỢP
st.header("Ảnh hưởng của yếu tố kinh tế, học vấn của gia đình lên học sinh")

# 1. Thêm Checkbox cho người dùng tự chọn (Mặc định là False - Ẩn dấu chấm)
show_data_points = st.checkbox("Hiển thị các điểm dữ liệu chi tiết", value=False)

SAMPLE_SCATTER = 3000
df_reg = df_sampled.sample(n=min(SAMPLE_SCATTER, len(df_sampled)), random_state=42)

edu_orders = ['Không bằng cấp', 'Tiểu học', 'THCS', 'THPT', 'Đại học', 'Sau đại học']

# 2. Vẽ biểu đồ ban đầu
fig_reg = px.scatter(
    df_reg, 
    x="family_income", 
    y="mental_stress", 
    color="parent_education_group",
    category_orders={"parent_education_group": edu_orders},
    trendline="ols",
    title="Biểu đồ tổng quan",
    labels={
        "family_income": "Thu nhập gia đình", 
        "mental_stress": "Mức độ Stress",
        "parent_education_group": "Học vấn của Phụ huynh"
    },
    color_discrete_sequence=px.colors.qualitative.Pastel
)

# 3. Xử lý logic Ẩn/Hiện dựa vào Checkbox
if show_data_points:
    fig_reg.update_traces(marker=dict(opacity=0.3), selector=dict(mode="markers"))
else:
    fig_reg.update_traces(marker=dict(opacity=0), selector=dict(mode="markers"))

# 4. Cấu hình giao diện
fig_reg.update_layout(
    height=600, 
    margin=dict(l=20, r=20, t=50, b=20),
    legend=dict(title="Học vấn (Nhấn đúp)", orientation="v", y=1, x=1.02)
)

st.plotly_chart(fig_reg, use_container_width=True)

with st.expander("💡 Hướng dẫn đọc biểu đồ và nhận xét", expanded=True):
    st.markdown("""
    **1. Cách đọc biểu đồ:**
    * **Đường xu hướng:** Thể hiện xu hướng chung. Hướng lên nghĩa là thu nhập tăng thì stress tăng, hướng xuống là ngược lại.
    * **Các chấm mờ:** Tùy chọn, dành cho chuyên gia phân tích muốn xem chi tiết.
    * **Tương tác:** Nhấn đúp vào một bậc học vấn bên phải để xem riêng nhóm đó.
                
    **2. Nhận xét:**
    * Các đường xu hướng đều nằm trong khoảng 4-6 (Mức độ trung bình).
    * Điều này cho thấy dù thu nhập có tăng hay học vấn phụ huynh có cao hơn thì mức độ stress vẫn ổn định trong đoạn trung bình, không có sự gia tăng rõ rệt.
    """)

st.divider()

# HỌC THÊM, THỜI LƯỢNG HỌC TẬP
st.header("Tình trạng học tập")
col5, col6 = st.columns(2)

with col5:
    st.subheader("Tỉ lệ học thêm")
    tuition_dist = df_full['private_tuition_label'].value_counts().reset_index()
    tuition_dist.columns = ['status', 'count']
    
    fig_pie = px.pie(
        tuition_dist, names='status', values='count', color='status',
        color_discrete_map={'Có học thêm': "#1DA06B", 'Không học thêm': "#C0280E", 'Không rõ': '#CCCCCC'},
        hole=0.4,
        labels={'status': 'Trạng thái', 'count': 'Số lượng HS'}
    )
    fig_pie.update_layout(
        margin=dict(l=90, r=90, t=80, b=80)
    )
    fig_pie.update_traces(textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with col6:
    st.subheader("Thời lượng học tập")


    study_dist = (
        df_full['study_hours_group']
        .value_counts()
        .sort_index()
        .reset_index()
    )
    study_dist.columns = ['Nhóm giờ học', 'Số lượng']

    fig_study_dist = px.bar(
        study_dist,
        x='Nhóm giờ học',
        y='Số lượng',
        color='Nhóm giờ học',
        category_orders={
            'Nhóm giờ học': [
                'Rất ít (0-1h)', 
                'Ít (1-3h)', 
                'Trung bình (3-5h)', 
                'Nhiều (>5h)'
            ]
        },
        labels={
            'Nhóm giờ học': 'Thời lượng học mỗi ngày',
            'Số lượng': 'Số học sinh'
        }
    )

    fig_study_dist.update_layout(
        height=400,
        margin=dict(l=40, r=20, t=40, b=20),
        showlegend=False
    )

    st.plotly_chart(fig_study_dist, use_container_width=True)

st.divider()


# HIỆU SUẤT TỔNG THỂ
st.header ("Hiệu suất học tập")
st.subheader("Kết quả học tập theo thời lượng và học thêm")
    
subject_map = {
    "Điểm Tổng kết (GPA)": "final_gpa",
    "Điểm Tin học": "computer_score",
    "Điểm Toán": "math_score",
    "Điểm Khoa học": "science_score",
    "Điểm Tiếng Anh": "english_score",
    "Điểm Lịch sử": "history_score"
}

selected_subject = st.selectbox(
    "Chọn môn học để so sánh:",
    options=list(subject_map.keys()),
    index=0, 
    key="p2_single_sel_fixed"
)

y_col = subject_map[selected_subject]


trend_data = df_sampled.groupby(['study_hours_daily_rounded', 'private_tuition_label'])[y_col].mean().reset_index()

fig_line = px.line(
    trend_data,
    x='study_hours_daily_rounded',
    y=y_col,
    color='private_tuition_label',
    markers=True,
    line_shape="spline",
    color_discrete_map={'Có học thêm': "#1DA06B", 'Không học thêm': "#C0280E"},
    labels={
        'study_hours_daily_rounded': 'Giờ học hàng ngày', 
        y_col: f'Điểm trung bình',
        'private_tuition_label': 'Trạng thái'
    }
)

fig_line.update_layout(
    height=400,
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(tickmode='linear', dtick=1) 
)

st.plotly_chart(fig_line, use_container_width=True)















        
