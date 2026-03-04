import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

# --- SETUP HỆ THỐNG ---
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from modules.data_loader import load_raw_dataset, add_risk_and_habit_features

st.set_page_config(page_title="Phân tích Hiệu suất & Rủi ro", layout="wide")

# --- XỬ LÝ DỮ LIỆU ---
with st.spinner("Đang tải cơ sở dữ liệu học tập..."):
    # Chạy pipeline xử lý dữ liệu ngay từ đầu
    raw_df = load_raw_dataset(sample_size=None)

    if raw_df.empty:
        st.warning("Không thể tải dữ liệu. Vui lòng kiểm tra đường dẫn.")
        st.stop()

    df_processed = raw_df.pipe(add_risk_and_habit_features)

# --- GIAO DIỆN ---
st.title("Phân tích Hiệu suất & Cảnh báo Rủi ro")
st.divider()

# ==========================================
# CÂU HỎI 1: TỔ HỢP HÀNH VI VÀ TỶ LỆ TRƯỢT MÔN
# ==========================================
st.header("1. Tác động của Điểm quá trình & Chuyên cần đến Kết quả")

# Thay đổi: Ánh xạ tên hiển thị -> tên cột trong DataFrame
diem_qua_trinh_cols = {
    "Trung bình Bài tập": "assignment_avg",
    "GPA hiện tại": "previous_gpa",
    "Điểm Toán": "math_score",
    "Điểm Khoa học": "science_score",
    "Điểm Tiếng Anh": "english_score",
    "Điểm Lịch sử": "history_score",
    "Điểm Tin học": "computer_score",
    "Điểm Quiz": "quiz_avg",
    "Điểm Dự án": "project_score"
}

chuyen_can_cols = {
    "Tỷ lệ Đi học": "attendance_rate",
    "Tần suất đăng nhập LMS": "lms_login_frequency",
}

col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    x_label = st.selectbox("Chọn yếu tố Điểm quá trình (Trục X):", list(diem_qua_trinh_cols.keys()))
with col_sel2:
    y_label = st.selectbox("Chọn yếu tố Chuyên cần (Trục Y):", list(chuyen_can_cols.keys()))

# Trích xuất dữ liệu trực tiếp bằng tên cột
x_col = diem_qua_trinh_cols[x_label]
y_col = chuyen_can_cols[y_label]

df_plot = df_processed[[x_col, y_col, 'pass_fail']].copy()
df_plot.columns = ['x_val', 'y_val', 'target']
df_plot = df_plot.dropna()

bins_labels = ['Rất thấp', 'Thấp', 'Trung bình', 'Cao', 'Rất cao']
if not df_plot.empty:
    with st.expander("Xem Phân phối Dữ liệu thực tế (Click để mở rộng)", expanded=True):
        col_dist1, col_dist2 = st.columns(2)

        with col_dist1:
            fig_hist_x = px.histogram(
                df_plot, x='x_val', marginal="box",
                title=f"Phân phối: {x_label}", color_discrete_sequence=['#3366CC']
            )
            fig_hist_x.update_layout(xaxis_title=x_label, yaxis_title="Số lượng SV",
                                     margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_hist_x, use_container_width=True)

        with col_dist2:
            fig_hist_y = px.histogram(
                df_plot, x='y_val', marginal="box",
                title=f"Phân phối: {y_label}", color_discrete_sequence=['#DC3912']
            )
            fig_hist_y.update_layout(xaxis_title=y_label, yaxis_title="Số lượng SV",
                                     margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_hist_y, use_container_width=True)
    try:
        df_plot['x_binned'] = pd.qcut(df_plot['x_val'], q=5, labels=bins_labels, duplicates='drop')
        df_plot['y_binned'] = pd.qcut(df_plot['y_val'], q=5, labels=bins_labels, duplicates='drop')

        st.subheader("Tỷ lệ Đậu môn theo Tổ hợp được chọn")
        pivot_pass = df_plot.pivot_table(index='y_binned', columns='x_binned', values='target', aggfunc='mean') * 100
        pivot_pass = pivot_pass.sort_index(ascending=False)

        fig_heat = px.imshow(
            pivot_pass, text_auto='.1f', color_continuous_scale='RdYlGn',
            labels=dict(x=x_label, y=y_label, color="Tỷ lệ Đậu (%)"), aspect="auto"
        )
        fig_heat.update_xaxes(side="bottom")
        fig_heat.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_heat, use_container_width=True)
    except Exception as e:
        st.error("Dữ liệu phân phối lệch nên không thể tự động chia nhóm. Vui lòng chọn tổ hợp khác.")
else:
    st.warning("Dữ liệu không khả dụng.")

st.divider()

# ==========================================
# CÂU HỎI 2: CẤU TRÚC THÓI QUEN (RADAR CHART & GAP ANALYSIS)
# ==========================================
st.header("2. So sánh Cấu trúc Thói quen: Nguy cơ Bỏ học vs Top Đầu")

habit_cols = {
    'Học tự túc (h)': 'study_hours_daily',
    'Ôn tập (h)': 'revision_hours',
    'Giấc ngủ (h)': 'sleep_hours',
    'Dùng thiết bị (h)': 'screen_time',
    'Xem Video (h)': 'video_watch_hours',
    'Thực hành Code (h)': 'coding_practice_hours',
    'Vận động thể chất': 'physical_activity'
}

# Chỉ lấy ra phần dữ liệu chứa các thói quen bằng danh sách tên cột
habit_col_names = list(habit_cols.values())
habit_df = df_processed[habit_col_names]

# Lọc dữ liệu bằng 2 cột cờ đã tạo ở tầng data_loader
risk_habits = df_processed[df_processed['is_high_risk']][habit_col_names]
top_habits = df_processed[df_processed['is_top_performer']][habit_col_names]

if not risk_habits.empty and not top_habits.empty:
    mean_risk = risk_habits.mean()
    mean_top = top_habits.mean()

    st.subheader("A. Hình dáng Cấu trúc thói quen (0-100%)")

    min_vals = habit_df.min()
    max_vals = habit_df.max()
    norm_risk = ((mean_risk - min_vals) / (max_vals - min_vals) * 100).fillna(0).tolist()
    norm_top = ((mean_top - min_vals) / (max_vals - min_vals) * 100).fillna(0).tolist()

    norm_risk += [norm_risk[0]]
    norm_top += [norm_top[0]]
    habit_labels_closed = list(habit_cols.keys()) + [list(habit_cols.keys())[0]]

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=norm_risk, theta=habit_labels_closed, fill='toself',
        line_color='#EF553B', name=f'Nguy cơ bỏ học (N={len(risk_habits):,})'
    ))

    fig_radar.add_trace(go.Scatterpolar(
        r=norm_top, theta=habit_labels_closed, fill='toself',
        line_color='#00CC96', name=f'Top đầu (N={len(top_habits):,})'
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                range=[0, 100], showticklabels=False, showline=False, ticklen=0
            )
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        margin=dict(t=40, b=20, l=40, r=40)
    )

    st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader("B. Mức độ chênh lệch thực tế giữa 2 nhóm")

    gap_df = pd.DataFrame({
        'Thói quen': list(habit_cols.keys()),
        'Top đầu': mean_top.values,
        'Nguy cơ': mean_risk.values
    })
    gap_df['Chênh lệch (%)'] = ((gap_df['Top đầu'] - gap_df['Nguy cơ']) / gap_df['Nguy cơ'].replace(0, 0.01) * 100)

    gap_df = gap_df.sort_values('Chênh lệch (%)', ascending=True)
    gap_df['Màu sắc'] = gap_df['Chênh lệch (%)'].apply(lambda x: '#00CC96' if x > 0 else '#EF553B')

    fig_gap = px.bar(
        gap_df, x='Chênh lệch (%)', y='Thói quen', orientation='h',
        text=gap_df['Chênh lệch (%)'].apply(lambda x: f"{x:+.1f}%"),
        labels={'Thói quen': '', 'Chênh lệch (%)': 'Nhóm Top đầu làm NHIỀU/ÍT hơn Nhóm Nguy cơ Bỏ học (%)'}
    )
    fig_gap.update_traces(marker_color=gap_df['Màu sắc'], textposition='outside')
    fig_gap.update_layout(margin=dict(t=20, b=20, l=20, r=20), xaxis=dict(showgrid=True))
    st.plotly_chart(fig_gap, use_container_width=True)

else:
    st.warning("Không tìm thấy đủ dữ liệu để tạo biểu đồ phân loại.")