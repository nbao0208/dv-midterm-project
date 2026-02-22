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
    load_raw_dataset, 
    add_ai_features, 
    add_coding_features, 
    add_lms_forum_features
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
        .pipe(add_ai_features)
        .pipe(add_coding_features)
        .pipe(add_lms_forum_features)
    )

    # 3. Tạo tập SAMPLE từ tập full đã xử lý (Dùng cho Box/Line/Heatmap để nhẹ)
    SAMPLE_N = 50000
    if len(df_full) > SAMPLE_N:
        df_sampled = df_full.sample(n=SAMPLE_N, random_state=42)
    else:
        df_sampled = df_full

# --- GIAO DIỆN ---
st.title("Phân tích chuyên sâu: Tác động của Công nghệ và Thói quen học tập")
st.divider()

# --- PHẦN 1: AI ---
st.header("Tác động của việc sử dụng công cụ AI")
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Phân bố nhóm học sinh")
    # DÙNG df_full: Để tính tỷ lệ % chính xác tuyệt đối
    ai_dist = df_full['ai_tool_usage_label'].value_counts().reset_index()
    ai_dist.columns = ['status', 'count']
    
    fig_pie = px.pie(
        ai_dist, names='status', values='count', color='status',
        color_discrete_map={'Có sử dụng AI': '#636EFA', 'Không sử dụng AI': '#EF553B', 'Không rõ': '#CCCCCC'},
        hole=0.4,
        labels={'status': 'Trạng thái', 'count': 'Số lượng HS'}
    )
    fig_pie.update_traces(textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("So sánh hiệu suất học tập")
    
    # 1. Định nghĩa danh sách các môn học (Tên hiển thị -> Tên cột trong DataFrame)
    metrics_map = {
        "Điểm Kỹ thuật (Computer Score)": "computer_score",
        "Điểm Toán (Math Score)": "math_score",
        "Điểm Khoa học (Science Score)": "science_score",
        "Điểm Tiếng Anh (English Score)": "english_score",
        "Điểm Lịch sử (History Score)": "history_score",
        "Điểm Tổng kết (Final GPA)": "final_gpa"
    }
    
    # 2. Selectbox: Lấy danh sách key từ metrics_map
    selected_metric_label = st.selectbox(
        "Chọn môn học/chỉ số phân tích:",
        options=list(metrics_map.keys()),
        key="p1_sel"
    )
    
    # Lấy tên cột tương ứng dựa trên lựa chọn
    y_col_p1 = metrics_map[selected_metric_label]
    
    # 3. Vẽ biểu đồ Box Plot (Dùng df_sampled để tối ưu tốc độ)
    fig_box = px.box(
        df_sampled, 
        x='ai_tool_usage_label', 
        y=y_col_p1, 
        color='ai_tool_usage_label',
        points=False, # Tắt outliers để không bị lag trình duyệt
        color_discrete_map={'Có sử dụng AI': '#636EFA', 'Không sử dụng AI': '#EF553B', 'Không rõ': '#CCCCCC'},
        # Labels: Mapping trục Y thành tên tiếng Việt người dùng đã chọn
        labels={
            'ai_tool_usage_label': 'Nhóm học sinh',
            y_col_p1: selected_metric_label
        }
    )
    
    # Tắt legend vì trục X và màu sắc đã thể hiện rõ thông tin
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

st.divider()

# --- PHẦN 2: CODING ---
st.header("Tác động của Thời gian Thực hành Lập trình")
col3, col4 = st.columns([1, 2])

with col3:
    st.subheader("Phân bố thời lượng")
    # DÙNG df_full: Histogram cần đếm số lượng chính xác
    coding_counts = df_full['coding_group'].value_counts().reset_index()
    coding_counts.columns = ['group', 'count']
    sort_order = ['Ít (0-2h)', 'Trung bình (2-5h)', 'Nhiều (5-10h)', 'Rất nhiều (>10h)']
    
    fig_bar = px.bar(
        coding_counts, x='group', y='count', color='group',
        category_orders={'group': sort_order},
        color_discrete_sequence=px.colors.qualitative.Safe,
        labels={'group': 'Nhóm thời gian', 'count': 'Số lượng HS'}
    )
    fig_bar.update_layout(showlegend=False, xaxis_title="Nhóm giờ thực hành", yaxis_title="Số lượng học sinh")
    st.plotly_chart(fig_bar, use_container_width=True)

with col4:
    st.subheader("Xu hướng điểm số các môn học")
    
    # 1. Định nghĩa danh sách môn học có thể chọn
    subject_map = {
        "Điểm Tổng kết (GPA)": "final_gpa",
        "Điểm Tin học": "computer_score",
        "Điểm Toán": "math_score",
        "Điểm Khoa học": "science_score",
        "Điểm Tiếng Anh": "english_score",
        "Điểm Lịch sử": "history_score"
    }
    
    # 2. Multiselect: Cho phép chọn nhiều môn
    selected_subjects = st.multiselect(
        "Chọn các môn học để so sánh:",
        options=list(subject_map.keys()),
        default=["Điểm Tổng kết (GPA)", "Điểm Tin học"], # Mặc định hiển thị 2 môn này
        key="p2_multi_sel"
    )
    
    if not selected_subjects:
        st.warning("Vui lòng chọn ít nhất một môn học.")
    else:
        # Lấy danh sách tên cột tương ứng
        y_cols = [subject_map[s] for s in selected_subjects]
        
        # DÙNG df_sampled
        # Tính trung bình cho TẤT CẢ các cột được chọn theo nhóm giờ code
        trend_data = df_sampled.groupby('coding_hours_rounded')[y_cols].mean().reset_index()
        
        # Đổi tên cột trong trend_data để Legend hiển thị tiếng Việt đẹp hơn
        # Tạo dict ngược: {'final_gpa': 'Điểm Tổng kết (GPA)', ...}
        reverse_map = {v: k for k, v in subject_map.items()}
        trend_data = trend_data.rename(columns=reverse_map)
        
        # Vẽ biểu đồ: y lấy danh sách các tên cột tiếng Việt mới
        fig_line = px.line(
            trend_data, 
            x='coding_hours_rounded', 
            y=selected_subjects, # Lúc này tên cột trong df đã là tên tiếng Việt
            markers=True, 
            line_shape="spline",
            labels={
                'coding_hours_rounded': 'Số giờ thực hành (làm tròn)', 
                'value': 'Điểm trung bình',
                'variable': 'Môn học' # Tên cho Legend
            }
        )
        
        fig_line.update_traces(hovertemplate='Giờ: %{x}<br>Điểm: %{y:.2f}')
        st.plotly_chart(fig_line, use_container_width=True)

st.divider()

# --- PHẦN 3: LMS & FORUM ---
st.header("3. Tương tác Hệ thống: LMS và Diễn đàn")

# 1. Selectbox nằm trên cùng để điều khiển chung (nếu logic cho phép)
score_opts = {
    "Điểm Tổng kết (GPA)": "final_gpa",
    "Điểm thi Chuẩn hóa": "standardized_exam_score",
    "Điểm Quiz TB": "quiz_avg",
    "Điểm Dự án": "project_score"
}

sel_label = st.selectbox("Loại điểm phân tích:", list(score_opts.keys()))
y_col_p3 = score_opts[sel_label]

# --- CHIA 2 CỘT ĐỂ CÂN ĐỐI BỐ CỤC ---
col_lms_1, col_lms_2 = st.columns(2)

# --- CỘT 1: AREA CHART ---
with col_lms_1:
    st.subheader(f"Xu hướng theo tần suất truy cập LMS")
    
    # Xử lý dữ liệu (Dùng df_sampled)
    lms_trend = df_sampled.groupby('lms_login_int')[y_col_p3].agg(['mean', 'count']).reset_index()
    lms_trend = lms_trend[lms_trend['count'] > 50] 

    fig_lms = px.area(
        lms_trend, x='lms_login_int', y='mean',
        labels={'lms_login_int': 'Tần suất đăng nhập (lần/tuần)', 'mean': 'Điểm TB'}
    )
    fig_lms.update_traces(line_color="#63B1FA")
    # Chỉnh margin để biểu đồ tận dụng tối đa không gian cột
    fig_lms.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_lms, use_container_width=True)

# --- CỘT 2: HEATMAP ---
with col_lms_2:
    st.subheader("Ma trận Hiệu suất: LMS & Diễn đàn")
    
    # Xử lý dữ liệu Heatmap
    lms_bins = [0, 10, 20, 30, 100]; lms_labs = ['0-10', '10-20', '20-30', '>30']
    df_sampled['lms_bin'] = pd.cut(df_sampled['lms_login_frequency'], bins=lms_bins, labels=lms_labs)

    pivot_table = df_sampled.pivot_table(values=y_col_p3, index='forum_activity_group', columns='lms_bin', aggfunc='mean')
    sort_f = ['Cao (>10)', 'Trung bình (6-10)', 'Thấp (1-5)', 'Không tham gia']
    pivot_table = pivot_table.reindex([x for x in sort_f if x in pivot_table.index])

    fig_heat = px.imshow(
        pivot_table, text_auto='.1f', color_continuous_scale='YlGnBu',
        labels=dict(x="Tần suất truy cập LMS", y="Diễn đàn", color="Điểm"),
        aspect="auto" # QUAN TRỌNG: Giúp Heatmap tự co giãn theo khung cột
    )
    # Đưa trục X xuống dưới & Tinh chỉnh layout
    fig_heat.update_xaxes(side="bottom")
    fig_heat.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    
    st.plotly_chart(fig_heat, use_container_width=True)