import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Thêm thư mục 'src' vào hệ thống tìm kiếm của Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.data_loader import load_raw_dataset  # dùng chung, đảm bảo merge ok

st.set_page_config(page_title="Đánh đổi & Cú đêm", layout="wide")

# ---------- Helpers ----------
def add_student_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "honors_flag" not in df.columns or "at_risk_flag" not in df.columns:
        return df

    # Đổi nhãn nhóm sang tiếng Việt
    df["student_group"] = np.select(
        [df["honors_flag"] == 1, df["at_risk_flag"] == 1],
        ["Nhóm tài năng", "Nhóm nguy cơ"],
        default="Nhóm thường",
    )
    # set order for consistent plotting
    df["student_group"] = pd.Categorical(
        df["student_group"],
        categories=["Nhóm tài năng", "Nhóm thường", "Nhóm nguy cơ"],
        ordered=True,
    )
    return df


def add_entertainment_screen_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    needed = {"screen_time", "online_course_hours"}
    if not needed.issubset(df.columns):
        return df

    df["entertainment_screen_time"] = (
        df["screen_time"] - df["online_course_hours"]
    ).clip(lower=0)
    return df


def agg_band_by_bins(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    bin_width: float = 1.0,
    max_x: float | None = None,
    agg: str = "mean",  # "mean" hoặc "median"
):
    d = df[[x_col, y_col]].dropna().copy()
    if max_x is not None:
        d = d[d[x_col] <= max_x]
    if d.empty:
        return pd.DataFrame(
            {"bin_left": [], "bin_center": [], "y": [], "lo": [], "hi": [], "n": []}
        )

    x_max = float(d[x_col].max())
    edges = np.arange(0, x_max + bin_width, bin_width)
    if len(edges) < 2:
        edges = np.array([0.0, max(1.0, x_max)])

    d["bin"] = pd.cut(d[x_col], bins=edges, include_lowest=True, right=False)
    d = d.dropna(subset=["bin"])
    if d.empty:
        return pd.DataFrame(
            {"bin_left": [], "bin_center": [], "y": [], "lo": [], "hi": [], "n": []}
        )

    g = d.groupby("bin", observed=True)[y_col]

    if agg == "mean":
        out = (
            g.agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"count": "n"})
        )
        out["se"] = out["std"] / np.sqrt(out["n"].clip(lower=1))
        out["y"] = out["mean"]
        out["lo"] = out["y"] - 1.96 * out["se"]
        out["hi"] = out["y"] + 1.96 * out["se"]
    else:
        # median + IQR band (ổn định và nhìn rõ hơn CI khi n lớn)
        out = g.quantile([0.25, 0.5, 0.75]).unstack().reset_index()
        out.columns = ["bin", "q25", "q50", "q75"]
        out["n"] = d.groupby("bin", observed=True)[y_col].size().values
        out["y"] = out["q50"]
        out["lo"] = out["q25"]
        out["hi"] = out["q75"]

    out["bin_left"] = out["bin"].apply(lambda b: float(b.left)).astype(float)
    out["bin_right"] = out["bin"].apply(lambda b: float(b.right)).astype(float)
    out["bin_center"] = (out["bin_left"] + out["bin_right"]) / 2

    return out[["bin_left", "bin_center", "y", "lo", "hi", "n"]].sort_values(
        "bin_left"
    )


# ---------- Load ----------
df_raw = load_raw_dataset()
if df_raw.empty:
    st.stop()

df = df_raw
df = add_student_group(df)
df = add_entertainment_screen_time(df)

st.title("Dashboard: Đánh đổi của Thủ khoa & Thời gian sử dụng màn hình")

tab1, tab2 = st.tabs(["1) Đánh đổi Ngủ vs Học", "2) Mức độ căng thẳng khi sử dụng màn hình"])

# =========================
# TAB 1: Trade-off
# =========================
with tab1:
    st.subheader("Sự đánh đổi của thủ khoa: ngủ ít để học nhiều?")
    cols_needed = {"student_group", "sleep_hours", "study_hours_daily"}
    if not cols_needed.issubset(df.columns):
        st.error(f"Thiếu cột cần thiết: {cols_needed - set(df.columns)}")
        st.stop()

    # Controls
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        sample_n = st.slider(
            "Số mẫu cho violin (giảm lag)", 50_000, 500_000, 200_000, step=50_000
        )
    with c2:
        show_points = st.selectbox("Hiển thị điểm", ["Không", "Rải nhẹ"], index=0)
    with c3:
        st.caption(
            "Violin + box giúp so sánh phân phối (trung vị/IQR/outliers) giữa "
            "Nhóm tài năng / Nhóm thường / Nhóm nguy cơ."
        )

    dplot = df[list(cols_needed)].dropna()
    if len(dplot) > sample_n:
        dplot = dplot.sample(sample_n, random_state=42)

    points_opt = False if show_points == "Không" else "outliers"

    left, right = st.columns(2)

    with left:
        fig_sleep = px.violin(
            dplot,
            x="student_group",
            y="sleep_hours",
            box=True,
            points=points_opt,
            category_orders={
                "student_group": ["Nhóm tài năng", "Nhóm thường", "Nhóm nguy cơ"]
            },
        )
        fig_sleep.update_layout(
            title="Phân phối Giờ ngủ (sleep_hours)",
            xaxis_title="",
            yaxis_title="Giờ",
        )
        st.plotly_chart(fig_sleep, use_container_width=True)

    with right:
        fig_study = px.violin(
            dplot,
            x="student_group",
            y="study_hours_daily",
            box=True,
            points=points_opt,
            category_orders={
                "student_group": ["Nhóm tài năng", "Nhóm thường", "Nhóm nguy cơ"]
            },
        )
        fig_study.update_layout(
            title="Phân phối Giờ học mỗi ngày (study_hours_daily)",
            xaxis_title="",
            yaxis_title="Giờ",
        )
        st.plotly_chart(fig_study, use_container_width=True)

    # ===== INSIGHT CARDS (Trade-off) =====
    agg = (
        df[["student_group", "sleep_hours", "study_hours_daily"]]
        .dropna()
        .groupby("student_group")
        .agg(
            sleep_median=("sleep_hours", "median"),
            sleep_mean=("sleep_hours", "mean"),
            study_median=("study_hours_daily", "median"),
            study_mean=("study_hours_daily", "mean"),
            n=("sleep_hours", "size"),
        )
    )

    # Lấy Nhóm tài năng vs Nhóm nguy cơ
    h = agg.loc["Nhóm tài năng"]
    r = agg.loc["Nhóm nguy cơ"]

    delta_sleep = float(h["sleep_median"] - r["sleep_median"])
    delta_study = float(h["study_median"] - r["study_median"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Δ Trung vị giờ ngủ (Tài năng - Nguy cơ)", f"{delta_sleep:.2f} giờ")
    m2.metric("Δ Trung vị giờ học (Tài năng - Nguy cơ)", f"{delta_study:.2f} giờ")
    m3.metric("Số lượng học sinh nhóm tài năng", f"{int(h['n']):,}")

    # Quick numeric summary (optional but useful)
    st.markdown("**Tóm tắt nhanh (trung vị / trung bình):**")
    summary = (
        df[list(cols_needed)]
        .dropna()
        .groupby("student_group")[["sleep_hours", "study_hours_daily"]]
        .agg(["median", "mean"])
    )

    # Đổi tên tầng 1 (tên biến)
    summary = summary.rename(columns={
        "sleep_hours": "Giờ ngủ", 
        "study_hours_daily": "Giờ học mỗi ngày"
    }, level=0)
    
    # Đổi tên tầng 2 (tên phép toán)
    summary = summary.rename(columns={
        "median": "Trung vị", 
        "mean": "Trung bình"
    }, level=1)

    # Đổi tên cột index
    summary.index.name = "Nhóm học sinh"

    st.dataframe(summary, use_container_width=True)

    st.markdown("### Đánh đổi trực tiếp (mật độ 2D): Ngủ vs Học")

    c_hm1, c_hm2 = st.columns([1, 2])
    with c_hm1:
        hm_group = st.selectbox(
            "Chọn nhóm để xem bản đồ nhiệt",
            ["Nhóm tài năng", "Nhóm thường", "Nhóm nguy cơ"],
            index=0,
        )
    with c_hm2:
        hm_sample = st.slider(
            "Số mẫu cho heatmap (giảm lag)", 50_000, 300_000, 100_000, step=50_000
        )

    df_hm = df[df["student_group"] == hm_group][
        ["sleep_hours", "study_hours_daily"]
    ].dropna()
    if len(df_hm) > hm_sample:
        df_hm = df_hm.sample(hm_sample, random_state=42)

    fig_hm = px.density_heatmap(
        df_hm,
        x="sleep_hours",
        y="study_hours_daily",
        nbinsx=40,
        nbinsy=40,
    )
    fig_hm.update_layout(
        title=f"Mật độ 2D: {hm_group} (Ngủ vs Học)",
        xaxis_title="Giờ ngủ",
        yaxis_title="Giờ học mỗi ngày",
    )
    st.plotly_chart(fig_hm, use_container_width=True)


# =========================
# TAB 2: Night owl
# =========================
with tab2:
    st.subheader("Mức độ căng thẳng giữa học và giải trí khi sử dụng màn hình")
    needed = {
        "mental_stress",
        "online_course_hours",
        "screen_time",
        "entertainment_screen_time",
    }
    if not needed.issubset(df.columns):
        st.error(f"Thiếu cột cần thiết: {needed - set(df.columns)}")
        st.stop()

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        bin_width = st.selectbox("Độ rộng bin (giờ)", [0.5, 1.0, 2.0], index=1)
    with c2:
        max_x = st.selectbox(
            "Giới hạn trục X (giờ)", [12, 16, 24, "Không giới hạn"], index=0
        )
        max_x_val = None if max_x == "Không giới hạn" else float(max_x)
    with c3:
        agg_mode = st.radio(
            "Tổng hợp", ["Mean + 95% CI", "Median + IQR"], horizontal=True, index=0
        )
        min_n = st.slider("Số mẫu tối thiểu / bin", 100, 5000, 1000, step=100)

    d2 = df[["mental_stress", "online_course_hours", "entertainment_screen_time"]].dropna()

    agg_key = "mean" if agg_mode.startswith("Mean") else "median"

    online = agg_band_by_bins(
        d2,
        "online_course_hours",
        "mental_stress",
        bin_width=bin_width,
        max_x=max_x_val,
        agg=agg_key,
    )
    ent = agg_band_by_bins(
        d2,
        "entertainment_screen_time",
        "mental_stress",
        bin_width=bin_width,
        max_x=max_x_val,
        agg=agg_key,
    )

    # lọc bin ít mẫu (giúp band cuối không “ảo”)
    online = online[online["n"] >= min_n]
    ent = ent[ent["n"] >= min_n]

    if online.empty and ent.empty:
        st.warning(
            "Không đủ dữ liệu hợp lệ để vẽ biểu đồ sau khi gom bin/lọc. "
            "Hãy thử tăng 'Giới hạn trục X' hoặc giảm 'Số mẫu tối thiểu / bin'."
        )
        st.stop()

    fig = go.Figure()

    def add_line_with_band(frame, name, band_name):
        # band
        fig.add_trace(
            go.Scatter(
                x=pd.concat([frame["bin_center"], frame["bin_center"][::-1]]),
                y=pd.concat([frame["hi"], frame["lo"][::-1]]),
                fill="toself",
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                opacity=0.25,
                name=band_name,
                hoverinfo="skip",
            )
        )
        # line (hover shows n)
        fig.add_trace(
            go.Scatter(
                x=frame["bin_center"],
                y=frame["y"],
                mode="lines",
                name=name,
                customdata=frame[["n", "lo", "hi"]],
                hovertemplate=(
                    "Giờ (tâm bin): %{x:.2f}<br>"
                    "Căng thẳng: %{y:.3f}<br>"
                    "Cỡ mẫu: %{customdata[0]:,.0f}<br>"
                    "Dải: [%{customdata[1]:.3f}, %{customdata[2]:.3f}]<extra></extra>"
                ),
            )
        )

    if not online.empty:
        add_line_with_band(online, "Học online (giờ học online)", "Dải (học online)")
    else:
        st.info("Không đủ dữ liệu để vẽ đường Học online (sau lọc min_n).")

    if not ent.empty:
        add_line_with_band(ent, "Giải trí (giờ màn hình giải trí)", "Dải (giải trí)")
    else:
        st.info("Không đủ dữ liệu để vẽ đường Giải trí (sau lọc min_n).")

    fig.update_layout(
        title=f"Căng thẳng theo thời gian màn hình (gom bin) – {agg_mode}",
        xaxis_title="Giờ",
        yaxis_title="Mức độ căng thẳng",
        legend_title="Loại thời gian màn hình",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "- `thời_gian_màn_hình_giải_trí = max(screen_time - online_course_hours, 0)`\n"
        "- Nếu đường **giải trí** dốc hơn → stress tăng nhanh hơn khi tăng thời gian giải trí (trong dataset)."
    )