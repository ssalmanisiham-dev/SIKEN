import streamlit as st
import pandas as pd
import time
import os

from Isolation_forest.isolation_forest import (
    isolation_forest,
    anomaly_score
)

# =========================
# DATASET LOADING
# =========================

st.set_page_config(
    page_title="Isolation Forest Dashboard",
    layout="wide"
)

current_script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_script_dir, "data", "train.csv")


@st.cache_data
def load_dataset(path):

    df = pd.read_csv(path)

    df = df.rename(columns={
        "Packet Length Mean": "Packet Length",
        "Label": "Anomaly Scores"
    })

    df["Packet Length"] = pd.to_numeric(
        df["Packet Length"],
        errors="coerce"
    )

    df["Anomaly Scores"] = df["Anomaly Scores"].apply(
        lambda x: 0 if str(x).upper() == "BENIGN" else 1
    )

    df = df.dropna(
        subset=["Packet Length", "Anomaly Scores"]
    )

    return df


dataset = load_dataset(csv_path)

# Bach ma yb9ach dashboard bti2
dataset = dataset.head(1000)

# =========================
# TRAIN ISOLATION FOREST
# =========================

@st.cache_resource
def train_forest(data):

    train_df = data[
        ["Packet Length", "Anomaly Scores"]
    ].copy()

    train_df.columns = ["feat1", "feat2"]

    forest = isolation_forest(
        train_df,
        n_trees=20,
        max_depth=10,
        subspace=128
    )

    return forest


iForest = train_forest(dataset)

# =========================
# SESSION STATE
# =========================

if "row_idx" not in st.session_state:
    st.session_state.row_idx = 0

if "is_running" not in st.session_state:
    st.session_state.is_running = False

# =========================
# SIDEBAR
# =========================

st.sidebar.title("Simulation")

if st.sidebar.button("▶ Start"):
    st.session_state.is_running = True

if st.sidebar.button("⏸ Pause"):
    st.session_state.is_running = False

if st.sidebar.button("🔄 Reset"):
    st.session_state.row_idx = 0
    st.session_state.is_running = False
    st.rerun()

speed = st.sidebar.slider(
    "Speed",
    0.05,
    1.0,
    0.2
)

# =========================
# HEADER
# =========================

st.title("Isolation Forest Cyber Dashboard")

col1, col2, col3 = st.columns(3)

metric_packets = col1.empty()
metric_score = col2.empty()
metric_outlier = col3.empty()

chart1 = st.empty()
chart2 = st.empty()

table_placeholder = st.empty()

# =========================
# SIMULATION
# =========================

history = []

if st.session_state.is_running:

    while (
        st.session_state.row_idx < len(dataset)
        and st.session_state.is_running
    ):

        idx = st.session_state.row_idx

        row = dataset.iloc[idx]

        packet_length = float(
            row["Packet Length"]
        )

        label = float(
            row["Anomaly Scores"]
        )

        point = pd.DataFrame(
            [{
                "feat1": packet_length,
                "feat2": label
            }]
        )

        score = anomaly_score(
            point,
            iForest,
            128
        )

        outlier = score > 0.6

        history.append({
            "Packet Length": packet_length,
            "Label": label,
            "Isolation Score": score,
            "Outlier": outlier
        })

        live_df = pd.DataFrame(history)

        metric_packets.metric(
            "Processed Packets",
            len(live_df)
        )

        metric_score.metric(
            "Current Score",
            f"{score:.4f}"
        )

        metric_outlier.metric(
            "Outlier",
            "YES" if outlier else "NO"
        )

        chart1.scatter_chart(
            live_df,
            x="Packet Length",
            y="Isolation Score"
        )

        chart2.bar_chart(
            live_df["Outlier"].value_counts()
        )

        table_placeholder.dataframe(
            live_df.tail(20),
            use_container_width=True
        )

        st.session_state.row_idx += 1

        time.sleep(speed)

else:

    st.info(
        "Press Start to launch the simulation."
    )