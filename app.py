from __future__ import annotations

import io
import json
import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="SafeSynth MVP", layout="wide")
st.title("🔐 SafeSynth – Privacy-Preserving Synthetic Data Platform")
st.caption("Train ML models without exposing sensitive financial records.")

for key, default in {
    "file_id": None,
    "task_id": None,
    "real_df": None,
    "synthetic_df": None,
    "report": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.container(border=True):
    st.subheader("1) Upload Dataset")
    uploaded_file = st.file_uploader("Upload real financial CSV", type=["csv"])

    if uploaded_file is not None:
        st.session_state.real_df = pd.read_csv(uploaded_file)
        st.dataframe(st.session_state.real_df.head(), use_container_width=True)

    if uploaded_file is not None and st.button("Upload to SafeSynth API", use_container_width=True):
        uploaded_file.seek(0)
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=60)
        if resp.ok:
            payload = resp.json()
            st.session_state.file_id = payload["file_id"]
            st.success(f"Uploaded successfully. file_id={st.session_state.file_id}")
        else:
            st.error(resp.text)

with st.container(border=True):
    st.subheader("2) Training Status")
    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.file_id is not None and st.button("Start CTGAN Training", use_container_width=True):
            resp = requests.post(f"{BACKEND_URL}/train/{st.session_state.file_id}", timeout=60)
            if resp.ok:
                st.session_state.task_id = resp.json()["task_id"]
                st.info(f"Training task created: task_id={st.session_state.task_id}")
            else:
                st.error(resp.text)

    with col2:
        if st.session_state.task_id is not None and st.button("Refresh Task", use_container_width=True):
            status_resp = requests.get(f"{BACKEND_URL}/task/{st.session_state.task_id}", timeout=60)
            if status_resp.ok:
                task_payload = status_resp.json()
                st.write(f"Status: **{task_payload['status']}**")
                if task_payload.get("error_message"):
                    st.error(task_payload["error_message"])

                if task_payload["status"] == "Completed":
                    report_resp = requests.get(f"{BACKEND_URL}/report/{st.session_state.task_id}", timeout=60)
                    if report_resp.ok:
                        st.session_state.report = report_resp.json()

                    dl_resp = requests.get(f"{BACKEND_URL}/download/{st.session_state.task_id}", timeout=120)
                    if dl_resp.ok:
                        st.session_state.synthetic_df = pd.read_csv(io.BytesIO(dl_resp.content))
                        st.download_button(
                            "Download Synthetic CSV",
                            data=dl_resp.content,
                            file_name=f"synthetic_{st.session_state.task_id}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
            else:
                st.error(status_resp.text)

if st.session_state.report is not None:
    with st.container(border=True):
        st.subheader("3) Results Dashboard")
        report = st.session_state.report

        k1, k2, k3 = st.columns(3)
        k1.metric("Overall Quality", f"{report['overall_score']}/100")
        k2.metric("Quality Label", report["quality_label"])
        k3.metric("Privacy Risk", report["privacy_risk"])
        st.success(report["summary"])

        st.markdown("#### Insights")
        for insight in report.get("insights", []):
            st.write(f"- {insight}")

        st.markdown("#### Column-wise Fidelity")
        st.dataframe(pd.DataFrame(report.get("metrics", [])), use_container_width=True)

if st.session_state.real_df is not None and st.session_state.synthetic_df is not None:
    with st.container(border=True):
        st.subheader("4) Distribution Comparison")
        numeric_cols = [
            col
            for col in st.session_state.real_df.select_dtypes(include="number").columns
            if col in st.session_state.synthetic_df.columns
        ]
        if numeric_cols:
            default_col = "Transaction Amount" if "Transaction Amount" in numeric_cols else numeric_cols[0]
            selected = st.selectbox("Choose numeric column", numeric_cols, index=numeric_cols.index(default_col))

            plot_df = pd.concat(
                [
                    pd.DataFrame({"value": st.session_state.real_df[selected], "dataset": "Real"}),
                    pd.DataFrame({"value": st.session_state.synthetic_df[selected], "dataset": "Synthetic"}),
                ],
                ignore_index=True,
            )
            fig = px.histogram(plot_df, x="value", color="dataset", barmode="overlay", nbins=40)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No common numeric columns available for charting.")
st.set_page_config(page_title="SafeSynth", layout="wide")
st.title("SafeSynth: Privacy-First Synthetic Financial Data")

if "file_id" not in st.session_state:
    st.session_state.file_id = None
if "task_id" not in st.session_state:
    st.session_state.task_id = None
if "real_df" not in st.session_state:
    st.session_state.real_df = None
if "synthetic_df" not in st.session_state:
    st.session_state.synthetic_df = None

uploaded_file = st.file_uploader("Upload a real financial CSV", type=["csv"])

if uploaded_file is not None:
    st.session_state.real_df = pd.read_csv(uploaded_file)
    st.write("Preview of real data:")
    st.dataframe(st.session_state.real_df.head())

    if st.button("Upload CSV to backend"):
        uploaded_file.seek(0)
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=60)
        if response.ok:
            payload = response.json()
            st.session_state.file_id = payload["file_id"]
            st.success(f"Uploaded as file_id={st.session_state.file_id}")
        else:
            st.error(f"Upload failed: {response.text}")

if st.session_state.file_id is not None:
    if st.button("Start Training"):
        response = requests.post(f"{BACKEND_URL}/train/{st.session_state.file_id}", timeout=60)
        if response.ok:
            payload = response.json()
            st.session_state.task_id = payload["task_id"]
            st.info(f"Training started. task_id={st.session_state.task_id}")
        else:
            st.error(f"Train request failed: {response.text}")

if st.session_state.task_id is not None:
    st.subheader("Task Monitoring")
    if st.button("Refresh Status"):
        response = requests.get(f"{BACKEND_URL}/task/{st.session_state.task_id}", timeout=60)
        if response.ok:
            payload = response.json()
            st.write(payload)

            if payload["status"] == "Completed":
                download_response = requests.get(
                    f"{BACKEND_URL}/download/{st.session_state.task_id}", timeout=120
                )
                if download_response.ok:
                    st.session_state.synthetic_df = pd.read_csv(io.BytesIO(download_response.content))
                    st.success("Synthetic data downloaded.")
                    st.download_button(
                        "Download Synthetic CSV",
                        data=download_response.content,
                        file_name=f"synthetic_{st.session_state.task_id}.csv",
                        mime="text/csv",
                    )
                else:
                    st.error(f"Download failed: {download_response.text}")

            if payload.get("fidelity_report"):
                report = json.loads(payload["fidelity_report"])
                st.subheader("Fidelity Check: Means Comparison")
                st.json(report)
        else:
            st.error(f"Status check failed: {response.text}")

if st.session_state.real_df is not None and st.session_state.synthetic_df is not None:
    st.subheader("Distribution Comparison")
    common_numeric_cols = [
        col
        for col in st.session_state.real_df.select_dtypes(include="number").columns
        if col in st.session_state.synthetic_df.columns
    ]

    if common_numeric_cols:
        default_col = "Transaction Amount" if "Transaction Amount" in common_numeric_cols else common_numeric_cols[0]
        selected_col = st.selectbox("Select numeric column", common_numeric_cols, index=common_numeric_cols.index(default_col))

        real_plot_df = pd.DataFrame(
            {
                "value": st.session_state.real_df[selected_col],
                "dataset": "Real",
            }
        )
        synthetic_plot_df = pd.DataFrame(
            {
                "value": st.session_state.synthetic_df[selected_col],
                "dataset": "Synthetic",
            }
        )
        combined = pd.concat([real_plot_df, synthetic_plot_df], ignore_index=True)
        fig = px.histogram(combined, x="value", color="dataset", barmode="overlay", nbins=40)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No common numeric columns available for distribution comparison.")
