from __future__ import annotations

import io
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
