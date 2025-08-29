# =========================
# home.py
# =========================
# Set 2 GB upload limit BEFORE importing streamlit
import os
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "2048"  # in MB

import streamlit as st

# ---------- Page config ----------
st.set_page_config(
    page_title="Text to Chart App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.config.set_option("server.maxUploadSize", 2048)

# ---------- Header ----------
st.markdown(
    """
    <div style="padding:16px 18px;border-radius:16px;
                background:linear-gradient(90deg,#60A5FA,#A78BFA);
                color:#0b1021;font-weight:700;">
      <div style="font-size:24px;">📊 Text to Chart App</div>
      <div style="font-size:13px;font-weight:600;opacity:.9;">
        Clean data • Build charts • Handle files up to <b>2GB</b> (CSV, XLSX/XLS, JSON, Parquet, TXT, TSV)
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")  # spacer

# ---------- Intro ----------
st.markdown(
    """
    This app helps you:
    - 🧼 **Upload & clean** your dataset
    - 📈 **Generate charts** (interactive) from your data
    - 💾 Work with **large files (up to 2GB)**
    - 🧮 Add **calculated fields** and handle missing values
    """
)

st.markdown("---")

# ---------- Quick Start ----------
left, right = st.columns([1, 1])
with left:
    st.subheader("🚀 Get Started")
    st.markdown("Begin by uploading a dataset on the Cleaning page, then jump to Chart Creation.")

    go_upload = st.button("➡ Go to Data Uploading & Cleaning", use_container_width=True)
    go_chart = st.button("📈 Go to Chart Creation", use_container_width=True)

    if go_upload:
        # Prefer st.switch_page when available
        if hasattr(st, "switch_page"):
            st.switch_page("pages/Data_uploading_and_Cleaning.py")
        else:
            st.error("Your Streamlit version doesn't support st.switch_page. Use the sidebar links.")

    if go_chart:
        if hasattr(st, "switch_page"):
            st.switch_page("pages/Text_to_Chart.py")
        else:
            st.error("Your Streamlit version doesn't support st.switch_page. Use the sidebar links.")

with right:
    st.subheader("ℹ️ Tips")
    st.markdown(
        """
        - For **TSV**, ensure the file uses tab separators.
        - **JSON**: supports both standard and JSON Lines (`lines=True`).
        - **Parquet** requires `pyarrow` or `fastparquet`.
        - Keep this home page open while navigating via the **sidebar** or the buttons above.
        """
    )

st.markdown("---")

# ---------- Sidebar ----------
st.sidebar.header("📚 Navigation")
# Newer Streamlit has page links; they work even without switch_page
try:
    st.sidebar.page_link("pages/Data_uploading_and_Cleaning.py", label="Data Uploading & Cleaning")
    st.sidebar.page_link("pages/Text_to_Chart.py", label="Text to Chart")
except Exception:
    st.sidebar.write("• Data Uploading & Cleaning\n• Text to Chart")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ App Settings")
st.sidebar.caption("Upload limit is set globally to 2GB for this app run.")
st.sidebar.code('os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "2048"', language="python")

# ---------- Footer ----------
st.markdown(
    """
    <div style="margin-top:24px;font-size:12px;opacity:.7;">
      Need help? Start at <b>Data Uploading & Cleaning</b>, then move to <b>Text to Chart</b>.
    </div>
    """,
    unsafe_allow_html=True,
)