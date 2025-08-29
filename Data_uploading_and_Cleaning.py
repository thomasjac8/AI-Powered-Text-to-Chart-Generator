# =========================
# pages/Data_uploading_and_Cleaning.py
# =========================
import os
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import pandas as pd
import numpy as np
import sqlite3
import chardet
from io import BytesIO
import tempfile
import re
import streamlit.config as config

# ✅ Set 2 GB upload limit BEFORE importing streamlit
os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '2048'  # MB
st.config.set_option("server.maxUploadSize", 2048)



# ==============================================================
# FIX CLIENT-SIDE UPLOAD LIMIT (200MB)
# ==============================================================
# Monkey patch to increase the client-side file size limit
def _get_upload_file_size_limit():
    return 2048  # 2GB in MB



# Apply the patch if the method exists
if hasattr(UploadedFile, '_get_upload_file_size_limit'):
    UploadedFile._get_upload_file_size_limit = staticmethod(_get_upload_file_size_limit)
else:
    # For newer versions of Streamlit, we might need a different approach
    try:
        from streamlit.runtime.uploaded_file_manager import UploadedFileManager
        UploadedFileManager._get_max_file_size = staticmethod(lambda: 2048 * 1024 * 1024)  # 2GB in bytes
    except:
        st.warning("Could not patch upload limit. Large files might not work.")

DB_NAME = "text_to_chart.db"

# ---------- Database Setup ----------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS datasets
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  data BLOB,
                  upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

# ==============================================================
# Calculated Field Helpers (Case-Insensitive)
# ==============================================================

def parse_calculated_field(expression, df):
    """
    Parse and evaluate calculated field expressions like "2xProfit = Profit * 2"
    Returns the new column name and the calculated values
    Case-insensitive column matching.
    """
    try:
        # Split on '=' to separate column name from expression
        if '=' in expression:
            parts = expression.split('=', 1)
            new_col_name = parts[0].strip()
            formula = parts[1].strip()
        else:
            # If no '=' found, use a default name
            new_col_name = f"calculated_{hash(expression) % 10000}"
            formula = expression.strip()
        
        # Clean up the column name (remove non-alphanumeric chars except underscore)
        new_col_name = re.sub(r'[^a-zA-Z0-9_]', '', new_col_name)
        if not new_col_name or new_col_name[0].isdigit():
            new_col_name = f"calc_{new_col_name}"
        
        # Build a lowercase mapping of column names
        col_map = {c.lower(): c for c in df.columns}
        
        # Extract all possible tokens (potential column names)
        possible_columns = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', formula)
        
        # Replace tokens in formula with df['ColumnName'] using case-insensitive matching
        for token in possible_columns:
            token_lower = token.lower()
            if token_lower in col_map and col_map[token_lower] != new_col_name:
                actual_col = col_map[token_lower]
                formula = re.sub(rf'\b{token}\b', f"df['{actual_col}']", formula)
        
        # Evaluate the formula safely
        allowed_globals = {'df': df, 'np': np, 'pd': pd}
        allowed_locals = {}
        
        # Add common math functions
        math_functions = ['sin', 'cos', 'tan', 'log', 'log10', 'exp', 'sqrt', 'abs']
        for func in math_functions:
            allowed_globals[func] = getattr(np, func, None)
        
        # Evaluate the expression
        result = eval(formula, allowed_globals, allowed_locals)
        
        return new_col_name, result
        
    except Exception as e:
        raise ValueError(f"Error parsing calculated field '{expression}': {e}")

def add_calculated_fields(df, calculated_fields):
    """
    Add calculated fields to the dataframe
    calculated_fields: list of expressions like ["2xProfit = Profit * 2", "Ratio = Sales / Expenses"]
    Case-insensitive column matching.
    """
    if not calculated_fields:
        return df
    
    df_copy = df.copy()
    
    for expression in calculated_fields:
        if expression.strip():
            try:
                col_name, values = parse_calculated_field(expression, df_copy)
                df_copy[col_name] = values
                st.success(f"✅ Added calculated column: '{col_name}' from expression: '{expression}'")
            except Exception as e:
                st.error(f"❌ Failed to add calculated field '{expression}': {e}")
    
    return df_copy


# ---------- Helpers ----------
def detect_encoding(file_content: bytes):
    """Detect encoding of uploaded file bytes."""
    result = chardet.detect(file_content)
    return result.get('encoding') or 'utf-8'

def read_csv_like(file_content: bytes, sep=None, encoding=None):
    """Read CSV/TSV/TXT with fallback encodings."""
    try:
        return pd.read_csv(BytesIO(file_content), encoding=encoding, sep=sep, engine='python')
    except UnicodeDecodeError:
        for enc in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']:
            try:
                return pd.read_csv(BytesIO(file_content), encoding=enc, sep=sep, engine='python')
            except UnicodeDecodeError:
                continue
    return None

def save_dataframe_to_db(df, filename):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    buffer = BytesIO()
    df.to_pickle(buffer)
    buffer.seek(0)
    c.execute("INSERT INTO datasets (name, data) VALUES (?, ?)", (filename, buffer.read()))
    conn.commit()
    dataset_id = c.lastrowid
    conn.close()
    return dataset_id

def update_dataframe_in_db(df, dataset_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    buffer = BytesIO()
    df.to_pickle(buffer)
    buffer.seek(0)
    c.execute("UPDATE datasets SET data = ? WHERE id = ?", (buffer.read(), dataset_id))
    conn.commit()
    conn.close()

# ---------- Page Config ----------
st.set_page_config(page_title="Data Uploading & Cleaning", layout="wide")

st.markdown("""
    <style>
    /* Completely hide the "Limit 200MB per file" message */
    .stFileUploader small {
        visibility: hidden !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("📂 Upload files **up to 2GB** (CSV, XLSX, JSON, PARQUET, TXT, TSV)")
uploaded_file = st.file_uploader("Choose a file", type=["csv","xlsx","xls","json","parquet","txt","tsv"])

# State management
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None

if uploaded_file and uploaded_file.name != st.session_state.current_file_name:
    for k in ["dataset_id", "df", "original_df"]:
        st.session_state.pop(k, None)
    st.session_state.current_file_name = uploaded_file.name
    st.rerun()

if st.sidebar.button("🗑️ Clear Current Data"):
    for k in ["dataset_id", "df", "original_df", "current_file_name"]:
        st.session_state.pop(k, None)
    st.rerun()

# ---------- Load Data ----------
if uploaded_file and ("df" not in st.session_state):
    try:
        file_bytes = uploaded_file.getvalue()
        file_mb = len(file_bytes) / (1024 * 1024)
        if file_mb > 2048:
            st.error(f"❌ File size ({file_mb:.2f} MB) exceeds 2GB limit.")
            st.stop()

        ext = uploaded_file.name.split(".")[-1].lower()
        df = None

        if ext in ["xlsx", "xls"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                df = pd.read_excel(tmp_path)
            finally:
                os.unlink(tmp_path)

        elif ext in ["csv", "tsv", "txt"]:
            enc = detect_encoding(file_bytes)
            sep = "\t" if ext == "tsv" else None
            df = read_csv_like(file_bytes, sep=sep, encoding=enc)

        elif ext == "json":
            try:
                df = pd.read_json(BytesIO(file_bytes), lines=True)
            except ValueError:
                df = pd.read_json(BytesIO(file_bytes))

        elif ext == "parquet":
            df = pd.read_parquet(BytesIO(file_bytes))

        dataset_id = save_dataframe_to_db(df, uploaded_file.name)
        st.session_state["original_df"] = df.copy()
        st.session_state["df"] = df.copy()
        st.session_state["dataset_id"] = dataset_id
        st.success(f"✅ Loaded {uploaded_file.name} with {len(df)} rows × {len(df.columns)} columns")

    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.stop()

# ---------- Cleaning UI ----------
if "df" in st.session_state:
    df = st.session_state["df"]

    # Preview
    st.subheader("🔍 Data Preview (first 100 rows)")
    st.dataframe(df.head(100))

    # --- Download options ---
    st.sidebar.header("💾 Download Options")
    download_format = st.sidebar.selectbox("Download format", ["CSV", "Excel"])
    if st.sidebar.button("⬇️ Download Cleaned Data"):
        try:
            if download_format == "CSV":
                csv = df.to_csv(index=False).encode("utf-8")
                st.sidebar.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )
            else:
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False)
                st.sidebar.download_button(
                    label="Download Excel",
                    data=buffer.getvalue(),
                    file_name="cleaned_data.xlsx",
                    mime="application/vnd.ms-excel"
                )
        except Exception as e:
            st.error(f"Download failed: {e}")

    # --- Calculated fields ---
    st.sidebar.header("🧮 Calculated Fields")
    num_calculated = st.sidebar.slider("Number of calculated fields", 0, 10, 0)
    calculated_expressions = []
    
    for i in range(num_calculated):
        expr = st.sidebar.text_input(f"Calculated Field {i+1}", 
                                    placeholder="E.g., 2xProfit = Profit * 2",
                                    key=f"calc_{i}")
        calculated_expressions.append(expr)
    
    # Apply calculated fields
    if calculated_expressions and any(expr.strip() for expr in calculated_expressions):
        df = add_calculated_fields(df, calculated_expressions)
        st.session_state["df"] = df
        update_dataframe_in_db(df, st.session_state["dataset_id"])
        st.success("✅ Calculated fields applied!")

    # --- Missing values ---
    missing_info = df.isnull().sum()
    missing_cols = missing_info[missing_info > 0]

    st.sidebar.header("🧩 Handle Missing Values")
    if not missing_cols.empty:
        col = st.sidebar.selectbox("Select column", missing_cols.index)
        method = st.sidebar.radio("Method", [
            "Drop rows", "Fill with Mean", "Fill with Median", "Fill with Mode", "Forward Fill", "Backward Fill"
        ])
        if st.sidebar.button("Apply"):
            if method == "Drop rows":
                df = df.dropna(subset=[col])
            elif method == "Fill with Mean":
                df[col] = df[col].fillna(df[col].mean())
            elif method == "Fill with Median":
                df[col] = df[col].fillna(df[col].median())
            elif method == "Fill with Mode":
                df[col] = df[col].fillna(df[col].mode()[0])
            elif method == "Forward Fill":
                df[col] = df[col].fillna(method="ffill")
            elif method == "Backward Fill":
                df[col] = df[col].fillna(method="bfill")
            st.session_state["df"] = df
            update_dataframe_in_db(df, st.session_state["dataset_id"])
            st.success(f"Handled missing values in `{col}` using `{method}`")
    else:
        st.sidebar.info("No missing values found.")

    # --- Remove characters ---
    st.sidebar.header("🧹 Remove Characters")
    clean_col = st.sidebar.selectbox("Column to Clean", df.columns)
    chars = st.sidebar.text_input("Characters to Remove")
    if st.sidebar.button("Clean Column"):
        try:
            df[clean_col] = df[clean_col].astype(str).str.replace(chars, "", regex=False)
            st.session_state["df"] = df
            update_dataframe_in_db(df, st.session_state["dataset_id"])
            st.success(f"Removed `{chars}` from `{clean_col}`")
        except Exception as e:
            st.error(f"Error: {e}")

    # --- Convert data types ---
    st.sidebar.header("🔁 Convert Data Types")
    dtype_col = st.sidebar.selectbox("Column to Convert", df.columns)
    dtype = st.sidebar.selectbox("New Type", ["int", "float", "str", "datetime"])
    if st.sidebar.button("Convert Type"):
        try:
            if dtype == "datetime":
                df[dtype_col] = pd.to_datetime(df[dtype_col], errors="coerce")
            else:
                df[dtype_col] = df[dtype_col].astype(dtype)
            st.session_state["df"] = df
            update_dataframe_in_db(df, st.session_state["dataset_id"])
            st.success(f"Converted `{dtype_col}` to {dtype}")
        except Exception as e:
            st.error(f"Conversion failed: {e}")

    # --- Replace values ---
    st.sidebar.header("🪄 Replace Values")
    rcol = st.sidebar.selectbox("Column", df.columns)
    to_replace = st.sidebar.text_input("Value to Replace")
    replace_with = st.sidebar.text_input("Replace With")
    if st.sidebar.button("Replace"):
        df[rcol] = df[rcol].replace(to_replace, replace_with)
        st.session_state["df"] = df
        update_dataframe_in_db(df, st.session_state["dataset_id"])
        st.success(f"Replaced `{to_replace}` with `{replace_with}` in `{rcol}`")

    # --- Extract date parts ---
    st.sidebar.header("📅 Extract Date Parts")
    date_col = st.sidebar.selectbox("Date Column", df.columns)
    if st.sidebar.button("Extract Parts"):
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df["Year"] = df[date_col].dt.year
            df["Month"] = df[date_col].dt.month
            df["Day"] = df[date_col].dt.day
            df["Quarter"] = df[date_col].dt.quarter
            df["DayOfWeek"] = df[date_col].dt.dayofweek
            df["DayName"] = df[date_col].dt.day_name()
            df["MonthName"] = df[date_col].dt.month_name()
            st.session_state["df"] = df
            update_dataframe_in_db(df, st.session_state["dataset_id"])
            st.success(f"Extracted date parts from `{date_col}`")
        except Exception as e:
            st.error(f"Error extracting: {e}")

    # --- Columns list ---
    st.subheader("🧭 Columns Available")
    cols_per_row = 4
    columns = df.columns.tolist()
    for i in range(0, len(columns), cols_per_row):
        row = st.columns(cols_per_row)
        for j, col in enumerate(columns[i:i+cols_per_row]):
            with row[j]:
                st.text(f"• {col}")

    # --- Next Page Button ---
    if st.button("➡ Go to Chart Creation"):
        update_dataframe_in_db(df, st.session_state["dataset_id"])
        st.switch_page("pages/Text_to_Chart.py")

else:
    st.info("📥 Upload a dataset to get started (up to 2GB).")