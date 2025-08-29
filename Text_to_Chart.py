# =========================
# pages/Text_to_Chart.py
# =========================
import os
# ✅ Set 2 GB upload limit BEFORE importing Streamlit
os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '2048'

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import plotly.express as px
import plotly.graph_objects as go
import re
from rapidfuzz import process, fuzz
import io
from io import BytesIO
import sqlite3
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import tempfile
import scipy.io.wavfile as wav  # For Whisper implementation
import streamlit.components.v1 as components
import requests

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

# ==============================================================
# Whisper Offline Implementation
# ==============================================================
@st.cache_resource
def load_whisper_model_offline():
    """Load Whisper model for offline use"""
    try:
        import whisper
        # Use smaller models for faster performance
        # Options: 'tiny', 'base', 'small', 'medium', 'large'
        return whisper.load_model("base")
    except ImportError:
        st.error("Whisper not installed. Run: pip install openai-whisper")
        return None
    except Exception as e:
        st.error(f"Failed to load Whisper model: {e}")
        return None

def whisper_offline_speech_to_text(duration=5):
    """Offline speech recognition using Whisper"""
    try:
        model = load_whisper_model_offline()
        if not model:
            return None
            
        st.info("🎤 Listening... Speak now!")
        samplerate = 16000
        
        # Record audio
        audio_data = sd.rec(int(duration * samplerate), 
                           samplerate=samplerate, 
                           channels=1, 
                           dtype='float32')
        sd.wait()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            wav.write(tmpfile.name, samplerate, audio_data)
            
            # Transcribe with Whisper
            result = model.transcribe(tmpfile.name)
            
            # Clean up
            os.unlink(tmpfile.name)
            
            return result["text"]
            
    except Exception as e:
        st.error(f"Whisper recognition failed: {e}")
        return None

# ==============================================================
# Database Helpers
# ==============================================================
def get_db_connection():
    return sqlite3.connect("text_to_chart.db")

def get_dataframe_from_db(dataset_id):
    """Retrieve dataframe from database by ID."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT data FROM datasets WHERE id = ?", (dataset_id,))
    result = c.fetchone()
    conn.close()
    
    if result:
        buffer = BytesIO(result[0])
        buffer.seek(0)
        try:
            return pd.read_pickle(buffer)
        except Exception as e:
            st.error(f"Error loading dataframe from database: {e}")
            return None
    return None

# ==============================================================
# Model Loaders
# ==============================================================
@st.cache_resource
def load_model():
    """Load chart-generation T5 model."""
    model = T5ForConditionalGeneration.from_pretrained(
        r"D:\Tinos apps\Text_to_Chart\flant5-text2chart-tuned"
    )
    tokenizer = T5Tokenizer.from_pretrained(
        r"D:\Tinos apps\Text_to_Chart\flant5-text2chart-tuned"
    )
    return model, tokenizer

@st.cache_resource
def load_filter_model():
    """Load filter-generation T5 model."""
    model = T5ForConditionalGeneration.from_pretrained(
        r"D:\Tinos apps\Text_to_Chart\flant5-filter-tuned"
    )
    tokenizer = T5Tokenizer.from_pretrained(
        r"D:\Tinos apps\Text_to_Chart\flant5-filter-tuned"
    )
    return model, tokenizer

# ==============================================================
# Model Utilities
# ==============================================================
def generate_chart_text(model, tokenizer, input_text):
    """Generate chart spec text from natural language."""
    model.eval()
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids.to(model.device)
    outputs = model.generate(inputs, max_length=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_filter_from_text(model, tokenizer, prompt: str, max_length: int = 64) -> dict:
    """Generate filter dict (column, operator, value) from natural language."""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    match = re.match(r"column:\s*(.+?);\s*operator:\s*(.+?);\s*value:\s*(.+)", output_text)
    if not match:
        raise ValueError(f"Could not parse filter: {output_text}")

    column, operator, value = match.group(1).strip(), match.group(2).strip(), match.group(3).strip().strip('"').strip("'")
    if operator == "=":
        operator = "=="
    if operator.lower() == "between" and "-" in value:
        min_val, max_val = value.split("-")
        return {"column": column, "operator": "between", "value": (float(min_val.strip()), float(max_val.strip()))}
    return {"column": column, "operator": operator, "value": value}

# ==============================================================
# Data Helpers
# ==============================================================
def match_column_name(input_col, df_columns, threshold=75):
    """Match a column name using multiple RapidFuzz scorers."""
    # First, try exact match (especially important for calculated fields)
    if input_col in df_columns:
        return input_col
    
    # Then try fuzzy matching
    scorers = [fuzz.ratio, fuzz.partial_ratio, fuzz.token_sort_ratio, fuzz.token_set_ratio]
    best_match, best_score = None, 0
    for scorer in scorers:
        match, score, _ = process.extractOne(input_col, df_columns, scorer=scorer)
        if score > best_score:
            best_match, best_score = match, score
    return best_match if best_score >= threshold else None

def apply_filter_gen(df, filter_dict):
    """Apply filter dictionary to dataframe."""
    col = match_column_name(filter_dict.get('column'), df.columns)
    if not col:
        raise ValueError(f"No matching column found for '{filter_dict.get('column')}'")

    op, val = filter_dict.get('operator'), filter_dict.get('value')
    col_dtype = df[col].dtype

    def convert_value(v):
        if pd.api.types.is_numeric_dtype(col_dtype):
            return float(v)
        elif pd.api.types.is_datetime64_any_dtype(col_dtype):
            return pd.to_datetime(v)
        return str(v)

    if isinstance(val, (list, tuple)):
        val = [convert_value(v) for v in val]
    else:
        val = convert_value(val)

    if op == '>':
        return df[df[col] > val]
    elif op == '<':
        return df[df[col] < val]
    elif op == '>=':
        return df[df[col] >= val]
    elif op == '<=':
        return df[df[col] <= val]
    elif op in ['==', '=', 'equals']:
        return df[df[col] == val]
    elif op == '!=':
        return df[df[col] != val]
    elif op == 'contains':
        return df[df[col].astype(str).str.contains(str(val), case=False)]
    elif op == 'starts with':
        return df[df[col].astype(str).str.startswith(str(val))]
    elif op == 'ends with':
        return df[df[col].astype(str).str.endswith(str(val))]
    elif op in ['between', 'range']:
        if not (isinstance(val, list) and len(val) == 2):
            raise ValueError(f"'between' requires two values: {val}")
        lower, upper = sorted(val)
        return df[(df[col] >= lower) & (df[col] <= upper)]
    else:
        raise ValueError(f"Unsupported operator: {op}")

# ==============================================================
# Chart Creation (Full Logic Preserved)
# ==============================================================
# Parses the T5 chart output into a dictionary of parameters 
def create_chart(df, t5_output, filter_model=None, filter_tokenizer=None, chart_model=None, chart_tokenizer=None):
    chart_info = {}
    parts = [part.strip() for part in t5_output.split(",")]
    for part in parts:
        if ":" in part:
            k, v = part.split(":", 1)
            chart_info[k.strip()] = v.strip()

    x = match_column_name(chart_info.get("x", ""), df.columns)
    y = match_column_name(chart_info.get("y", ""), df.columns)
    group = match_column_name(chart_info.get("group", ""), df.columns)
    stack = match_column_name(chart_info.get("stack", ""), df.columns)
    size_col= match_column_name(chart_info.get("size", ""), df.columns)
    y1 = match_column_name(chart_info.get("y1", ""), df.columns)
    y2 = match_column_name(chart_info.get("y2", ""), df.columns)
    location = match_column_name(chart_info.get("location", ""), df.columns)
    color_y = match_column_name(chart_info.get("color", ""), df.columns)

    agg = chart_info.get("agg", "sum").lower()
    filter_text = chart_info.get("filter", "")
    chart_type = chart_info.get("chart", "")

    # Apply filter
    if filter_text and filter_model and filter_tokenizer:
        try:
            filter_dict = generate_filter_from_text(filter_model, filter_tokenizer,f"Extract filters: {filter_text}")
            df = apply_filter_gen(df, filter_dict)
            #st.success(f"Applied: `{filter_dict}`")
        except Exception as e:
            st.error(f"Filter failed: {e}")

    # KPI: skip aggregation
    if chart_type == "kpi" or chart_type == "gauge" :
        try:
            if agg in ["sum", "total"]:
                agg_value = df[y].sum()
            elif agg in ["mean", "average"]:
                agg_value = df[y].mean()
            elif agg == "count":
                agg_value = df[y].count()
            elif agg in ["min", "minimum"]:
                agg_value = df[y].min()
            elif agg in ["max", "maximum"]:
                agg_value = df[y].max()
            else:
                st.warning("Unknown aggregation, using sum.")
                agg_value = df[y].sum()

            # st.metric(label=y, value=round(agg_value, 2))

        except Exception as e:
            st.error(f"KPI calculation error: {e}")
            return
    elif chart_type=='box' or chart_type=='violin' or chart_type=='histogram' or chart_type=='correlation_heatmap' :
        agg_df=df
    elif chart_type=='choropleth':
        group_cols = [location]
        if agg in ["sum", 'total']:
            agg_df = df.groupby(group_cols)[color_y].sum().reset_index()
        elif agg in ['mean', 'average']:
            agg_df = df.groupby(group_cols)[color_y].mean().reset_index()
        elif agg == 'count':
            agg_df = df.groupby(group_cols)[color_y].count().reset_index()
        elif agg in ['min', 'minimum']:
            agg_df = df.groupby(group_cols)[color_y].min().reset_index()
        elif agg in ['max', 'maximum']:
            agg_df = df.groupby(group_cols)[color_y].max().reset_index()
        else:
            st.warning("Unknown aggregation, using sum.")
            agg_df = df.groupby(group_cols)[color_y].sum().reset_index()
    elif chart_type=='bubble':
        if  size_col == None:
            size_col=y
        bubble_group_cols = [x]
        try:
            if agg in ['sum', 'total']:
                agg_df = df.groupby(bubble_group_cols).agg({
                    y: 'sum',
                    size_col: 'sum'
                }).reset_index()
            elif agg in ['mean', 'average']:
                agg_df = df.groupby(bubble_group_cols).agg({
                    y: 'sum',
                    size_col: 'mean'
                }).reset_index()
            elif agg == 'count':
                agg_df = df.groupby(bubble_group_cols).agg({
                    y: 'sum',
                    size_col: 'count'
                }).reset_index()
            elif agg in ['min', 'minimum']:
                agg_df = df.groupby(bubble_group_cols).agg({
                    y: 'sum',
                    size_col: 'min'
                }).reset_index()
            elif agg in ['max', 'maximum']:
                agg_df = df.groupby(bubble_group_cols).agg({
                    y: 'sum',
                    size_col:'max'
                }).reset_index()
            else:
                st.warning("Unknown aggregation, using sum.")
                agg_df = df.groupby(bubble_group_cols).agg({
                          y: 'sum',
                          size_col: 'sum'
                          }).reset_index()
        except Exception as e:
            st.error(f"Aggregation error: {e}")
            return
    elif chart_type=='combo':
        combo_group_cols = [x]
        try:
            if agg in ['sum', 'total']:
                agg_df = df.groupby(combo_group_cols).agg({
                    y1: 'sum',
                    y2: 'sum'
                }).reset_index()
            elif agg in ['mean', 'average']:
                agg_df = df.groupby(combo_group_cols).agg({
                    y1: 'sum',
                    y2: 'mean'
                }).reset_index()
            elif agg == 'count':
                agg_df = df.groupby(combo_group_cols).agg({
                    y1: 'sum',
                    y2: 'count'
                }).reset_index()
            elif agg in ['min', 'minimum']:
                agg_df = df.groupby(combo_group_cols).agg({
                    y1: 'sum',
                    y2: 'min'
                }).reset_index()
            elif agg in ['max', 'maximum']:
                agg_df = df.groupby(combo_group_cols).agg({
                    y1: 'sum',
                    y2:'max'
                }).reset_index()
            else:
                st.warning("Unknown aggregation, using sum.")
                agg_df = df.groupby(combo_group_cols).agg({
                          y1: 'sum',
                          y2: 'sum'
                          }).reset_index()
        except Exception as e:
            st.error(f"Aggregation error: {e}")
            return
    else:
        # Aggregation for charts (excluding KPI)
        try:
            if chart_type == "grouped_bar" and group:
                group_cols = [x, group]
            elif chart_type == "stacked_bar" and stack:
                group_cols = [x, stack]
            elif chart_type == "multi_line" and group:
                group_cols = [x, group]
            elif chart_type == "stacked_area" and group:
                group_cols = [x, group]
            elif chart_type == "stacked_area" and stack:
                group_cols = [x, stack]

            else:
                group_cols = [x]

            if agg in ['sum', 'total']:
                agg_df = df.groupby(group_cols)[y].sum().reset_index()
            elif agg in ['mean', 'average']:
                agg_df = df.groupby(group_cols)[y].mean().reset_index()
            elif agg == 'count':
                agg_df = df.groupby(group_cols)[y].count().reset_index()
            elif agg in ['min', 'minimum']:
                agg_df = df.groupby(group_cols)[y].min().reset_index()
            elif agg in ['max', 'maximum']:
                agg_df = df.groupby(group_cols)[y].max().reset_index()
            else:
                st.warning("Unknown aggregation, using sum.")
                agg_df = df.groupby(group_cols)[y].sum().reset_index()

        except Exception as e:
            st.error(f"Aggregation error: {e}")
            return

    # Color picker
    color =  "#1f77b4" #st.color_picker("Pick color", "#1f77b4", key=f"color_{x}_{y}_{chart_type}_{group}_{stack}_{size_col}")

    try:
        if chart_type == "bar":
            fig = px.bar(agg_df, x=x, y=y, color_discrete_sequence=[color])
        elif chart_type == "horizontal_bar":
            fig= px.bar(agg_df, x=y, y=x, orientation="h", title="Horizontal Bar Chart")
        elif chart_type == "grouped_bar" and group:
            fig = px.bar(agg_df, x=x, y=y, color=group, barmode="group")
        elif chart_type == "stacked_bar":
            fig= px.bar(agg_df, x=x, y=y, color=stack, barmode="stack", title="Stacked Bar Chart")
        elif chart_type == "line":
            fig = px.line(agg_df, x=x, y=y, line_shape="linear")
        elif chart_type == "multi_line" and group:
            fig= px.line(agg_df, x=x, y=y, color=group,  title="Multi-Line Chart")
        elif chart_type == "step_line":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=agg_df[x],
                y=agg_df[y],
                mode='lines',
                line=dict(shape='hv'),  # "hv" = horizontal then vertical steps
                name="Step Line"
            ))
            fig.update_layout(title="Step Line Chart", xaxis_title=x, yaxis_title=y)

        elif chart_type == "area":
            fig = px.area(agg_df, x=x, y=y, color=group) if group else px.area(agg_df, x=x, y=y)
            fig.update_layout(title="Area Chart", xaxis_title=x, yaxis_title=y)

        elif chart_type == "stacked_area":
            fig = px.area(agg_df, x=x, y=y, color=stack or group , groupnorm="percent")
            fig.update_layout(title="Stacked Area Chart", xaxis_title=x, yaxis_title=y)
        elif chart_type == "histogram":
            fig = px.histogram(agg_df, x=y, color=group if group else None, nbins=30)
            fig.update_layout(title="Histogram", xaxis_title=x, yaxis_title="Count")

        elif chart_type == "box":
            fig = px.box(agg_df, x=x, y=y)
            fig.update_layout(title="Box Plot", xaxis_title=x, yaxis_title=y)

        elif chart_type == "violin":
            fig = px.violin(agg_df, x=x , y=y,  box=True)
            fig.update_layout(title="Violin Plot", xaxis_title=x, yaxis_title=y)

        elif chart_type == "pie":
            fig = px.pie(agg_df, names=x, values=y)
        elif chart_type == "donut":
            fig = px.pie(agg_df, names=x, values=y, hole=0.5)
        elif chart_type == "treemap":
            fig = px.treemap(agg_df, path=[x], values=y, title="Treemap Chart")
        elif chart_type == "waterfall":
            fig = go.Figure(go.Waterfall(
                x=agg_df[x],
                y=agg_df[y],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            fig.update_layout(
                title="Waterfall Chart",
                xaxis_title=x,
                yaxis_title=y
            )
        elif chart_type == "funnel":
            agg_df = agg_df.sort_values(by=y, ascending=False)
            total = agg_df[y].sum()
            stages=agg_df[x]
            values=agg_df[y]
            labels = [f"{round(v / total * 100, 2)}%" for stage, v in zip(stages, values)]
            n = len(stages)
            colors = [f'rgba(0, 0, 255, {opacity})' for opacity in [1.0 - 0.7 * (i / (n - 1)) for i in range(n)]]

            fig = go.Figure(go.Funnel(
                            y=stages,
                            x=values,
                            text=labels,
                            textposition="inside",
                            marker={"color": colors},
                            textfont=dict(size=36),
                            opacity=0.9
                        ))
            fig.update_layout(
                xaxis_title=x,
                yaxis_title=y
            )

        elif chart_type == "kpi":
            fig = go.Figure(go.Indicator(
                mode="number",
                value=agg_value,
                title={"text": f"KPI: {y}"},
                number={'font': {'size': 48}},
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
        # Gauge chart
        elif chart_type == "gauge":
            # User input for target/reference value
            target_value = st.number_input("Enter target value (reference for delta):", value=0.0)

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=agg_value,
                delta={'reference': target_value},
                gauge={
                    'axis': {'range': [None, df[y].max()]},
                    'bar': {'color': "mediumslateblue"},
                    'steps': [
                        {'range': [0, df[y].max() * 0.5], 'color': 'lightgray'},
                        {'range': [df[y].max() * 0.5, df[y].max()], 'color': 'gray'}
                    ],
                },
                title={'text': f"{y} Gauge"}
            ))

        elif chart_type == "combo":
            if not y1 or not y2:
                st.warning("Combo chart requires both y1 and y2.")
                return

            fig = go.Figure()

            # Bar chart (y1 - left axis)
            fig.add_trace(go.Bar(
                x=agg_df[x],
                y=agg_df[y1],
                name=y1,
                yaxis='y1'
            ))

            # Line chart (y2 - right axis)
            fig.add_trace(go.Scatter(
                x=agg_df[x],
                y=agg_df[y2],
                name=y2,
                yaxis='y2',
                mode='lines+markers',
                line=dict(color='orange')
            ))

            # Set layout with secondary y-axis
            fig.update_layout(
                title="Combo Chart",
                xaxis=dict(title=x),
                yaxis=dict(title=y1),  # Primary y-axis (left)
                yaxis2=dict(
                    title=y2,
                    overlaying='y',
                    side='right',
                    showgrid=False  # Optional: hides grid from second y-axis
                ),
                legend=dict(x=0.5, xanchor='center', y=1.1, orientation='h'),
                bargap=0.3,
                height=500
            )

        elif chart_type == "scatter":
            fig = px.scatter(agg_df, x=x, y=y)
            fig.update_layout(title="Scatter Plot", xaxis_title=x, yaxis_title=y)

        elif chart_type == "bubble":
            size_scaled = (agg_df[size_col] - agg_df[size_col].min()) / (
                        agg_df[size_col].max() - agg_df[size_col].min())
            agg_df["scaled_size"] = 10 + size_scaled * 90  # Bubble size between 10 and 100

            fig = px.scatter(agg_df, x=x, y=y, size=size_scaled, color=size_col if size_col else None, hover_name=x,size_max=100)
            fig.update_layout(title="Bubble Chart", xaxis_title=x, yaxis_title=y)
        elif chart_type == "correlation_heatmap":
            corr = agg_df.select_dtypes(include='number').corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", aspect="auto")
            fig.update_layout(title="Correlation Heatmap")

        elif chart_type == "choropleth":
            fig = px.choropleth(
                df,
                locations=location,  # e.g., 'country'
                locationmode="country names",  # or "ISO-3" based on your data
                color=color_y,  # e.g., 'value' or 'sales'
                hover_name=location,
                color_continuous_scale="Viridis",
                template="plotly_white"
            )

        else:
            st.error(f"Unsupported chart type: `{chart_type}`")
            return

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Chart rendering failed: {e}")
        
# ==============================================================
# Currency Converter Helper
# ==============================================================



CURRENCY_CODES = ["USD", "INR", "GBP", "JPY", "EUR"]

@st.cache_data
def get_exchange_rates(base="USD"):
    """
    Fetches real-time exchange rates from exchangerate.host
    Returns a dict of rates relative to the base currency.
    """
    url = f"https://api.exchangerate.host/latest?base={base}"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        return data.get("rates", {})
    except Exception as e:
        st.error(f"Failed to fetch currency rates: {e}")
        return {}

def convert_currency(df, currency_column, from_currency, to_currency, rates):
    """
    Convert values in a currency column from one currency to another.
    """
    if from_currency == to_currency:
        return df
    
    if not rates or to_currency not in rates:
        st.error("Currency rates unavailable.")
        return df
    
    factor = rates[to_currency]
    df[currency_column] = df[currency_column] * factor
    st.success(f"💱 Converted {currency_column} from {from_currency} to {to_currency}")
    return df


# ==============================================================
# Streamlit UI
# ==============================================================
st.set_page_config(page_title="Text2Chart Dashboard", layout="wide")

# Initialize session state for instructions if not exists
if "instructions" not in st.session_state:
    st.session_state.instructions = [""] * 25  # Initialize with empty strings

if "dataset_id" not in st.session_state:
    st.warning("⚠️ Please upload and clean data first.")
    st.stop()

df = get_dataframe_from_db(st.session_state["dataset_id"])
if df is None:
    st.error("No dataset found. Please re-upload your data.")
    st.stop()

# Header
st.markdown(
    """
    <div style="padding:14px;border-radius:14px;background:linear-gradient(90deg,#FBCFE8,#FDE68A);color:#111;font-weight:600;">
      <span style="font-size:22px;">📊 AI-Powered Chart Dashboard</span>
      <div style="font-size:13px;opacity:.85;">Use natural language or voice to generate charts</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Columns List
st.subheader("Available Columns")
cols_per_row = 4
columns = df.columns.tolist()
for i in range(0, len(columns), cols_per_row):
    col_group = columns[i:i+cols_per_row]
    cols = st.columns(cols_per_row)
    for j, col_name in enumerate(col_group):
        with cols[j]:
            st.text(f"• {col_name}")

st.subheader("Dataset Preview")
st.dataframe(df.head(10))

# Sidebar currency conversion
st.sidebar.markdown("### 💱 Currency Converter")

currency_col = st.sidebar.selectbox("Currency Column", ["None"] + df.columns.tolist())

from_currency = st.sidebar.selectbox("From", CURRENCY_CODES, index=0)
to_currency = st.sidebar.selectbox("To", CURRENCY_CODES, index=1)



if currency_col != "None":
    rates = get_exchange_rates(from_currency)
    if st.sidebar.button("Convert Currency"):
        df = convert_currency(df, currency_col, from_currency, to_currency, rates)


# Load Models
model, tokenizer = load_model()
filter_model, filter_tokenizer = load_filter_model()

# Sidebar Controls
st.sidebar.header("🎛 Chart Controls")
try:
    import pyaudio
    pyaudio_available = True
except ImportError:
    pyaudio_available = False
    st.sidebar.info("Voice input requires:\n```pip install pyaudio```")

num_charts = st.sidebar.slider("Number of charts", 1, 25, 3)
chart_instructions = []

for i in range(num_charts):
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"Chart {i+1}")
    col1, col2 = st.sidebar.columns([1, 4])
    with col1:
        if st.button("🎤", key=f"voice_{i}"):
            voice_text = speech_to_text()
            if voice_text:
                # Ensure we're storing a string, not some other object
                st.session_state[f"instruction_{i}"] = str(voice_text)
                st.session_state.instructions[i] = str(voice_text)
    with col2:
        # Use a unique key and ensure we're working with strings
        text_input = st.sidebar.text_input(
            f"Chart {i+1} Instruction", 
            value=st.session_state.get(f"instruction_{i}", ""),
            key=f"instruction_input_{i}", 
            label_visibility="collapsed"
        )
        # Store the instruction in both session state locations
        st.session_state[f"instruction_{i}"] = text_input
        st.session_state.instructions[i] = text_input
        chart_instructions.append(text_input)

# Filters
st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Global Filters")
num_filters = st.sidebar.slider("Number of filters", 0, 5, 2)
nl_filters = []
for i in range(num_filters):
    filter_text = st.sidebar.text_input(f"Filter {i+1}", key=f"nl_filter_{i}")
    nl_filters.append(filter_text)

parsed_filters, filtered_df = [], df.copy()
for i, f in enumerate(nl_filters):
    if f.strip():
        try:
            parsed = generate_filter_from_text(filter_model, filter_tokenizer, f"Extract filters: {f.strip()}")
            parsed_filters.append(parsed)
            filtered_df = apply_filter_gen(filtered_df, parsed)
            st.sidebar.success(f"Parsed: {parsed}")
        except Exception as e:
            st.sidebar.error(f"Failed to parse '{f}': {e}")

# Charts
st.subheader("📈 Generated Charts")
for idx, instruction in enumerate(chart_instructions):
    if instruction and instruction.strip():
        st.markdown(f"#### Chart {idx+1}: `{instruction}`")
        with st.spinner("Generating chart..."):
            try:
                t5_output = generate_chart_text(model, tokenizer, instruction)
                create_chart(filtered_df, t5_output, filter_model, filter_tokenizer)
            except Exception as e:
                st.error(f"Chart generation failed: {e}")