import ast
import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Telemetry Data", layout="wide")

# ──────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Fonts & background ── */
  html, body, [class*="css"] { font-family: Calibri, 'Segoe UI', sans-serif; }
  [data-testid="stAppViewContainer"] { background:#F4F6F9; }
  .block-container { padding-top:1rem !important; max-width:100% !important; padding-left:1rem !important; padding-right:1rem !important; }

  /* ── CURSOR FIX: all static markdown is non-selectable ── */
  [data-testid="stMarkdownContainer"] * {
    cursor: default !important;
    user-select: none !important;
    -webkit-user-select: none !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] { background:#1F3864; }
  [data-testid="stSidebar"] * { color:#D0DCF0 !important; }
  [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] { background:#2C7C7E !important; }
  [data-testid="stSidebar"] label { color:#A8C0D8 !important; font-size:11px !important; font-weight:700 !important; text-transform:uppercase; letter-spacing:0.6px; }
  [data-testid="stSidebar"] [data-testid="stSelectbox"] div { background:#162d52 !important; border-color:#2C7C7E !important; }

  /* ── KPI metric cards ── */
  [data-testid="metric-container"] {
    background:#ffffff;
    border:1px solid #D9DCE0;
    border-radius:8px;
    padding:16px 18px 12px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }
  [data-testid="metric-container"] label {
    font-size:10px !important; font-weight:700 !important;
    text-transform:uppercase; letter-spacing:0.7px;
    color:#5A6070 !important; font-family:Calibri,sans-serif !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size:28px !important; font-weight:700 !important;
    color:#1F3864 !important; font-family:Calibri,sans-serif !important;
    line-height:1.1;
  }
  [data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size:11px !important; color:#2C7C7E !important;
  }

  /* ── Expander ── */
  [data-testid="stExpander"] {
    background:#ffffff !important;
    border:1px solid #D9DCE0 !important;
    border-radius:8px !important;
    margin-bottom:6px;
    box-shadow:0 1px 3px rgba(0,0,0,0.04);
  }
  [data-testid="stExpander"] summary {
    font-size:13px !important; font-weight:700 !important;
    color:#1F3864 !important; padding:12px 16px !important;
  }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] {
    border:1px solid #D9DCE0 !important;
    border-radius:8px !important;
    overflow:hidden;
  }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# FILE PATHS
# ──────────────────────────────────────────────────────────────
LC_LOG_CSV_1        = "case_209301297 2(1).csv"
LC_LOG_CSV_2        = "case_209468636 1.csv"
CONFIG_PARQUET_PATH = "part-00002-e4896001-b40f-4ec4-8751-a5cf60d342b2.c000.snappy 1(1).parquet"
CONFIG_PARQUET_PATH_2 = 'part-00001-4d1feda6-6f0e-4637-a95b-75184175a150.c000.snappy.parquet'
CASE_DETAILS_PATH   = "case_details (1).csv"
PARTS_PARQUET_PATH = "100k_adgent_facing_ground_truth (1) 1.parquet"

# ──────────────────────────────────────────────────────────────
# HEALTH-CHECK THRESHOLDS
# ──────────────────────────────────────────────────────────────
BAD_PRIMARY_STATUS  = {"2", "3", "Degraded", "Error", "Critical", "Failed", "Unknown"}
BAD_PREDICTIVE_FAIL = {"1", "Smart Alert Present", "Predictive Failure"}
BAD_REDUNDANCY      = {"4", "Lost"}

DCIM_CLASSES = [
    "DCIM_PhysicalDiskView", "DCIM_MemoryView", "DCIM_PowerSupplyView",
    "DCIM_FanView", "DCIM_ControllerBatteryView",
]
CLASS_LABELS = {
    "DCIM_PhysicalDiskView":      "Physical Disk",
    "DCIM_MemoryView":            "Memory",
    "DCIM_PowerSupplyView":       "Power Supply",
    "DCIM_FanView":               "Fan",
    "DCIM_ControllerBatteryView": "Controller Battery",
}

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def parse_props(props):
    if props is None:
        return {}
    if isinstance(props, dict):
        return props
    try:
        return dict(ast.literal_eval(str(props)))
    except Exception:
        return {}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        props  = parse_props(row["properties"])
        objcls = row["objectclass"]
        ps     = str(props.get("PrimaryStatus",          "?"))
        pf     = str(props.get("PredictiveFailureState", "?"))
        red    = str(props.get("RedundancyStatus",       "?"))
        if objcls in DCIM_CLASSES:
            ps_bad  = ps  in BAD_PRIMARY_STATUS
            pf_bad  = pf  in BAD_PREDICTIVE_FAIL
            red_bad = red in BAD_REDUNDANCY
            ok_bad  = "BAD" if (ps_bad or pf_bad or red_bad) else "OK"
        else:
            ok_bad = "N/A"
        records.append({
            "ok_bad":                 ok_bad,
            "component":              CLASS_LABELS.get(objcls, objcls),
            "objectclass":            objcls,
            "system_id":              row.get("system_id",  ""),
            "FQDD":                   props.get("FQDD",         "?"),
            "SerialNumber":           props.get("SerialNumber", "?"),
            "PrimaryStatus":          ps,
            "PredictiveFailureState": pf,
            "RedundancyStatus":       red,
            "objectid":               row.get("objectid",   ""),
            "collectiontimestamp":    row.get("collectiontimestamp", ""),
            "payload_type":           row.get("payload_type", ""),
            "case_nbr":               row["case_nbr"],
            "case_id":                row.get("case_id", ""),
        })
    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────
# DATA LOADERS
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_log_data() -> pd.DataFrame:
    df = pd.concat(
        [pd.read_csv(LC_LOG_CSV_1), pd.read_csv(LC_LOG_CSV_2)],
        ignore_index=True
    ).drop_duplicates()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ["case_nbr", "messageid", "severity", "message"]:
        df[col] = df[col].astype(str).str.strip()
    return df


@st.cache_data
def load_config_data() -> pd.DataFrame:
    df = pd.concat(
        [pd.read_parquet(CONFIG_PARQUET_PATH), pd.read_parquet(CONFIG_PARQUET_PATH_2)],
        ignore_index=True
    )
    # drop_duplicates() fails when columns contain unhashable types (lists/dicts).
    # Dedupe only on scalar key columns that are guaranteed hashable.
    dedup_cols = [c for c in ["case_nbr", "objectid", "objectclass", "collectiontimestamp"]
                  if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols)
    df["case_nbr"] = df["case_nbr"].astype(str).str.strip()
    for col in ["case_crt_dts", "collectiontimestamp", "debi_etl_inst_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

@st.cache_data
def load_parts_data() -> pd.DataFrame:
    df = pd.read_parquet(PARTS_PARQUET_PATH)
    df["case_nbr"] = df["case_nbr"].astype(str).str.strip()
    return df

@st.cache_data
def load_case_details() -> pd.DataFrame:
    df = pd.read_csv(CASE_DETAILS_PATH)
    df["case_nbr"] = df["case_nbr"].astype(str).str.strip()
    # Parse recommendation_values — numpy-array-like string e.g. "['HARD DRIVE' 'T8060']"
    def _parse_rec(val):
        try:
            vals = re.findall(r"'([^']+)'", str(val))
            hw_labels = [v for v in vals if re.fullmatch(r'[A-Za-z][A-Za-z ]+', v)]
            return ", ".join(hw_labels) if hw_labels else "—"
        except Exception:
            return "—"
    df["case_resolution"] = df["recommendation_values"].apply(_parse_rec)
    return df



df_logs        = load_log_data()
df_config      = load_config_data()
df_case_detail = load_case_details()
df_parts       = load_parts_data()        

all_cases = sorted(
    set(df_logs["case_nbr"].dropna().unique()) |
    set(df_config["case_nbr"].dropna().unique())
)

def build_parts_resolution(case_nbr: str, df_parts: pd.DataFrame) -> str:
    row = df_parts[df_parts["case_nbr"] == case_nbr]
    if row.empty:
        return "—"
    try:
        parts = row.iloc[0]["part_comdty_nm"]   
        qtys  = row.iloc[0]["itm_qty"]     

        # Normalise — could be a list, numpy array, or string representation
        if isinstance(parts, str):
            parts = re.findall(r"'([^']+)'", parts)
        if isinstance(qtys, str):
            qtys = [int(x) for x in re.findall(r'\d+', qtys)]

        parts = list(parts)
        qtys  = [int(q) for q in qtys]

        labels = []
        for part, qty in zip(parts, qtys):
            part = part.strip()
            if not part:
                continue
            labels.append(f"{part} ×{qty}" if qty > 1 else part)

        return ", ".join(labels) if labels else "—"
    except Exception:
        return "—"

if "drill_day" not in st.session_state:
    st.session_state["drill_day"] = None

# ══════════════════════════════════════════════════════════════
# HEADER — full-width navy title bar, then info row below
# ══════════════════════════════════════════════════════════════

# Full-width title bar
st.markdown("""
<div style="
    background:linear-gradient(135deg,#1A2F5A 0%,#1F3864 60%,#24527A 100%);
    padding:18px 28px;
    margin-top:35px;
    border-radius:10px;
    border-bottom:3px solid #2C7C7E;
    display:flex;
    align-items:center;
    gap:16px;
    box-shadow:0 2px 8px rgba(31,56,100,0.18);
">
    <div style="
        width:5px;height:44px;
        background:linear-gradient(180deg,#2C7C7E,#7ECFD0);
        border-radius:3px;flex-shrink:0;
    "></div>
    <div>
        <div style="
            color:#FFFFFF;
            font-size:26px;
            font-weight:800;
            font-family:'Segoe UI',Calibri,sans-serif;
            letter-spacing:0.5px;
            line-height:1.15;
        ">Telemetry Data</div>
        <div style="
            color:#7ECFD0;
            font-size:11px;
            font-family:'Segoe UI',Calibri,sans-serif;
            font-weight:600;
            letter-spacing:1.2px;
            text-transform:uppercase;
            margin-top:4px;
        ">DCIM Property View &nbsp;·&nbsp; LC Event Log</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# Case selector + Resolution in one row
info_case_col, info_res_col = st.columns([1, 2])

with info_case_col:
    st.markdown("""
    <div style="font-size:10px;font-weight:700;text-transform:uppercase;
        letter-spacing:0.7px;color:#1F3864;margin-bottom:4px;">
        Select Case Number
    </div>
    """, unsafe_allow_html=True)
    selected_case = st.selectbox(
        "Case Number", options=all_cases, key="global_case",
        label_visibility="collapsed"
    )

# ──────────────────────────────────────────────────────────────
# FILTER DATA
# ──────────────────────────────────────────────────────────────
logs_for_case   = df_logs[df_logs["case_nbr"] == selected_case].copy()
config_for_case = df_config[df_config["case_nbr"] == selected_case].copy()

case_row         = df_case_detail[df_case_detail["case_nbr"] == selected_case]
case_subject     = case_row["case_subject"].values[0]     if not case_row.empty else "—"
case_description = case_row["case_description"].values[0] if not case_row.empty else "—"
case_resolution  = build_parts_resolution(selected_case, df_parts)

# ══════════════════════════════════════════════════════════════
# CASE INFO ROW — Resolution in same row as case selector
# Below that: Subject + Description side by side
# ══════════════════════════════════════════════════════════════

# Resolution box shares the row with the case selectbox
with info_res_col:
    st.markdown("""
    <div style="font-size:10px;font-weight:700;text-transform:uppercase;
        letter-spacing:0.7px;color:#1F3864;margin-bottom:4px;">
        Case Resolution
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div style="
        background:#fff5f5;
        border:1px solid #f5c6c6;
        border-left:4px solid #C0392B;
        border-radius:8px;
        padding:8px 14px;
        box-shadow:0 1px 3px rgba(0,0,0,0.04);
        display:flex;
        align-items:center;
        min-height:38px;
    ">
        <div style="font-size:14px;font-weight:700;color:#C0392B;
            cursor:text;user-select:text;letter-spacing:0.5px;">
            {case_resolution}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# Subject + Description in a 1:2 layout below
subj_col, desc_col = st.columns([1, 2])

with subj_col:
    st.markdown(f"""
    <div style="
        background:#ffffff;
        border:1px solid #D9DCE0;
        border-left:4px solid #2C7C7E;
        border-radius:8px;
        padding:14px 18px;
        height:110px;
        overflow-y:auto;
        box-shadow:0 1px 3px rgba(0,0,0,0.05);
    ">
        <div style="font-size:10px;font-weight:700;text-transform:uppercase;
            letter-spacing:0.7px;color:#2C7C7E;margin-bottom:7px;">Subject</div>
        <div style="font-size:13px;font-weight:700;color:#1F3864;
            line-height:1.45;cursor:text;user-select:text;">{case_subject}</div>
    </div>
    """, unsafe_allow_html=True)

with desc_col:
    st.markdown(f"""
    <div style="
        background:#ffffff;
        border:1px solid #D9DCE0;
        border-left:4px solid #C18A3D;
        border-radius:8px;
        padding:14px 18px;
        height:110px;
        overflow-y:auto;
        box-shadow:0 1px 3px rgba(0,0,0,0.05);
    ">
        <div style="font-size:10px;font-weight:700;text-transform:uppercase;
            letter-spacing:0.7px;color:#C18A3D;margin-bottom:7px;">Description</div>
        <div style="font-size:13px;color:#333;line-height:1.55;
            cursor:text;user-select:text;">{case_description}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 4px 8px 4px;user-select:none;">
      <div style="font-size:11px;font-weight:700;text-transform:uppercase;
          letter-spacing:0.7px;color:#7ECFD0;border-bottom:1px solid #2C4872;
          padding-bottom:6px;margin-bottom:12px;">LC Log Filters</div>
    </div>
    """, unsafe_allow_html=True)

    prev_sev = st.session_state.get("log_severity",   [])
    prev_mid = st.session_state.get("log_messageids", [])

    sev_opts = sorted(
        (logs_for_case[logs_for_case["messageid"].isin(prev_mid)]
         if prev_mid else logs_for_case)["severity"].dropna().unique().tolist()
    )
    mid_opts = sorted(
        (logs_for_case[logs_for_case["severity"].isin(prev_sev)]
         if prev_sev else logs_for_case)["messageid"].dropna().unique().tolist()
    )

    selected_severity = st.multiselect(
        "Severity", options=sev_opts,
        default=[s for s in prev_sev if s in sev_opts], key="log_severity"
    )
    selected_msgids = st.multiselect(
        "Message ID", options=mid_opts,
        default=[m for m in prev_mid if m in mid_opts], key="log_messageids"
    )

    st.markdown("""
    <div style="padding:16px 4px 8px 4px;user-select:none;">
      <div style="font-size:11px;font-weight:700;text-transform:uppercase;
          letter-spacing:0.7px;color:#7ECFD0;border-bottom:1px solid #2C4872;
          padding-bottom:6px;margin-bottom:12px;">Config Filters</div>
    </div>
    """, unsafe_allow_html=True)

    # Only expose the 5 DCIM classes in the filter
    obj_opts = sorted([
        cls for cls in DCIM_CLASSES
        if cls in config_for_case["objectclass"].dropna().unique().tolist()
    ])
    selected_classes = st.multiselect(
        "objectclass", options=obj_opts, default=[], key="config_obj_classes"
    )
    ok_bad_filter = st.selectbox(
        "Status", options=["All", "OK", "BAD", "N/A"], index=0, key="ok_bad_filter"
    )

# ──────────────────────────────────────────────────────────────
# APPLY FILTERS
# ──────────────────────────────────────────────────────────────
final_logs = logs_for_case.copy()
if selected_severity:
    final_logs = final_logs[final_logs["severity"].isin(selected_severity)]
if selected_msgids:
    final_logs = final_logs[final_logs["messageid"].isin(selected_msgids)]
final_logs = final_logs.sort_values("timestamp", ascending=True)

case_config = config_for_case.copy()
if selected_classes:
    case_config = case_config[case_config["objectclass"].isin(selected_classes)]

full_result = engineer_features(case_config) if not case_config.empty else pd.DataFrame()
result_df   = (
    full_result[full_result["ok_bad"] == ok_bad_filter].reset_index(drop=True)
    if (not full_result.empty and ok_bad_filter != "All") else full_result.copy()
)

# ══════════════════════════════════════════════════════════════
# SECTION LABEL HELPER
# ══════════════════════════════════════════════════════════════
def section_label(title: str):
    st.markdown(f"""
    <div style="
        font-size:11px;font-weight:700;text-transform:uppercase;
        letter-spacing:0.8px;color:#2C7C7E;
        padding:20px 0 8px 0;border-bottom:2px solid #2C7C7E;
        margin-bottom:16px;user-select:none;
    ">{title}</div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SECTION 1 — LC LOGS KPI CARDS
# Each severity card shows: count + unique msg IDs below
# ══════════════════════════════════════════════════════════════
section_label("LC Logs")

sev_norm       = final_logs["severity"].str.strip().str.title()
total_events   = len(final_logs)
info_count     = int((sev_norm == "Informational").sum())
warn_count     = int((sev_norm == "Warning").sum())
crit_count     = int((sev_norm == "Critical").sum())

# Unique message IDs per severity
info_ids = final_logs[sev_norm == "Informational"]["messageid"].nunique()
warn_ids = final_logs[sev_norm == "Warning"]["messageid"].nunique()
crit_ids = final_logs[sev_norm == "Critical"]["messageid"].nunique()

# KPI helper — renders a styled card with count + sub-label for unique IDs
def kpi_card(label, count, unique_ids, accent_color, bg_color):
    return f"""
    <div style="
        background:{bg_color};
        border:1px solid #D9DCE0;
        border-top:4px solid {accent_color};
        border-radius:8px;
        padding:16px 18px 14px 18px;
        box-shadow:0 1px 4px rgba(0,0,0,0.06);
        user-select:none;
    ">
        <div style="font-size:10px;font-weight:700;text-transform:uppercase;
            letter-spacing:0.7px;color:#5A6070;margin-bottom:8px;">{label}</div>
        <div style="font-size:32px;font-weight:700;color:{accent_color};
            line-height:1;">{count}</div>
        <div style="font-size:11px;color:#5A6070;margin-top:6px;">
            <span style="font-weight:700;color:{accent_color};">{unique_ids}</span>
            &nbsp;unique msg IDs
        </div>
    </div>"""

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div style="
        background:#EAF4FB;
        border:1px solid #D9DCE0;
        border-top:4px solid #1565a0;
        border-radius:8px;
        padding:16px 18px 14px 18px;
        box-shadow:0 1px 4px rgba(0,0,0,0.06);
        user-select:none;
    ">
        <div style="font-size:10px;font-weight:700;text-transform:uppercase;
            letter-spacing:0.7px;color:#5A6070;margin-bottom:8px;">Total Events</div>
        <div style="font-size:32px;font-weight:700;color:#1F3864;line-height:1;">
            {total_events}</div>
        <div style="font-size:11px;color:#5A6070;margin-top:6px;">
            <span style="font-weight:700;color:#1565a0;">
                {final_logs['messageid'].nunique()}
            </span>&nbsp;unique msg IDs
        </div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(kpi_card("Informational", info_count, info_ids,
                         "#1565a0", "#EAF4FB"), unsafe_allow_html=True)

with k3:
    st.markdown(kpi_card("Warning", warn_count, warn_ids,
                         "#8a5a00", "#FEF6E7"), unsafe_allow_html=True)

with k4:
    st.markdown(kpi_card("Critical", crit_count, crit_ids,
                         "#C0392B", "#FDECEA"), unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════
# SECTION 2 — CONFIG DATA
# ══════════════════════════════════════════════════════════════
section_label("Config Data")

if full_result.empty:
    st.info("No config data available for this case.")
else:
    present_classes = [cls for cls in DCIM_CLASSES if cls in full_result["objectclass"].values]

    if present_classes:
        dcim_only  = full_result[full_result["objectclass"].isin(DCIM_CLASSES)]
        total_comp = len(dcim_only)
        total_ok   = int((dcim_only["ok_bad"] == "OK").sum())
        total_bad  = int((dcim_only["ok_bad"] == "BAD").sum())
        # Unique FQDD counts per status
        ok_fqdds  = dcim_only[dcim_only["ok_bad"] == "OK"]["FQDD"].nunique()
        bad_fqdds = dcim_only[dcim_only["ok_bad"] == "BAD"]["FQDD"].nunique()

        # Config KPI cards
        cc1, cc2, cc3 = st.columns(3)

        with cc1:
            st.markdown(f"""
            <div style="
                background:#E5EBF5;border:1px solid #D9DCE0;
                border-top:4px solid #1F3864;border-radius:8px;
                padding:16px 18px 14px 18px;
                box-shadow:0 1px 4px rgba(0,0,0,0.06);user-select:none;
            ">
                <div style="font-size:10px;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.7px;color:#5A6070;margin-bottom:8px;">Total Components</div>
                <div style="font-size:32px;font-weight:700;color:#1F3864;line-height:1;">
                    {total_comp}</div>
                <div style="font-size:11px;color:#5A6070;margin-top:6px;">
                    across {len(present_classes)} classes
                </div>
            </div>""", unsafe_allow_html=True)

        with cc2:
            st.markdown(f"""
            <div style="
                background:#EAF7EC;border:1px solid #D9DCE0;
                border-top:4px solid #27ae60;border-radius:8px;
                padding:16px 18px 14px 18px;
                box-shadow:0 1px 4px rgba(0,0,0,0.06);user-select:none;
            ">
                <div style="font-size:10px;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.7px;color:#5A6070;margin-bottom:8px;">Total OK</div>
                <div style="font-size:32px;font-weight:700;color:#27ae60;line-height:1;">
                    {total_ok}</div>
                <div style="font-size:11px;color:#5A6070;margin-top:6px;">
                    <span style="font-weight:700;color:#27ae60;">{ok_fqdds}</span>
                    &nbsp;unique FQDDs
                </div>
            </div>""", unsafe_allow_html=True)

        with cc3:
            st.markdown(f"""
            <div style="
                background:#FDECEA;border:1px solid #D9DCE0;
                border-top:4px solid #C0392B;border-radius:8px;
                padding:16px 18px 14px 18px;
                box-shadow:0 1px 4px rgba(0,0,0,0.06);user-select:none;
            ">
                <div style="font-size:10px;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.7px;color:#5A6070;margin-bottom:8px;">Total BAD</div>
                <div style="font-size:32px;font-weight:700;color:#C0392B;line-height:1;">
                    {total_bad}</div>
                <div style="font-size:11px;color:#5A6070;margin-top:6px;">
                    <span style="font-weight:700;color:#C0392B;">{bad_fqdds}</span>
                    &nbsp;unique FQDDs
                </div>
            </div>""", unsafe_allow_html=True)



        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        # ── DCIM BAD COMPONENTS cards (matches sample.png) ──────────
        bad_rows = result_df[result_df["ok_bad"] == "BAD"].reset_index(drop=True)

        if not bad_rows.empty:
            st.markdown("""
            <div style="font-size:11px;font-weight:700;text-transform:uppercase;
                letter-spacing:0.8px;color:#C0392B;
                padding-bottom:6px;border-bottom:1px solid #e8d0cf;
                margin-bottom:14px;user-select:none;">
                DCIM Bad Components
            </div>""", unsafe_allow_html=True)

            # Render in rows of 2 cards each
            for i in range(0, len(bad_rows), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx >= len(bad_rows):
                        break
                    row = bad_rows.iloc[idx]

                    fqdd      = row.get("FQDD", "?")
                    objcls    = row.get("objectclass", "?")
                    ps        = row.get("PrimaryStatus", "?")
                    pf        = row.get("PredictiveFailureState", "?")
                    red       = row.get("RedundancyStatus", "?")
                    sn        = row.get("SerialNumber", "?")

                    ps_color  = "#C0392B" if ps in BAD_PRIMARY_STATUS  else "#1F3864"
                    pf_color  = "#C18A3D" if pf in BAD_PREDICTIVE_FAIL else "#1F3864"
                    red_color = "#C0392B" if red in BAD_REDUNDANCY      else "#1F3864"

                    ps_label  = f"<span style='font-weight:700;color:{ps_color};font-family:Courier New,monospace'>{ps}</span>"
                    pf_label  = f"<span style='font-weight:700;color:{pf_color};font-family:Courier New,monospace'>{pf}</span>"
                    red_label = f"<span style='font-weight:700;color:{red_color};font-family:Courier New,monospace'>{red}</span>"
                    sn_label  = f"<span style='font-weight:700;font-family:Courier New,monospace'>{sn}</span>"

                    with col:
                        st.markdown(f"""
                        <div style="
                            background:#ffffff;
                            border:1px solid #D9DCE0;
                            border-left:4px solid #C0392B;
                            border-radius:8px;
                            padding:16px 20px 14px 16px;
                            margin-bottom:10px;
                            box-shadow:0 1px 4px rgba(0,0,0,0.07);
                        ">
                            <div style="margin-bottom:8px;">
                                <span style="
                                    background:#C0392B;color:#fff;
                                    font-size:10px;font-weight:700;
                                    padding:2px 9px;border-radius:3px;
                                    letter-spacing:0.8px;text-transform:uppercase;
                                ">BAD</span>
                            </div>
                            <div style="
                                font-family:'Courier New',monospace;
                                font-size:12px;font-weight:700;
                                color:#1F3864;margin-bottom:4px;
                                word-break:break-all;
                            ">{fqdd}</div>
                            <div style="
                                font-size:12px;color:#2C7C7E;
                                margin-bottom:14px;font-weight:600;
                            ">{objcls}</div>
                            <div style="
                                display:grid;grid-template-columns:1fr 1fr;
                                gap:8px 20px;
                            ">
                                <div>
                                    <div style="font-size:11px;color:#5A6070;margin-bottom:2px;">Primary Status</div>
                                    <div>{ps_label}</div>
                                </div>
                                <div>
                                    <div style="font-size:11px;color:#5A6070;margin-bottom:2px;">Predictive Failure</div>
                                    <div>{pf_label}</div>
                                </div>
                                <div>
                                    <div style="font-size:11px;color:#5A6070;margin-bottom:2px;">Redundancy Status</div>
                                    <div>{red_label}</div>
                                </div>
                                <div>
                                    <div style="font-size:11px;color:#5A6070;margin-bottom:2px;">Serial Number</div>
                                    <div>{sn_label}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        # ── Class-level expanders (OK + BAD summary per class) ──────
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        for cls in present_classes:
            subset_detail = result_df[result_df["objectclass"] == cls]
            if subset_detail.empty:
                continue
            label = CLASS_LABELS[cls]
            bad_n = int((full_result[full_result["objectclass"] == cls]["ok_bad"] == "BAD").sum())
            ok_n  = int((full_result[full_result["objectclass"] == cls]["ok_bad"] == "OK").sum())

            with st.expander(f"{label}  —  {ok_n} OK  |  {bad_n} BAD", expanded=False):
                st.markdown(
                    f"<span style='color:#27ae60;font-weight:700;font-size:13px'>{ok_n} OK</span>"
                    f"<span style='color:#bdc3c7;margin:0 8px'>|</span>"
                    f"<span style='color:#C0392B;font-weight:700;font-size:13px'>{bad_n} BAD</span>",
                    unsafe_allow_html=True
                )
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

                dcols = [c for c in [
                    "ok_bad", "FQDD", "SerialNumber",
                    "PrimaryStatus", "PredictiveFailureState",
                    "RedundancyStatus", "system_id", "collectiontimestamp"
                ] if c in subset_detail.columns]

                def _hl(row):
                    bg = "#FDECEA" if row["ok_bad"] == "BAD" else "#EAF7EC"
                    return [f"background-color:{bg}"] * len(row)

                st.dataframe(
                    subset_detail[dcols].reset_index(drop=True).style.apply(_hl, axis=1),
                    use_container_width=True, hide_index=True
                )



# ══════════════════════════════════════════════════════════════
# SECTION — LOG EVENT TIMELINE (Day → Hour drill-down)
# ══════════════════════════════════════════════════════════════
section_label("Log Event Timeline")

SEV_COLOR = {
    "Critical":      "#C0392B",
    "Warning":       "#C18A3D",
    "Informational": "#1F3864",
    "Other":         "#7F8C8D",
}

def _normalise_sev(s: str) -> str:
    t = s.strip().title()
    return t if t in SEV_COLOR else "Other"

if final_logs.empty:
    st.info("No log events to chart for this case and filter combination.")
else:
    chart_logs = final_logs.copy()
    chart_logs = chart_logs[chart_logs["timestamp"].notna()]
    chart_logs["sev_norm"] = chart_logs["severity"].apply(_normalise_sev)
    chart_logs["date"]     = chart_logs["timestamp"].dt.date

    drill_day = st.session_state.get("drill_day")

    # ── DRILL-DOWN: hour view ────────────────────────────────
    if drill_day is not None:
        day_logs = chart_logs[
            chart_logs["timestamp"].dt.date == drill_day
        ].copy()
        day_logs["hour"] = day_logs["timestamp"].dt.hour

        col_back, col_title = st.columns([1, 6])
        with col_back:
            if st.button("← All Days", key="back_to_days"):
                st.session_state["drill_day"] = None
                st.rerun()
        with col_title:
            st.markdown(
                f"<div style='font-size:13px;font-weight:700;color:#1F3864;"
                f"padding-top:6px;'>Hourly breakdown — "
                f"<span style='color:#2C7C7E;'>{drill_day.strftime('%d %b %Y')}</span></div>",
                unsafe_allow_html=True,
            )

        hour_fig  = go.Figure()
        all_hours = list(range(24))

        for sev in ["Critical", "Warning", "Informational", "Other"]:
            sev_data = day_logs[day_logs["sev_norm"] == sev]
            if sev_data.empty:
                continue
            counts = sev_data.groupby("hour").size().reindex(all_hours, fill_value=0)

            hover_texts = []
            for h in all_hours:
                hr_sev = sev_data[sev_data["hour"] == h]
                if hr_sev.empty:
                    hover_texts.append(f"<b>{sev}</b><br>No events at {h:02d}:00")
                    continue
                top_ids  = hr_sev["messageid"].value_counts().head(5)
                id_lines = "<br>".join(
                    f"&nbsp;&nbsp;{mid} ×{cnt}"
                    for mid, cnt in top_ids.items()
                )
                hover_texts.append(
                    f"<b>{sev}</b><br>"
                    f"Events: <b>{len(hr_sev)}</b><br>"
                    f"──────────────<br>"
                    f"<b>Message IDs:</b><br>{id_lines}"
                )

            hour_fig.add_trace(go.Bar(
                x=[f"{h:02d}:00" for h in all_hours],
                y=counts.values,
                name=sev,
                marker_color=SEV_COLOR[sev],
                marker_line_width=0,
                text=hover_texts,
                textposition="none",
                hovertemplate="%{text}<extra></extra>",
            ))

        total_day = len(day_logs)
        peak_hour = int(day_logs.groupby("hour").size().idxmax()) if not day_logs.empty else 0

        hour_fig.update_layout(
            barmode="stack",
            height=320,
            margin=dict(l=0, r=0, t=44, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#F8FAFC",
            font=dict(family="Calibri, Segoe UI, sans-serif", size=12, color="#1F3864"),
            hoverlabel=dict(
                bgcolor="#1F3864",
                font=dict(size=11, color="#FFFFFF", family="Calibri, Segoe UI, sans-serif"),
                bordercolor="#2C7C7E",
                namelength=0,
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1,
                font=dict(size=11),
                bgcolor="rgba(0,0,0,0)",
            ),
            xaxis=dict(
                showgrid=False,
                tickfont=dict(size=10, color="#5A6070"),
                linecolor="#D9DCE0",
            ),
            yaxis=dict(
                gridcolor="#EEF0F3",
                tickfont=dict(size=10, color="#5A6070"),
                title=dict(text="Events", font=dict(size=10, color="#5A6070")),
                linecolor="#D9DCE0",
            ),
            bargap=0.18,
            annotations=[dict(
                text=(
                    f"<b>{total_day}</b> events on {drill_day.strftime('%d %b')}  ·  "
                    f"Peak hour: <b>{peak_hour:02d}:00</b>"
                ),
                xref="paper", yref="paper",
                x=0, y=1.13, showarrow=False,
                font=dict(size=11, color="#2C7C7E"),
                align="left",
            )],
        )
        st.plotly_chart(hour_fig, use_container_width=True, config={"displayModeBar": False})

    # ── DAY VIEW ────────────────────────────────────────────
    else:
        all_dates = sorted(chart_logs["date"].unique())
        day_fig   = go.Figure()

        for sev in ["Critical", "Warning", "Informational", "Other"]:
            sev_data = chart_logs[chart_logs["sev_norm"] == sev]
            if sev_data.empty:
                continue
            counts = sev_data.groupby("date").size().reindex(all_dates, fill_value=0)
            day_fig.add_trace(go.Bar(
                x=[d.strftime("%d %b") for d in all_dates],
                y=counts.values,
                name=sev,
                marker_color=SEV_COLOR[sev],
                marker_line_width=0,
                hovertemplate=(
                    f"<b>{sev}</b><br>"
                    "Date: %{x}<br>"
                    "Events: %{y}<br>"
                    "<i>Select day below to drill into hours</i>"
                    "<extra></extra>"
                ),
            ))

        daily_totals = chart_logs.groupby("date").size()
        peak_day     = daily_totals.idxmax() if not daily_totals.empty else None
        peak_label   = peak_day.strftime("%d %b") if peak_day else ""
        peak_count   = int(daily_totals.max()) if not daily_totals.empty else 0

        day_fig.update_layout(
            barmode="stack",
            height=320,
            margin=dict(l=0, r=0, t=44, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#F8FAFC",
            font=dict(family="Calibri, Segoe UI, sans-serif", size=12, color="#1F3864"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.06,
                xanchor="right", x=1,
                font=dict(size=11),
                bgcolor="rgba(0,0,0,0)",
            ),
            xaxis=dict(
                showgrid=False,
                tickfont=dict(size=10, color="#5A6070"),
                linecolor="#D9DCE0",
            ),
            yaxis=dict(
                gridcolor="#EEF0F3",
                tickfont=dict(size=10, color="#5A6070"),
                title=dict(text="Events", font=dict(size=10, color="#5A6070")),
                linecolor="#D9DCE0",
            ),
            bargap=0.25,
            annotations=[dict(
                text=(
                    f"Peak day: <b>{peak_label}</b> ({peak_count} events)  ·  "
                    f"Select a day below to drill into hourly view"
                ),
                xref="paper", yref="paper",
                x=0, y=1.16, showarrow=False,
                font=dict(size=11, color="#2C7C7E"),
                align="left",
            )],
        )

        st.plotly_chart(day_fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            "<div style='font-size:11px;color:#5A6070;margin-top:-8px;margin-bottom:8px;"
            "user-select:none;'>Select a day below to drill into hourly view</div>",
            unsafe_allow_html=True,
        )

        date_options = [d.strftime("%d %b %Y") for d in all_dates]
        date_map     = {d.strftime("%d %b %Y"): d for d in all_dates}

        selected_date_str = st.selectbox(
            "Drill into day",
            options=["— select a day —"] + date_options,
            key="day_drill_select",
            label_visibility="collapsed",
        )
        if selected_date_str != "— select a day —":
            st.session_state["drill_day"] = date_map[selected_date_str]
            st.rerun()


# ══════════════════════════════════════════════════════════════
# SECTION 3 — LOG EVENTS TABLE
# ══════════════════════════════════════════════════════════════
section_label("Log Events Table")

if final_logs.empty:
    st.info("No log events found for this case and filter combination.")
else:
    display_logs = final_logs[
        ["messageid", "severity", "message", "timestamp"]
    ].reset_index(drop=True)

    def _colour_log_row(row):
        sev = str(row["severity"]).strip().title()
        if sev == "Critical":
            bg = "#FDECEA"   # red tint  — matches config BAD card
        elif sev == "Warning":
            bg = "#FEF6E7"   # amber tint — matches warning KPI card
        elif sev == "Informational":
            bg = "#EAF4FB"   # blue tint  — matches info KPI card
        else:
            bg = "#FFFFFF"
        return [f"background-color:{bg}"] * len(row)

    st.dataframe(
        display_logs.style.apply(_colour_log_row, axis=1),
        use_container_width=True,
        height=520,
        hide_index=True,
    )