import io
import re
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import curve_fit
import plotly.graph_objects as go


# ----------------------------
# 4PL model and helpers
# ----------------------------
def four_pl(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    4-parameter logistic (4PL):
    y = d + (a - d) / (1 + (x/c)^b)
    """
    x = np.asarray(x, dtype=float)
    return d + (a - d) / (1.0 + (x / c) ** b)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1.0 - (ss_res / ss_tot)


def initial_guess_4pl(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Reasonable defaults
    a0 = float(np.max(y))  # upper asymptote (can flip depending on assay, but this is fine as start)
    d0 = float(np.min(y))  # lower asymptote
    c0 = float(np.median(x)) if np.all(np.isfinite(x)) and len(x) else 1.0
    b0 = 1.0

    # If curve is decreasing, swap a and d guess to help optimizer
    if len(y) >= 2 and (y[-1] - y[0]) < 0:
        a0, d0 = d0, a0

    # Ensure c0 positive
    c0 = max(c0, np.min(x[x > 0]) if np.any(x > 0) else 1.0)
    return a0, b0, c0, d0


def fit_4pl(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[str]]:
    """
    Returns: (popt, pcov, r2, error_message)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 4:
        return None, None, None, "Need at least 4 points to fit 4PL."

    if np.any(x <= 0):
        return None, None, None, "All concentrations must be > 0 for log-scale and 4PL."

    p0 = initial_guess_4pl(x, y)

    # Bounds: keep c positive; keep b within a broad range to stabilize; a/d within wide y-range
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_span = max(y_max - y_min, 1e-9)

    lower = [y_min - 5 * y_span, -10.0, np.min(x) * 1e-6, y_min - 5 * y_span]
    upper = [y_max + 5 * y_span,  10.0, np.max(x) * 1e6, y_max + 5 * y_span]

    try:
        popt, pcov = curve_fit(
            four_pl, x, y,
            p0=p0,
            bounds=(lower, upper),
            maxfev=20000,
        )
        y_hat = four_pl(x, *popt)
        r2 = float(r_squared(y, y_hat))
        return popt, pcov, r2, None
    except Exception as e:
        return None, None, None, f"Fit failed: {e}"


def invert_4pl(y: float, a: float, b: float, c: float, d: float) -> float:
    """
    Invert 4PL to solve for x given y.
    y = d + (a-d)/(1+(x/c)^b)

    Rearranged:
    (a-d)/(y-d) - 1 = (x/c)^b
    x = c * [ ( (a-d)/(y-d) ) - 1 ]^(1/b)

    Returns np.nan if inversion is invalid.
    """
    y = float(y)
    denom = (y - d)
    num = (a - d)

    if denom == 0:
        return np.nan

    t = (num / denom) - 1.0

    # For valid real solution, t must be > 0 if 1/b is real for general b.
    # If b is an odd integer you might allow negatives, but keep it simple and robust here.
    if not np.isfinite(t) or t <= 0 or not np.isfinite(b) or b == 0 or c <= 0:
        return np.nan

    x = c * (t ** (1.0 / b))
    return float(x) if np.isfinite(x) and x > 0 else np.nan


# ----------------------------
# Parsing pasted data
# ----------------------------
def sniff_delimiter(text: str) -> Optional[str]:
    """
    Try to detect delimiter among comma, tab, or whitespace.
    Returns:
      - ',' or '\t' or None (meaning whitespace)
    """
    head = "\n".join(text.strip().splitlines()[:5])
    if "," in head:
        return ","
    if "\t" in head:
        return "\t"
    return None  # whitespace


def parse_table(text: str) -> pd.DataFrame:
    """
    Parse CSV, TSV, or whitespace-delimited data with a header row.
    First column is concentration, subsequent columns are series signals.
    """
    text = text.strip()
    if not text:
        raise ValueError("Empty input.")

    delim = sniff_delimiter(text)
    buf = io.StringIO(text)

    if delim is None:
        df = pd.read_csv(buf, sep=r"\s+", engine="python")
    else:
        df = pd.read_csv(buf, sep=delim)

    if df.shape[1] < 2:
        raise ValueError("Need at least 2 columns (Concentration + at least one signal series).")

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # First column is concentration
    conc_col = df.columns[0]
    df = df.rename(columns={conc_col: "Concentration"})

    # Coerce numeric
    df["Concentration"] = pd.to_numeric(df["Concentration"], errors="coerce")
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["Concentration"].isna().any():
        raise ValueError("Concentration column contains non-numeric values.")

    if (df["Concentration"] <= 0).any():
        raise ValueError("All concentrations must be > 0.")

    return df


def make_unique_internal(existing: set, base: str) -> str:
    base_clean = re.sub(r"\s+", " ", base.strip())
    if base_clean == "":
        base_clean = "Series"
    candidate = base_clean
    i = 2
    while candidate in existing:
        candidate = f"{base_clean} ({i})"
        i += 1
    return candidate


# ----------------------------
# Streamlit app
# ----------------------------
st.set_page_config(page_title="4PL Standard Curves", layout="wide")

st.title("4PL Standard Curves (paste CSV/TSV/whitespace)")

if "wide_df" not in st.session_state:
    st.session_state.wide_df = None  # pd.DataFrame with Concentration + internal series columns
if "series_meta" not in st.session_state:
    # Dict internal_col -> display_name
    st.session_state.series_meta: Dict[str, str] = {}
if "last_paste_error" not in st.session_state:
    st.session_state.last_paste_error = None

with st.sidebar:
    st.header("Actions")
    if st.button("Clear all data", type="secondary"):
        st.session_state.wide_df = None
        st.session_state.series_meta = {}
        st.session_state.last_paste_error = None
        st.rerun()

    st.caption("Paste data into the main input box to add new series columns.")


st.subheader("Paste tabular data to add series")
paste = st.text_area(
    "Paste CSV, TSV, or whitespace-delimited data (header required). First column is Concentration.",
    height=180,
    placeholder="Conc,Manual,Auto R1\n5,0.02075,0.05155\n10,0.04206,0.09497\n...",
)

col_add, col_note = st.columns([1, 2], vertical_alignment="center")
with col_add:
    add_clicked = st.button("Add to plot", type="primary")
with col_note:
    st.caption("Adds only new columns. Concentrations must match existing concentrations (no new rows).")

if add_clicked:
    try:
        df_new = parse_table(paste)

        # If first time, initialize
        if st.session_state.wide_df is None:
            wide = df_new.copy()

            # Rename incoming series columns to unique internal ids (so display names can change)
            existing_internal = set(["Concentration"])
            new_meta = {}

            for col in wide.columns[1:]:
                internal = make_unique_internal(existing_internal, col)
                existing_internal.add(internal)
                new_meta[internal] = col  # display name initial equals header
                wide = wide.rename(columns={col: internal})

            # Sort concentrations just to be nice
            wide = wide.sort_values("Concentration").reset_index(drop=True)

            st.session_state.wide_df = wide
            st.session_state.series_meta = new_meta
            st.session_state.last_paste_error = None
            st.success(f"Added {len(new_meta)} series.")
        else:
            wide = st.session_state.wide_df.copy()

            # Validate concentrations match, and align order
            conc_existing = wide["Concentration"].to_numpy()
            conc_new = df_new["Concentration"].to_numpy()

            if len(conc_existing) != len(conc_new) or not np.allclose(
                np.sort(conc_existing), np.sort(conc_new), rtol=0, atol=0
            ):
                raise ValueError(
                    "Concentrations do not match existing data. This app only supports adding new columns (no new rows)."
                )

            # Reindex new df to match existing concentration order
            df_new_idxed = df_new.set_index("Concentration").reindex(wide["Concentration"]).reset_index()

            # Add only new signal columns
            existing_internal = set(wide.columns)
            added = 0
            for col in df_new_idxed.columns[1:]:
                internal = make_unique_internal(existing_internal, col)
                existing_internal.add(internal)

                wide[internal] = df_new_idxed[col].to_numpy()
                st.session_state.series_meta[internal] = col
                added += 1

            st.session_state.wide_df = wide
            st.session_state.last_paste_error = None
            if added > 0:
                st.success(f"Added {added} new series.")
            else:
                st.info("No new series columns found to add.")
    except Exception as e:
        st.session_state.last_paste_error = str(e)

if st.session_state.last_paste_error:
    st.error(st.session_state.last_paste_error)

if st.session_state.wide_df is None:
    st.info("Paste some data and click 'Add to plot' to begin.")
    st.stop()

wide = st.session_state.wide_df.copy()
series_meta = st.session_state.series_meta.copy()
series_cols = [c for c in wide.columns if c != "Concentration"]

st.divider()

# ----------------------------
# Editable series names (keys)
# ----------------------------
st.subheader("Series keys (editable)")
meta_df = pd.DataFrame(
    [{"internal": c, "key": series_meta.get(c, c)} for c in series_cols]
)

edited_meta = st.data_editor(
    meta_df,
    hide_index=True,
    use_container_width=True,
    column_config={
        "internal": st.column_config.TextColumn("Internal ID", disabled=True),
        "key": st.column_config.TextColumn("Series Key", help="Edit this to rename the series in plot and tables."),
    },
    key="meta_editor",
)

# Apply edits
new_meta = {row["internal"]: row["key"] for _, row in edited_meta.iterrows()}
st.session_state.series_meta = new_meta
series_meta = new_meta

# ----------------------------
# Data table display (wide)
# ----------------------------
st.subheader("Accumulated data")
display_wide = wide.rename(columns={c: series_meta.get(c, c) for c in series_cols})
st.dataframe(display_wide, use_container_width=True)

st.divider()

# ----------------------------
# Fit each series and build plot
# ----------------------------
st.subheader("Plot and 4PL fits")

fits_rows: List[dict] = []

fig = go.Figure()
fig.update_layout(
    xaxis_title="Concentration",
    yaxis_title="Signal",
    xaxis_type="log",
    legend_title_text="Series",
    height=520,
)

x_all = wide["Concentration"].to_numpy(dtype=float)
x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
x_smooth = np.logspace(np.log10(x_min), np.log10(x_max), 250)

for internal_col in series_cols:
    key_name = series_meta.get(internal_col, internal_col)
    y = wide[internal_col].to_numpy(dtype=float)

    # Points
    fig.add_trace(
        go.Scatter(
            x=x_all,
            y=y,
            mode="markers",
            name=f"{key_name} (data)",
        )
    )

    popt, pcov, r2, err = fit_4pl(x_all, y)

    if popt is not None:
        a, b, c, d = popt
        y_fit = four_pl(x_smooth, *popt)

        fig.add_trace(
            go.Scatter(
                x=x_smooth,
                y=y_fit,
                mode="lines",
                name=f"{key_name} (4PL)",
            )
        )

        fits_rows.append(
            {
                "Series Key": key_name,
                "A": a,
                "B": b,
                "C": c,
                "D": d,
                "R^2": r2,
                "Fit Status": "OK",
            }
        )
    else:
        fits_rows.append(
            {
                "Series Key": key_name,
                "A": np.nan,
                "B": np.nan,
                "C": np.nan,
                "D": np.nan,
                "R^2": np.nan,
                "Fit Status": err or "Failed",
            }
        )

st.plotly_chart(fig, use_container_width=True)

fits_df = pd.DataFrame(fits_rows)
st.subheader("4PL parameters per series")
st.dataframe(fits_df, use_container_width=True)

st.divider()

# ----------------------------
# Unknown concentration inference
# ----------------------------
st.subheader("Infer concentration from an unknown signal")
unknown_signal = st.number_input("Unknown signal value", value=0.0, step=0.001, format="%.6f")

infer_rows = []
for row in fits_rows:
    key_name = row["Series Key"]
    if row["Fit Status"] != "OK":
        infer_rows.append({"Series Key": key_name, "Inferred Concentration": np.nan, "Note": "Fit unavailable"})
        continue

    a, b, c, d = row["A"], row["B"], row["C"], row["D"]
    x_inf = invert_4pl(unknown_signal, a, b, c, d)

    note = ""
    if np.isnan(x_inf):
        note = "Signal out of invertible range for this curve (or unstable parameters)."

    infer_rows.append({"Series Key": key_name, "Inferred Concentration": x_inf, "Note": note})

infer_df = pd.DataFrame(infer_rows)
st.dataframe(infer_df, use_container_width=True)

# Optional: simple combined summary across valid curves
valid = infer_df["Inferred Concentration"].to_numpy(dtype=float)
valid = valid[np.isfinite(valid) & (valid > 0)]
if len(valid) > 0:
    st.caption(
        f"Summary across valid curves: median={np.median(valid):.6g}, mean={np.mean(valid):.6g}, n={len(valid)}"
    )
else:
    st.caption("No valid inferred concentrations (check fits and unknown signal range).")
