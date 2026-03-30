"""
VaR & ES Risk Dashboard — Streamlit Deployment
================================================
Capstone: Comparative Analysis of VaR and ES with Backtesting
under Stress Market Conditions + ML Extension

Run:  streamlit run app.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import yfinance as yf
from scipy import stats
from scipy.stats import norm, t as student_t
import os, io

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VaR & ES Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1rem; color: #666; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa; border-left: 4px solid #0f3460;
        padding: 1rem; border-radius: 0.5rem; margin-bottom: 0.5rem;
    }
    .zone-green  { color: #28a745; font-weight: 700; }
    .zone-yellow { color: #ffc107; font-weight: 700; }
    .zone-red    { color: #dc3545; font-weight: 700; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 0.4rem 1rem; border-radius: 0.4rem 0.4rem 0 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PRESET_TICKERS = {
    "S&P 500":         "^GSPC",
    "Nifty 50":        "^NSEI",
    "Lehman Brothers": "LEHMQ",
    "JPMorgan Chase":  "JPM",
    "Bitcoin":         "BTC-USD",
    "Gold":            "GC=F",
    "Apple":           "AAPL",
    "Tesla":           "TSLA",
}

CRISIS_PERIODS = [
    ("GFC 2008",       "2008-09-01", "2009-03-31", "#FF6B6B"),
    ("COVID-19 2020",  "2020-02-01", "2020-04-30", "#FFA07A"),
    ("Rate Hike 2022", "2022-01-01", "2022-10-31", "#FFD700"),
]

COLORS = {
    "Historical":  "#2E86AB",
    "Normal":      "#A23B72",
    "Student-t":   "#F18F01",
    "Monte Carlo": "#C73E1D",
    "LSTM":        "#28a745",
    "XGBoost":     "#6f42c1",
}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ─────────────────────────────────────────────────────────────────────────────
# CORE RISK FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def var_es_historical(r, conf):
    alpha = 1 - conf
    var = -np.percentile(r, alpha * 100)
    tail = r[r <= -var]
    es = -tail.mean() if len(tail) > 0 else var
    return var, es

def var_es_normal(r, conf):
    mu, sigma = r.mean(), r.std()
    z   = norm.ppf(1 - conf)
    var = -(mu + z * sigma)
    es  = -(mu - sigma * norm.pdf(norm.ppf(1 - conf)) / (1 - conf))
    return var, es

def var_es_student_t(r, conf):
    try:
        df_fit, loc, scale = student_t.fit(r, floc=r.mean())
        alpha = 1 - conf
        q     = student_t.ppf(alpha, df_fit, loc, scale)
        var   = -q
        es_num = student_t.expect(lambda x: x, args=(df_fit,),
                                   loc=loc, scale=scale, lb=-np.inf, ub=q)
        es = -es_num / alpha
        return var, es
    except Exception:
        return var_es_normal(r, conf)

def var_es_montecarlo(r, conf, n_sim=10000):
    np.random.seed(42)
    mu, sigma = r.mean(), r.std()
    sim   = np.random.normal(mu, sigma, n_sim)
    alpha = 1 - conf
    var   = -np.percentile(sim, alpha * 100)
    tail  = sim[sim <= -var]
    es    = -tail.mean() if len(tail) > 0 else var
    return var, es

METHODS = {
    "Historical":  var_es_historical,
    "Normal":      var_es_normal,
    "Student-t":   var_es_student_t,
    "Monte Carlo": var_es_montecarlo,
}

def kupiec_pof(exceptions, n_obs, confidence):
    p, x, T = 1 - confidence, exceptions, n_obs
    if x == 0:   LR = -2 * T * np.log(1 - p + 1e-10)
    elif x == T: LR = -2 * T * np.log(p + 1e-10)
    else:
        p_hat = x / T
        LR = -2 * (np.log((1-p)**(T-x) * p**x + 1e-100) -
                   np.log((1-p_hat)**(T-x) * p_hat**x + 1e-100))
    pval = 1 - stats.chi2.cdf(LR, df=1)
    return round(LR, 4), round(pval, 4), pval > 0.05

def christoffersen_test(exc_series):
    ex = np.array(exc_series, dtype=int)
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(ex)):
        prev, curr = ex[i-1], ex[i]
        if   prev==0 and curr==0: n00 += 1
        elif prev==0 and curr==1: n01 += 1
        elif prev==1 and curr==0: n10 += 1
        else:                     n11 += 1
    n0 = n00 + n01; n1 = n10 + n11
    pi0 = n01 / n0 if n0 > 0 else 0
    pi1 = n11 / n1 if n1 > 0 else 0
    pi  = (n01 + n11) / (n00 + n01 + n10 + n11 + 1e-9)
    def sl(x): return np.log(x) if x > 0 else 0
    L_i = ((1-pi)**n00 * pi**n01 * (1-pi)**n10 * pi**n11)
    L_d = ((1-pi0)**n00 * pi0**n01 * (1-pi1)**n10 * pi1**n11)
    LR  = -2 * (sl(L_i) - sl(L_d)) if L_d > 0 else 0
    pval = 1 - stats.chi2.cdf(LR, df=1)
    return round(LR, 4), round(pval, 4), pval > 0.05

def traffic_light(exceptions, n_obs=250):
    rate = exceptions / n_obs * 250
    if rate <= 4:  return "🟢 Green"
    elif rate <= 9: return "🟡 Yellow"
    else:           return "🔴 Red"

# ─────────────────────────────────────────────────────────────────────────────
# ML FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def build_features_simple(r_series, conf=0.99):
    """Build feature DataFrame from return series (no heavy ML libs)."""
    df = pd.DataFrame({"ret": r_series})
    df["vol_5"]   = df["ret"].rolling(5).std()
    df["vol_21"]  = df["ret"].rolling(21).std()
    df["vol_63"]  = df["ret"].rolling(63).std()
    df["skew_21"] = df["ret"].rolling(21).skew()
    df["kurt_21"] = df["ret"].rolling(21).kurt()
    for lag in range(1, 6):
        df[f"ret_lag{lag}"] = df["ret"].shift(lag)
    df["vol_lag1"] = df["vol_21"].shift(1)

    W = 250
    var_hist, es_hist = [], []
    arr = df["ret"].values
    for i in range(len(arr)):
        if i < W:
            var_hist.append(np.nan); es_hist.append(np.nan)
        else:
            window = arr[i-W:i]
            alpha  = 1 - conf
            v = -np.percentile(window, alpha * 100)
            tail = window[window <= -v]
            e = -tail.mean() if len(tail) > 0 else v
            var_hist.append(v); es_hist.append(e)

    df["var_hist"] = var_hist
    df["es_hist"]  = es_hist
    return df.dropna()


def xgboost_var_predict(returns_arr, conf=0.99, test_frac=0.2):
    """
    XGBoost quantile regression for VaR.
    Uses scikit-learn GradientBoostingRegressor as fallback
    when xgboost is unavailable.
    """
    try:
        from xgboost import XGBRegressor
        model_cls = lambda: XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=1 - conf,
            n_estimators=200, max_depth=4,
            learning_rate=0.05, subsample=0.8,
            random_state=42, verbosity=0
        )
        lib_name = "XGBoost"
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        model_cls = lambda: GradientBoostingRegressor(
            loss="quantile", alpha=1 - conf,
            n_estimators=200, max_depth=4,
            learning_rate=0.05, subsample=0.8,
            random_state=42
        )
        lib_name = "GradientBoosting (fallback)"

    r_series = pd.Series(returns_arr)
    df_f = build_features_simple(r_series, conf)

    feat_cols = ["vol_5", "vol_21", "vol_63", "skew_21", "kurt_21",
                 "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4", "ret_lag5",
                 "vol_lag1"]
    X = df_f[feat_cols].values
    y = df_f["ret"].values
    split = int(len(X) * (1 - test_frac))

    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    dates_te = df_f.index[split:]

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    model = model_cls()
    model.fit(X_tr_s, y_tr)
    var_pred = -model.predict(X_te_s)   # negate: losses positive

    exc_bool = (-y_te) > var_pred
    n_exc = int(exc_bool.sum())
    n_obs = len(exc_bool)
    lr_k, pval_k, pass_k = kupiec_pof(n_exc, n_obs, conf)
    tl = traffic_light(n_exc, n_obs)

    return {
        "lib": lib_name,
        "var_pred": var_pred,
        "actual": y_te,
        "dates": dates_te,
        "exc_bool": exc_bool,
        "n_exc": n_exc,
        "n_obs": n_obs,
        "kupiec_pass": "✓ Pass" if pass_k else "✗ Fail",
        "traffic_light": tl,
        "exc_rate": round(n_exc / n_obs * 100, 2),
    }


def isolation_forest_detect(returns_arr, r_index, contamination=0.05):
    """Isolation Forest anomaly detection — stress regime identification."""
    from sklearn.ensemble import IsolationForest
    r_series = pd.Series(returns_arr, index=r_index)
    df_f = build_features_simple(r_series)

    feat_cols = ["vol_5", "vol_21", "vol_63", "skew_21", "kurt_21",
                 "ret_lag1", "vol_lag1"]
    X = df_f[feat_cols].values
    dates = df_f.index

    iforest = IsolationForest(n_estimators=200, contamination=contamination,
                              random_state=42, n_jobs=-1)
    iforest.fit(X)
    preds   = iforest.predict(X)
    scores  = -iforest.score_samples(X)
    anomaly = (preds == -1).astype(int)

    # Ground truth from crisis periods
    gt = np.zeros(len(dates), dtype=int)
    for _, start, end, _ in CRISIS_PERIODS:
        mask = (dates >= start) & (dates <= end)
        gt[mask] = 1

    from sklearn.metrics import precision_score, recall_score, f1_score
    prec = precision_score(gt, anomaly, zero_division=0)
    rec  = recall_score(gt, anomaly, zero_division=0)
    f1   = f1_score(gt, anomaly, zero_division=0)

    return {
        "anomaly": anomaly, "scores": scores,
        "gt": gt, "dates": dates,
        "precision": prec, "recall": rec, "f1": f1,
        "n_detected": anomaly.sum(),
    }


def random_forest_zone(all_returns_dict, conf=0.99):
    """Random Forest Basel zone classifier across all assets."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    feat_cols = ["vol_5", "vol_21", "vol_63", "skew_21", "kurt_21",
                 "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4", "ret_lag5",
                 "vol_lag1"]

    def get_zone(exc):
        rate = exc / 250 * 250
        if rate <= 4:  return 0
        elif rate <= 9: return 1
        else:          return 2

    all_X, all_y = [], []
    for name, r_series in all_returns_dict.items():
        df_f = build_features_simple(r_series, conf)
        arr  = r_series.values
        W    = 250
        alpha = 1 - conf
        zones = []
        for i in range(W, len(arr)):
            window = arr[i-W:i]
            v = -np.percentile(window, alpha * 100)
            exc = int(np.sum(-window > v))
            zones.append(get_zone(exc))
        zones = np.array(zones)
        X_piece = df_f[feat_cols].values[-len(zones):]
        min_len = min(len(X_piece), len(zones))
        all_X.append(X_piece[:min_len])
        all_y.append(zones[:min_len])

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)

    if len(np.unique(y_all)) < 2:
        return None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                 class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    rf.fit(X_tr_s, y_tr)
    y_pred = rf.predict(X_te_s)
    acc    = accuracy_score(y_te, y_pred)
    fi     = pd.Series(rf.feature_importances_, index=feat_cols).sort_values()

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_te, y_pred)

    return {
        "accuracy": acc, "feature_importance": fi,
        "y_te": y_te, "y_pred": y_pred,
        "confusion_matrix": cm,
        "class_dist": np.bincount(y_all, minlength=3),
    }

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (CACHED)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(tickers_dict, start, end):
    prices  = {}
    returns = {}
    for name, ticker in tickers_dict.items():
        try:
            raw = yf.download(ticker, start=start, end=end,
                              auto_adjust=True, progress=False)
            if raw.empty:
                continue
            s = raw["Close"].squeeze()
            prices[name]  = s
            returns[name] = np.log(s / s.shift(1)).dropna()
        except Exception:
            pass
    return prices, returns

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown("### 📌 Asset Selection")
    selected_assets = st.multiselect(
        "Choose assets:",
        options=list(PRESET_TICKERS.keys()),
        default=["S&P 500", "JPMorgan Chase"],
    )
    custom_ticker = st.text_input("Or add custom ticker (e.g. MSFT):", "")
    if custom_ticker.strip():
        selected_assets.append(custom_ticker.strip().upper())

    st.markdown("### 📅 Date Range")
    start_date = st.date_input("Start date", pd.to_datetime("2010-01-01"))
    end_date   = st.date_input("End date",   pd.to_datetime("2024-12-31"))

    st.markdown("### 🎯 Risk Parameters")
    confidence = st.select_slider(
        "Confidence level:",
        options=[0.90, 0.95, 0.99],
        value=0.99,
        format_func=lambda x: f"{int(x*100)}%",
    )
    rolling_window = st.slider("Rolling window (days):", 100, 500, 250, 25)

    st.markdown("### 🤖 ML Options")
    run_xgb     = st.checkbox("XGBoost Quantile VaR", value=True)
    run_iforest = st.checkbox("Isolation Forest (Stress Detection)", value=True)
    run_rf      = st.checkbox("Random Forest (Basel Zone Classifier)", value=True)

    run_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">📊 VaR & ES Risk Dashboard</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Comparative Analysis of Value-at-Risk (VaR) and '
    'Expected Shortfall (ES) with Backtesting under Stress Market Conditions '
    '+ ML Extension</p>',
    unsafe_allow_html=True
)

if not run_btn:
    st.info("👈 Configure assets and parameters in the sidebar, then click **Run Analysis**.")
    st.markdown("""
    ### What this dashboard does
    This capstone dashboard implements **6 traditional risk methods** and **4 ML algorithms**:

    | Phase | Description |
    |-------|-------------|
    | **Phase 1** | Data loading, log-returns, descriptive statistics |
    | **Phase 2** | Point VaR & ES estimates (Historical, Normal, Student-t, Monte Carlo) |
    | **Phase 3** | Rolling 250-day VaR & ES over time |
    | **Phase 4** | Backtesting: Kupiec POF, Christoffersen, Basel Traffic Light |
    | **Phase 5** | Stress testing within GFC / COVID / Rate Hike windows |
    | **Phase 6** | Correlation, QQ plots, ES-VaR gap, exception heatmap |
    | **ML-1** | XGBoost Quantile Regression — direct VaR prediction |
    | **ML-2** | Isolation Forest — auto-detect stress regimes |
    | **ML-3** | Random Forest — predict Basel traffic light zone |
    """)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
tickers_to_load = {}
for a in selected_assets:
    tickers_to_load[a] = PRESET_TICKERS.get(a, a)

with st.spinner("📡 Downloading market data from Yahoo Finance..."):
    prices, returns = load_data(
        tickers_to_load,
        str(start_date),
        str(end_date)
    )

if not returns:
    st.error("No data returned. Please check tickers and date range.")
    st.stop()

available = list(returns.keys())
st.success(f"✅ Loaded data for: {', '.join(available)}")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📈 Phase 1: Data",
    "📐 Phase 2: VaR & ES",
    "📊 Phase 3: Rolling",
    "✅ Phase 4: Backtest",
    "🔥 Phase 5: Stress",
    "📉 Phase 6: Analysis",
    "🤖 ML Models",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Phase 1 — Data Collection & Exploratory Analysis")

    # Descriptive stats
    rows = []
    for name, r in returns.items():
        rows.append({
            "Asset": name, "Obs": len(r),
            "Mean (%)": round(r.mean()*100, 4),
            "Std (%)":  round(r.std()*100,  4),
            "Skewness": round(r.skew(), 4),
            "Kurtosis": round(r.kurt(), 4),
            "Min (%)":  round(r.min()*100,  4),
            "Max (%)":  round(r.max()*100,  4),
        })
    df_stats = pd.DataFrame(rows)
    st.markdown("#### Descriptive Statistics")
    st.dataframe(df_stats.set_index("Asset"), use_container_width=True)

    # Return time series
    st.markdown("#### Return Time Series")
    n = len(returns)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5*n))
    if n == 1: axes = [axes]
    for ax, (name, r) in zip(axes, returns.items()):
        ax.plot(r.index, r*100, color="#2E86AB", lw=0.6, alpha=0.8)
        for label, s, e, c in CRISIS_PERIODS:
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                       alpha=0.2, color=c, label=label)
        ax.set_title(name, fontsize=11)
        ax.set_ylabel("Return (%)")
        handles = [mpatches.Patch(color=c, alpha=0.5, label=l)
                   for l, _, _, c in CRISIS_PERIODS]
        ax.legend(handles=handles, fontsize=7, loc="upper right")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Distribution
    st.markdown("#### Return Distributions vs Normal")
    fig2, axes2 = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1: axes2 = [axes2]
    for ax, (name, r) in zip(axes2, returns.items()):
        ax.hist(r*100, bins=80, density=True, color="#2E86AB",
                alpha=0.6, label="Empirical")
        x = np.linspace(r.min()*100, r.max()*100, 300)
        ax.plot(x, norm.pdf(x, r.mean()*100, r.std()*100),
                color="#C73E1D", lw=2, label="Normal fit")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Return (%)")
        ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — VAR & ES POINT ESTIMATES
# ═══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Phase 2 — VaR & ES Point Estimates (4 Methods)")

    with st.spinner("Computing risk estimates..."):
        rows2 = []
        for name, r in returns.items():
            arr = r.values
            for conf_val in [0.95, 0.99]:
                for method, fn in METHODS.items():
                    try:
                        var, es = fn(arr, conf_val)
                        rows2.append({
                            "Asset": name, "Confidence": f"{int(conf_val*100)}%",
                            "Method": method,
                            "VaR (%)": round(var*100, 4),
                            "ES (%)":  round(es*100, 4),
                            "ES/VaR":  round(es/var, 4) if var != 0 else np.nan,
                        })
                    except Exception:
                        pass
        df_point = pd.DataFrame(rows2)

    st.dataframe(df_point, use_container_width=True)

    # Bar chart at selected confidence
    conf_label = f"{int(confidence*100)}%"
    df_sel = df_point[df_point["Confidence"] == conf_label].copy()
    assets  = df_sel["Asset"].unique()
    methods = df_sel["Method"].unique()

    fig3, axes3 = plt.subplots(1, len(assets), figsize=(5*len(assets), 5))
    if len(assets) == 1: axes3 = [axes3]
    fig3.suptitle(f"VaR vs ES at {conf_label} Confidence (All Methods)", fontsize=13)
    for ax, asset in zip(axes3, assets):
        sub = df_sel[df_sel["Asset"] == asset]
        x, w = np.arange(len(methods)), 0.35
        b1 = ax.bar(x - w/2, sub["VaR (%)"].values, w,
                    label="VaR", color="#2E86AB", alpha=0.85)
        b2 = ax.bar(x + w/2, sub["ES (%)"].values,  w,
                    label="ES",  color="#C73E1D", alpha=0.85)
        ax.set_title(asset, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, fontsize=8)
        ax.set_ylabel("Risk (%)")
        ax.legend(fontsize=8)
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h+0.02,
                    f"{h:.2f}%", ha="center", va="bottom", fontsize=6)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — ROLLING VAR & ES
# ═══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader(f"Phase 3 — Rolling {rolling_window}-Day VaR & ES at {int(confidence*100)}%")

    with st.spinner("Computing rolling risk estimates (this may take a moment)..."):
        all_rolling = {}
        for name, r in returns.items():
            arr, idx = r.values, r.index
            n_obs, W = len(arr), rolling_window
            records  = {m: {"VaR": [], "ES": []} for m in METHODS}
            roll_idx = []
            for i in range(W, n_obs):
                window = arr[i-W:i]
                roll_idx.append(idx[i])
                for method, fn in METHODS.items():
                    try:
                        var, es = fn(window, confidence)
                        records[method]["VaR"].append(var*100)
                        records[method]["ES"].append(es*100)
                    except Exception:
                        records[method]["VaR"].append(np.nan)
                        records[method]["ES"].append(np.nan)
            all_rolling[name] = {
                "index":  roll_idx,
                "data":   records,
                "actual": r.iloc[W:] * 100,
            }

    selected_asset = st.selectbox("Select asset:", available)
    bundle = all_rolling[selected_asset]
    idx_r  = bundle["index"]
    data_r = bundle["data"]
    act_r  = bundle["actual"]

    fig4, axes4 = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig4.suptitle(f"Rolling {int(confidence*100)}% VaR & ES: {selected_asset}", fontsize=13)

    for ax, metric in zip(axes4, ["VaR", "ES"]):
        for label, s, e, c in CRISIS_PERIODS:
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                       alpha=0.15, color=c, label=label)
        for method in METHODS:
            if method in data_r:
                ax.plot(idx_r, data_r[method][metric],
                        lw=1.2, alpha=0.85,
                        color=COLORS[method], label=method)
        ax.plot(act_r.index, -act_r.values,
                lw=0.5, color="black", alpha=0.3, label="Actual loss")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_ylabel(f"{metric} (%)")
        ax.set_title(f"Rolling {int(confidence*100)}% {metric}")
        handles, labels_ = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels_):
            if l not in seen: seen[l] = h
        ax.legend(seen.values(), seen.keys(), fontsize=7, ncol=3)

    plt.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Phase 4 — Backtesting Framework")
    st.markdown("""
    - **Kupiec POF Test** — Tests whether exception frequency matches expected rate
    - **Christoffersen Test** — Tests for exception clustering
    - **Basel Traffic Light** — 🟢 ≤4 | 🟡 5–9 | 🔴 ≥10 exceptions per 250 days
    """)

    with st.spinner("Running backtests..."):
        backtest_rows = []
        exception_store = {}
        for name, bundle in all_rolling.items():
            idx_b  = bundle["index"]
            actual = bundle["actual"].values
            data_b = bundle["data"]
            exception_store[name] = {}
            for method in METHODS:
                var_series = np.array(data_b[method]["VaR"])
                exc_bool   = (-actual) > var_series
                n_exc, n_obs = int(exc_bool.sum()), len(var_series)
                lr_k, pval_k, pass_k = kupiec_pof(n_exc, n_obs, confidence)
                lr_c, pval_c, pass_c = christoffersen_test(exc_bool.astype(int))
                tl = traffic_light(n_exc, n_obs)
                exception_store[name][method] = {"bool": exc_bool, "index": idx_b}
                backtest_rows.append({
                    "Asset": name, "Method": method,
                    "Observations": n_obs,
                    "Exceptions": n_exc,
                    "Exc Rate (%)": round(n_exc/n_obs*100, 2),
                    "Expected (%)": round((1-confidence)*100, 2),
                    "Kupiec LR": lr_k, "Kupiec p-val": pval_k,
                    "Kupiec Pass": "✓ Pass" if pass_k else "✗ Fail",
                    "Christo LR": lr_c, "Christo p-val": pval_c,
                    "Christo Pass": "✓ Pass" if pass_c else "✗ Fail",
                    "Traffic Light": tl,
                })
        df_backtest = pd.DataFrame(backtest_rows)

    # KPI summary
    col1, col2, col3 = st.columns(3)
    n_pass_k = (df_backtest["Kupiec Pass"] == "✓ Pass").sum()
    n_pass_c = (df_backtest["Christo Pass"] == "✓ Pass").sum()
    n_green  = (df_backtest["Traffic Light"] == "🟢 Green").sum()
    col1.metric("Kupiec Pass Rate",       f"{n_pass_k}/{len(df_backtest)}")
    col2.metric("Christoffersen Pass Rate",f"{n_pass_c}/{len(df_backtest)}")
    col3.metric("Basel Green Zone",       f"{n_green}/{len(df_backtest)}")

    st.dataframe(df_backtest[[
        "Asset","Method","Exceptions","Exc Rate (%)","Expected (%)",
        "Kupiec Pass","Christo Pass","Traffic Light"
    ]], use_container_width=True)

    # Exception heatmap
    pivot_bt = df_backtest.pivot_table(
        index="Asset", columns="Method",
        values="Exc Rate (%)", aggfunc="mean"
    )
    fig5, ax5 = plt.subplots(figsize=(10, max(3, len(available))))
    sns.heatmap(pivot_bt, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, ax=ax5, annot_kws={"size": 11})
    ax5.set_title(f"Exception Rate Heatmap (%) at {int(confidence*100)}% VaR\n"
                  f"Expected: {(1-confidence)*100:.0f}%", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig5, use_container_width=True)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — STRESS TESTING
# ═══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("Phase 5 — Stress Testing & Crisis Analysis")

    with st.spinner("Running stress analysis..."):
        crisis_rows = []
        for name, r in returns.items():
            for crisis_label, s, e, _ in CRISIS_PERIODS:
                try:
                    crisis_r = r.loc[s:e]
                    if len(crisis_r) < 20: continue
                    arr      = crisis_r.values
                    max_loss = -arr.min() * 100
                    avg_loss = -arr.mean() * 100
                    for method, fn in METHODS.items():
                        try:
                            var, es = fn(arr, confidence)
                            crisis_rows.append({
                                "Asset": name, "Crisis": crisis_label, "Method": method,
                                "VaR (%)": round(var*100, 4),
                                "ES (%)":  round(es*100,  4),
                                "Max Loss (%)": round(max_loss, 4),
                                "Avg Loss (%)": round(avg_loss, 4),
                                "ES/VaR": round(es/var, 4) if var != 0 else np.nan,
                            })
                        except Exception:
                            pass
                except Exception:
                    pass
        df_stress = pd.DataFrame(crisis_rows)

    if df_stress.empty:
        st.warning("No crisis data available for the selected date range.")
    else:
        st.dataframe(df_stress[["Asset","Crisis","Method",
                                 "VaR (%)","ES (%)","Max Loss (%)","ES/VaR"]],
                     use_container_width=True)

        sub_stress = df_stress[df_stress["Method"] == "Historical"]
        crises_u   = sub_stress["Crisis"].unique()
        n_c = len(crises_u)
        fig6, axes6 = plt.subplots(1, n_c, figsize=(6*n_c, 5))
        if n_c == 1: axes6 = [axes6]
        fig6.suptitle(f"VaR vs ES vs Max Loss During Crisis Periods "
                      f"(Historical, {int(confidence*100)}%)", fontsize=13)
        for ax, crisis in zip(axes6, crises_u):
            csub = sub_stress[sub_stress["Crisis"] == crisis]
            x, w = np.arange(len(csub)), 0.28
            ax.bar(x - w, csub["VaR (%)"].values, w, label=f"VaR {int(confidence*100)}%",
                   color="#2E86AB", alpha=0.85)
            ax.bar(x,     csub["ES (%)"].values,  w, label=f"ES {int(confidence*100)}%",
                   color="#C73E1D", alpha=0.85)
            ax.scatter(x + w/2, csub["Max Loss (%)"].values,
                       color="black", zorder=5, s=70, marker="D",
                       label="Max 1-Day Loss")
            ax.set_xticks(x)
            ax.set_xticklabels(csub["Asset"].values, rotation=15, fontsize=8)
            ax.set_title(crisis, fontsize=11)
            ax.set_ylabel("Risk / Loss (%)")
            ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig6, use_container_width=True)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("Phase 6 — Final Analysis & Visualizations")

    col_a, col_b = st.columns(2)

    # Correlation heatmap
    with col_a:
        st.markdown("#### Correlation Matrix")
        df_r = pd.DataFrame({n: r for n, r in returns.items()}).dropna()
        fig7, ax7 = plt.subplots(figsize=(6, 5))
        sns.heatmap(df_r.corr(), annot=True, fmt=".3f", cmap="RdYlGn",
                    center=0, ax=ax7, linewidths=0.5, annot_kws={"size": 11})
        ax7.set_title("Asset Return Correlations", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig7, use_container_width=True)
        plt.close()

    # QQ plots
    with col_b:
        st.markdown("#### QQ Plot (Fat Tails)")
        qq_asset = st.selectbox("Asset for QQ:", available, key="qq")
        fig8, ax8 = plt.subplots(figsize=(6, 5))
        stats.probplot(returns[qq_asset].values, dist="norm", plot=ax8)
        ax8.set_title(f"{qq_asset} — QQ vs Normal", fontsize=11)
        ax8.get_lines()[0].set(markersize=2, alpha=0.4, color="#2E86AB")
        ax8.get_lines()[1].set(color="#C73E1D", linewidth=1.5)
        plt.tight_layout()
        st.pyplot(fig8, use_container_width=True)
        plt.close()

    # ES-VaR gap
    st.markdown("#### ES − VaR Gap (The Core Argument for ES over VaR)")
    df99 = df_point[df_point["Confidence"] == f"{int(confidence*100)}%"].copy()
    df99["Gap (ES-VaR) %"] = df99["ES (%)"] - df99["VaR (%)"]
    assets_u  = df99["Asset"].unique()
    methods_u = df99["Method"].unique()
    x_g, w_g  = np.arange(len(assets_u)), 0.18

    fig9, ax9 = plt.subplots(figsize=(12, 5))
    for i, method in enumerate(methods_u):
        sub_g = df99[df99["Method"] == method]
        gaps  = []
        for a in assets_u:
            row = sub_g[sub_g["Asset"] == a]
            gaps.append(row["Gap (ES-VaR) %"].values[0] if len(row) > 0 else 0)
        ax9.bar(x_g + i * w_g - 0.27, gaps, w_g, label=method,
                color=list(COLORS.values())[i], alpha=0.85)
    ax9.set_xticks(x_g)
    ax9.set_xticklabels(assets_u, fontsize=10)
    ax9.set_ylabel("ES − VaR (%)")
    ax9.set_title(f"ES vs VaR Gap at {int(confidence*100)}%: How Much More Risk Does ES Capture?")
    ax9.axhline(0, color="gray", lw=0.5)
    ax9.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig9, use_container_width=True)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 7 — ML MODELS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader("Phase 7 — Machine Learning Risk Models")

    ml_tab1, ml_tab2, ml_tab3 = st.tabs([
        "📦 XGBoost Quantile VaR",
        "🔍 Isolation Forest",
        "🌲 Random Forest Classifier",
    ])

    # ── ML TAB 1: XGBoost ────────────────────────────────────────────────
    with ml_tab1:
        st.markdown("""
        #### XGBoost Quantile Regression
        Directly predicts the 1st percentile of the return distribution — i.e., 99% VaR —
        without any distributional assumption. Uses rolling volatility, skewness, kurtosis,
        and lagged returns as features.
        """)

        if not run_xgb:
            st.info("Enable XGBoost in the sidebar to run this model.")
        else:
            xgb_asset = st.selectbox("Asset:", available, key="xgb_asset")
            with st.spinner(f"Training XGBoost for {xgb_asset}..."):
                xgb_res = xgboost_var_predict(
                    returns[xgb_asset].values, confidence
                )

            col_xgb1, col_xgb2, col_xgb3, col_xgb4 = st.columns(4)
            col_xgb1.metric("Library",         xgb_res["lib"])
            col_xgb2.metric("Exceptions",      f"{xgb_res['n_exc']} / {xgb_res['n_obs']}")
            col_xgb3.metric("Exc Rate",         f"{xgb_res['exc_rate']}% (exp: {(1-confidence)*100:.0f}%)")
            col_xgb4.metric("Kupiec Test",      xgb_res["kupiec_pass"])
            st.metric("Basel Traffic Light",    xgb_res["traffic_light"])

            # Plot
            dates = xgb_res["dates"]
            fig_xgb, ax_xgb = plt.subplots(figsize=(14, 5))
            ax_xgb.plot(dates, -xgb_res["actual"], color="#2E86AB",
                        lw=0.7, alpha=0.7, label="Actual Daily Loss")
            ax_xgb.plot(dates, xgb_res["var_pred"], color="#6f42c1",
                        lw=1.5, linestyle="--",
                        label=f"XGBoost {int(confidence*100)}% VaR")
            exc_d = [d for d, b in zip(dates, xgb_res["exc_bool"]) if b]
            exc_l = [-xgb_res["actual"][i]
                     for i, b in enumerate(xgb_res["exc_bool"]) if b]
            ax_xgb.scatter(exc_d, exc_l, color="red", s=20, zorder=5,
                           label=f"Exceptions (n={xgb_res['n_exc']})")
            for label, s, e, c in CRISIS_PERIODS:
                ax_xgb.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                               alpha=0.12, color=c, label=label)
            ax_xgb.set_title(f"XGBoost Quantile VaR — {xgb_asset}", fontsize=12)
            ax_xgb.set_ylabel("Loss / VaR (%)")
            handles_, labels_ = ax_xgb.get_legend_handles_labels()
            seen = {}
            for h, l in zip(handles_, labels_):
                if l not in seen: seen[l] = h
            ax_xgb.legend(seen.values(), seen.keys(), fontsize=8, ncol=3)
            plt.tight_layout()
            st.pyplot(fig_xgb, use_container_width=True)
            plt.close()

    # ── ML TAB 2: Isolation Forest ────────────────────────────────────────
    with ml_tab2:
        st.markdown("""
        #### Isolation Forest — Automatic Stress Regime Detection
        Unsupervised anomaly detection that identifies stress periods purely from
        return data — no hand-labelled crisis windows needed.
        Detected anomalies are compared against GFC 2008, COVID-19, and Rate Hike 2022.
        """)

        if not run_iforest:
            st.info("Enable Isolation Forest in the sidebar to run this model.")
        else:
            contamination = st.slider("Contamination (expected anomaly fraction):",
                                      0.01, 0.15, 0.05, 0.01)
            iforest_asset = st.selectbox("Asset:", available, key="if_asset")

            with st.spinner(f"Running Isolation Forest for {iforest_asset}..."):
                r_arr   = returns[iforest_asset].values
                r_index = returns[iforest_asset].index
                if_res  = isolation_forest_detect(r_arr, r_index, contamination)

            col_if1, col_if2, col_if3, col_if4 = st.columns(4)
            col_if1.metric("Anomalies Detected", f"{if_res['n_detected']}")
            col_if2.metric("Precision vs Crisis GT", f"{if_res['precision']:.2f}")
            col_if3.metric("Recall vs Crisis GT",    f"{if_res['recall']:.2f}")
            col_if4.metric("F1 Score",               f"{if_res['f1']:.2f}")

            # Plot
            dates_if = if_res["dates"]
            scores   = if_res["scores"]
            anomaly  = if_res["anomaly"]

            fig_if, ax_if = plt.subplots(figsize=(14, 5))
            for crisis_label, start, end, color in CRISIS_PERIODS:
                ax_if.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                              alpha=0.15, color=color, label=f"GT: {crisis_label}")
            ax_if.plot(dates_if, scores, color="#2E86AB", lw=0.7, alpha=0.8,
                       label="Anomaly Score")
            anom_idx = np.where(anomaly == 1)[0]
            ax_if.scatter(dates_if[anom_idx], scores[anom_idx],
                          color="red", s=8, zorder=5, alpha=0.6,
                          label=f"Detected anomaly (n={len(anom_idx)})")
            ax_if.set_title(
                f"Isolation Forest — {iforest_asset} | "
                f"Prec={if_res['precision']:.2f} Rec={if_res['recall']:.2f} "
                f"F1={if_res['f1']:.2f}", fontsize=12
            )
            ax_if.set_ylabel("Anomaly Score")
            handles_, labels_ = ax_if.get_legend_handles_labels()
            seen = {}
            for h, l in zip(handles_, labels_):
                if l not in seen: seen[l] = h
            ax_if.legend(seen.values(), seen.keys(), fontsize=8, ncol=3)
            plt.tight_layout()
            st.pyplot(fig_if, use_container_width=True)
            plt.close()

    # ── ML TAB 3: Random Forest ───────────────────────────────────────────
    with ml_tab3:
        st.markdown("""
        #### Random Forest — Basel Traffic Light Zone Classifier
        Predicts whether a VaR model will fall in the Green / Yellow / Red Basel zone,
        using only market features (volatility, skewness, kurtosis, lagged returns).
        Provides regulators with a **forward-looking early-warning signal**.
        """)

        if not run_rf:
            st.info("Enable Random Forest in the sidebar to run this model.")
        elif len(available) < 2:
            st.warning("Random Forest requires at least 2 assets for a meaningful dataset. "
                       "Please select more assets.")
        else:
            with st.spinner("Training Random Forest classifier (all assets)..."):
                rf_res = random_forest_zone(returns, confidence)

            if rf_res is None:
                st.warning("Not enough class variety in the data. "
                            "Try a longer date range or more assets.")
            else:
                col_rf1, col_rf2 = st.columns(2)
                col_rf1.metric("Accuracy", f"{rf_res['accuracy']*100:.1f}%")
                class_dist = rf_res["class_dist"]
                col_rf2.markdown(
                    f"**Class distribution:** "
                    f"🟢 Green={class_dist[0]} | "
                    f"🟡 Yellow={class_dist[1] if len(class_dist)>1 else 0} | "
                    f"🔴 Red={class_dist[2] if len(class_dist)>2 else 0}"
                )

                fig_rf, axes_rf = plt.subplots(1, 2, figsize=(12, 5))
                fig_rf.suptitle("Random Forest: Basel Zone Classifier", fontsize=13)

                # Confusion matrix
                ax_cm = axes_rf[0]
                cm_labels = [l for l, c in
                             zip(["Green","Yellow","Red"], class_dist)
                             if c > 0]
                sns.heatmap(rf_res["confusion_matrix"], annot=True, fmt="d",
                            cmap="Blues", ax=ax_cm,
                            xticklabels=cm_labels,
                            yticklabels=cm_labels)
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                ax_cm.set_title(f"Confusion Matrix (Acc={rf_res['accuracy']:.3f})")

                # Feature importance
                ax_fi = axes_rf[1]
                fi = rf_res["feature_importance"]
                ax_fi.barh(fi.index, fi.values, color="#F18F01", alpha=0.85)
                ax_fi.set_title("Feature Importance")
                ax_fi.set_xlabel("Importance")

                plt.tight_layout()
                st.pyplot(fig_rf, use_container_width=True)
                plt.close()

                st.markdown("""
                **Interpretation:**
                - High `vol_21` / `vol_63` importance confirms rolling volatility
                  is the primary driver of exception zone.
                - High recall for 🔴 Red zone means the model can flag dangerous
                  periods before they occur — valuable for pre-emptive capital buffering.
                """)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.85rem;'>
    VaR & ES Capstone Dashboard · Built with Streamlit · 
    Methods: Historical · Normal · Student-t · Monte Carlo · XGBoost · Isolation Forest · Random Forest<br>
    References: Artzner et al. (1999) · BCBS FRTB (2016) · McNeil & Frey (2000)
</div>
""", unsafe_allow_html=True)
