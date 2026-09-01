"""
Statistical vs Machine Learning Stock Trend Prediction
========================================================
An educational, website-style Streamlit application that compares classical
statistical time-series modelling (ARIMA) against machine learning
classifiers (Logistic Regression, Random Forest, Gradient Boosting) for
predicting next-day stock price direction — with interactive 3D
visualizations of price, volume, and feature space.

Author : <Your Name>
Purpose: Portfolio project demonstrating data engineering, statistical
         modelling, machine learning, and interactive dashboard design.
Stack  : Python, Streamlit, yfinance, statsmodels, scikit-learn, Plotly

Run with:
    streamlit run stock_trend_comparison.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Try to import plotly with error handling
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    # Fallback to matplotlib if plotly is not available
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    st.warning("⚠️ Plotly not available. Using matplotlib fallback for visualizations.")

# Try to import other libraries
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.error("❌ scikit-learn not available. Please install scikit-learn.")

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    st.error("❌ statsmodels not available. Please install statsmodels.")

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# Navigation pages
PAGES: List[Tuple[str, str, str]] = [
    ("home", "🏠 Home", "Overview"),
    ("data", "📊 Data Explorer", "Explore Data"),
    ("market3d", "🌐 3D Market View", "3D Visualizations"),
    ("stats", "📐 Statistical Models", "ARIMA"),
    ("ml", "🤖 Machine Learning", "ML Models"),
    ("compare", "⚖️ Comparison", "Compare Results"),
    ("theory", "📚 Theory", "Learn More"),
]

ACCENT = "#22d3ee"
ACCENT_2 = "#a78bfa"
UP_COLOR = "#22c55e"
DOWN_COLOR = "#f43f5e"


# =====================================================================
# Page configuration & styling
# =====================================================================
def configure_page() -> None:
    st.set_page_config(
        page_title="Statistical vs ML Stock Trend Prediction",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
            .main > div {{padding-top: 0.5rem;}}
            footer {{visibility: hidden;}}
            #MainMenu {{visibility: hidden;}}
            
            h1, h2, h3 {{font-weight: 700; letter-spacing: -0.02em;}}
            
            div[data-testid="stMetricValue"] {{font-size: 1.8rem; font-weight: 700;}}
            div[data-testid="stMetric"] {{
                background: linear-gradient(160deg, rgba(34,211,238,0.08), rgba(167,139,250,0.08));
                border: 1px solid rgba(148,163,184,0.15);
                border-radius: 16px;
                padding: 18px 20px 8px 20px;
                backdrop-filter: blur(10px);
                transition: all 0.3s ease;
            }}
            div[data-testid="stMetric"]:hover {{
                border-color: rgba(34,211,238,0.4);
                transform: translateY(-2px);
                box-shadow: 0 8px 30px rgba(34,211,238,0.1);
            }}
            
            .hero {{
                background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
                border: 1px solid rgba(148,163,184,0.15);
                border-radius: 24px;
                padding: 50px 48px;
                margin-bottom: 30px;
                position: relative;
                overflow: hidden;
            }}
            .hero::before {{
                content: '';
                position: absolute;
                top: -50%;
                right: -20%;
                width: 600px;
                height: 600px;
                background: radial-gradient(circle, rgba(34,211,238,0.08) 0%, transparent 70%);
                border-radius: 50%;
                pointer-events: none;
            }}
            .hero h1 {{
                font-size: 3rem;
                margin-bottom: 8px;
                background: linear-gradient(90deg, {ACCENT}, {ACCENT_2}, {ACCENT});
                background-size: 200% auto;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: shimmer 3s ease-in-out infinite;
                position: relative;
                z-index: 1;
            }}
            @keyframes shimmer {{
                0%, 100% {{ background-position: 0% center; }}
                50% {{ background-position: 200% center; }}
            }}
            .hero p {{
                color: #cbd5e1;
                font-size: 1.1rem;
                max-width: 750px;
                line-height: 1.7;
                position: relative;
                z-index: 1;
            }}
            .hero-tag {{
                display: inline-block;
                background: rgba(34,211,238,0.12);
                color: {ACCENT};
                border: 1px solid rgba(34,211,238,0.25);
                border-radius: 999px;
                padding: 4px 16px;
                font-size: 0.75rem;
                font-weight: 600;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                margin-bottom: 16px;
                position: relative;
                z-index: 1;
            }}
            
            .nav-card {{
                background: rgba(148,163,184,0.05);
                border: 1px solid rgba(148,163,184,0.12);
                border-radius: 18px;
                padding: 22px 20px 18px 20px;
                height: 100%;
                transition: all 0.3s ease;
                cursor: pointer;
                position: relative;
                overflow: hidden;
            }}
            .nav-card:hover {{
                border-color: rgba(34,211,238,0.4);
                background: rgba(34,211,238,0.06);
                transform: translateY(-4px);
                box-shadow: 0 12px 40px rgba(34,211,238,0.08);
            }}
            .nav-card .icon {{font-size: 2rem; margin-bottom: 8px; display: block;}}
            .nav-card h4 {{margin-bottom: 6px; color: #e2e8f0;}}
            .nav-card p {{color: #94a3b8; font-size: 0.9rem; margin-bottom: 0; line-height: 1.5;}}
            
            .stat-box {{
                background: rgba(34,211,238,0.06);
                border-left: 3px solid {ACCENT};
                border-radius: 8px;
                padding: 16px 20px;
                margin: 10px 0;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# Fallback visualization functions when plotly is not available
# =====================================================================
def create_matplotlib_figure(df, ticker, title="Stock Price"):
    """Create a matplotlib figure as fallback when plotly is not available."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['Close'], color=UP_COLOR, linewidth=2, label='Close')
    ax.set_title(f"{ticker} - {title}", fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def create_matplotlib_candlestick(df, ticker):
    """Create a simplified candlestick chart using matplotlib."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot close price
    ax.plot(df.index, df['Close'], color=UP_COLOR, linewidth=2, label='Close')
    ax.fill_between(df.index, df['Low'], df['High'], alpha=0.2, color='gray')
    
    ax.set_title(f"{ticker} - Price Chart", fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


# =====================================================================
# Data layer
# =====================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_price_data(ticker: str, start, end) -> pd.DataFrame:
    """
    Download OHLCV data for a ticker from Yahoo Finance.
    """
    try:
        raw = yf.download(ticker, start=start, end=end, progress=False)
    except Exception:
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    return raw[cols].dropna()


# =====================================================================
# Feature engineering
# =====================================================================
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index — a momentum oscillator bounded in [0, 100]."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def engineer_features(data: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Build the technical-indicator and lagged-return feature set used by
    the ML classifiers, plus the binary next-day-direction target.
    """
    df = data.copy()
    df["Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["EMA_12"] = df["Close"].ewm(span=12).mean()
    df["EMA_26"] = df["Close"].ewm(span=26).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["RSI"] = compute_rsi(df["Close"], 14)
    df["Volatility"] = df["Return"].rolling(lookback).std()
    df["Volume_Change"] = df["Volume"].pct_change()

    for lag in range(1, lookback + 1):
        df[f"Return_Lag_{lag}"] = df["Return"].shift(lag)

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df.dropna()


# =====================================================================
# Statistical model (ARIMA)
# =====================================================================
@dataclass
class ArimaResult:
    """Container for everything the statistical-models page needs."""
    train: pd.Series
    test: pd.Series
    forecast: pd.Series
    conf_int: pd.DataFrame
    residuals: pd.Series
    directional_accuracy: float
    error: str | None = None


def fit_arima(close: pd.Series, test_size: float, order: Tuple[int, int, int] = (5, 1, 2)) -> ArimaResult:
    """
    Fit an ARIMA model on a chronological train split and forecast the
    held-out test window, returning point forecasts, 95% confidence
    intervals, residuals, and a directional-accuracy score.
    """
    split = int(len(close) * (1 - test_size))
    train, test = close.iloc[:split], close.iloc[split:]

    try:
        fitted = ARIMA(train, order=order).fit()
        forecast = fitted.forecast(steps=len(test))
        conf_int = fitted.get_forecast(steps=len(test)).conf_int(alpha=0.05)

        actual_dir = (test.values[1:] > test.values[:-1]).astype(int)
        pred_dir = (forecast.values[1:] > forecast.values[:-1]).astype(int)
        n = min(len(actual_dir), len(pred_dir))
        acc = accuracy_score(actual_dir[:n], pred_dir[:n]) if n > 0 else 0.5

        return ArimaResult(
            train=train, test=test, forecast=forecast, conf_int=conf_int,
            residuals=fitted.resid, directional_accuracy=acc,
        )
    except Exception as exc:
        return ArimaResult(
            train=train, test=test, forecast=pd.Series(dtype=float),
            conf_int=pd.DataFrame(), residuals=pd.Series(dtype=float),
            directional_accuracy=0.5, error=str(exc),
        )


# =====================================================================
# Machine learning models
# =====================================================================
@dataclass
class ModelResult:
    accuracy: float
    predictions: np.ndarray
    model: object


def train_ml_models(
    df_feat: pd.DataFrame, test_size: float, random_state: int = RANDOM_STATE
) -> Tuple[Dict[str, ModelResult], List[str], pd.Series, np.ndarray]:
    """Train Logistic Regression, Random Forest, and Gradient Boosting classifiers."""
    exclude = {"Open", "High", "Low", "Close", "Volume", "Target", "Return", "Log_Return"}
    feature_cols = [c for c in df_feat.columns if c not in exclude]

    X, y = df_feat[feature_cols], df_feat["Target"]
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, random_state=random_state
        ),
    }

    results: Dict[str, ModelResult] = {}
    for name, model in candidates.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        results[name] = ModelResult(
            accuracy=accuracy_score(y_test, preds), predictions=preds, model=model
        )

    return results, feature_cols, y_test, X_test_s


# =====================================================================
# Visualization builders with plotly fallback
# =====================================================================
def build_3d_price_landscape(df: pd.DataFrame, ticker: str, window: int = 120):
    """
    A stunning 3D terrain of OHLC prices over time.
    """
    if not PLOTLY_AVAILABLE:
        return create_matplotlib_figure(df.tail(window), ticker, "Price Landscape")
    
    d = df.tail(window)
    categories = ["Open", "Low", "Close", "High"]
    z = np.array([d[c].values for c in categories], dtype=float)
    x = np.arange(len(d))
    y = np.arange(len(categories))

    fig = go.Figure(
        data=[
            go.Surface(
                x=x, y=y, z=z,
                colorscale=[
                    [0, '#0f172a'], [0.25, '#1e3a5f'], [0.5, '#22d3ee'],
                    [0.75, '#a78bfa'], [1.0, '#f43f5e']
                ],
                colorbar=dict(title="Price ($)"),
                contours={
                    "z": {
                        "show": True,
                        "usecolormap": True,
                        "highlightcolor": "white",
                        "project_z": True,
                    }
                },
                lighting=dict(ambient=0.65, diffuse=0.85, specular=0.3),
                hovertemplate="Day: %{x}<br>Price Type: %{y}<br>Price: $%{z:.2f}<extra></extra>",
            )
        ]
    )
    
    fig.update_layout(
        title=dict(
            text=f"{ticker} — 3D Price Landscape (last {window} trading days)",
            font=dict(size=20, weight=700),
            x=0.5,
            xanchor="center",
        ),
        scene=dict(
            xaxis=dict(title="Trading Day →", gridcolor="rgba(148,163,184,0.1)"),
            yaxis=dict(title="Price Type", tickvals=y, ticktext=categories, gridcolor="rgba(148,163,184,0.1)"),
            zaxis=dict(title="Price ($)", gridcolor="rgba(148,163,184,0.1)"),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.0)),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=650,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_3d_price_volume_time(df: pd.DataFrame, window: int = 180):
    """A dynamic 3D ribbon showing price and volume flowing through time."""
    if not PLOTLY_AVAILABLE:
        return create_matplotlib_figure(df.tail(window), "Price & Volume", "Price & Volume Flow")
    
    d = df.tail(window)
    x = np.arange(len(d))
    volume_norm = (d["Volume"] - d["Volume"].min()) / (d["Volume"].max() - d["Volume"].min()) * 0.8 + 0.1
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=x,
        y=d["Close"],
        z=volume_norm * d["Close"].max() * 0.3,
        mode="lines",
        line=dict(color=d["Close"], colorscale="Viridis", width=6, showscale=True, colorbar=dict(title="Close ($)")),
        name="Price",
        hovertemplate="Day: %{x}<br>Close: $%{y:.2f}<br>Volume: %{text:,.0f}<extra></extra>",
        text=d["Volume"].values,
    ))
    
    fig.add_trace(go.Scatter3d(
        x=x,
        y=d["Close"] * 0.95,
        z=volume_norm * d["Close"].max() * 0.3,
        mode="markers",
        marker=dict(size=volume_norm * 15, color=d["Volume"], colorscale="Blues", opacity=0.6),
        name="Volume",
        hovertemplate="Day: %{x}<br>Volume: %{text:,.0f}<extra></extra>",
        text=d["Volume"].values,
    ))
    
    fig.update_layout(
        title=dict(text="Price & Volume Flow Through Time — 3D View", font=dict(size=20, weight=700), x=0.5, xanchor="center"),
        scene=dict(
            xaxis=dict(title="Trading Day →", gridcolor="rgba(148,163,184,0.1)"),
            yaxis=dict(title="Close Price ($)", gridcolor="rgba(148,163,184,0.1)"),
            zaxis=dict(title="Volume Intensity", showticklabels=False, gridcolor="rgba(148,163,184,0.1)"),
            camera=dict(eye=dict(x=1.2, y=-1.8, z=1.0)),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=650,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0.3)"),
    )
    return fig


def build_3d_feature_scatter(df_feat: pd.DataFrame, sample_size: int = 800):
    """An immersive 3D feature space visualization."""
    if not PLOTLY_AVAILABLE:
        fig, ax = plt.subplots(figsize=(12, 6))
        sample = df_feat.tail(min(sample_size, len(df_feat)))
        colors = ['green' if t == 1 else 'red' for t in sample['Target']]
        ax.scatter(sample['Return'] * 100, sample['Volatility'] * 100, c=colors, alpha=0.5)
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Volatility (%)')
        ax.set_title('Feature Space: Return vs Volatility')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    sample = df_feat.tail(min(sample_size, len(df_feat)))
    colors = np.where(sample["Target"] == 1, UP_COLOR, DOWN_COLOR)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=sample["Return"] * 100,
        y=sample["Volatility"] * 100,
        z=sample["RSI"],
        mode="markers",
        marker=dict(size=5, color=colors, opacity=0.7, line=dict(width=0.5, color="rgba(255,255,255,0.2)")),
        text=[f"Return: {r:.2f}%<br>Volatility: {v:.2f}%<br>RSI: {rsi:.1f}<br>Next Day: {'↑ Up' if t == 1 else '↓ Down'}"
              for r, v, rsi, t in zip(sample["Return"] * 100, sample["Volatility"] * 100, sample["RSI"], sample["Target"])],
        hoverinfo="text",
        name="Points",
    )])
    
    fig.update_layout(
        title=dict(text="3D Feature Space: Return vs Volatility vs RSI", font=dict(size=20, weight=700), x=0.5, xanchor="center"),
        scene=dict(
            xaxis=dict(title="Daily Return (%)", gridcolor="rgba(148,163,184,0.1)"),
            yaxis=dict(title="Volatility (%)", gridcolor="rgba(148,163,184,0.1)"),
            zaxis=dict(title="RSI", range=[0, 100], gridcolor="rgba(148,163,184,0.1)"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=650,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    
    # Add custom legend
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", marker=dict(size=10, color=UP_COLOR), name="↑ Up Next Day"))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", marker=dict(size=10, color=DOWN_COLOR), name="↓ Down Next Day"))
    
    return fig


def build_3d_arima_ribbon(result: ArimaResult):
    """A sophisticated 3D view of ARIMA forecast with confidence bands."""
    if not PLOTLY_AVAILABLE:
        fig, ax = plt.subplots(figsize=(12, 6))
        if not result.error and not result.forecast.empty:
            ax.plot(range(len(result.test)), result.test.values, color='green', label='Actual', linewidth=2)
            ax.plot(range(len(result.forecast)), result.forecast.values, color='orange', label='Forecast', linestyle='--', linewidth=2)
            ax.fill_between(range(len(result.test)), 
                           result.conf_int.iloc[:, 0].values, 
                           result.conf_int.iloc[:, 1].values, 
                           alpha=0.2, color='orange')
            ax.set_title('ARIMA Forecast vs Actual')
            ax.set_xlabel('Test-set Day')
            ax.set_ylabel('Price ($)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
        return fig
    
    fig = go.Figure()
    
    if result.error or result.forecast.empty:
        fig.update_layout(title=dict(text="ARIMA Forecast Unavailable", font=dict(size=18, weight=700), x=0.5, xanchor="center"), height=500)
        return fig

    x = np.arange(len(result.test))
    
    fig.add_trace(go.Scatter3d(
        x=x, y=result.test.values, z=np.zeros_like(x),
        mode="lines", name="Actual", line=dict(color=UP_COLOR, width=5),
        hovertemplate="Day %{x}<br>Actual: $%{y:.2f}<extra></extra>",
    ))
    
    fig.add_trace(go.Scatter3d(
        x=x, y=result.forecast.values, z=np.ones_like(x),
        mode="lines", name="ARIMA Forecast", line=dict(color="#f59e0b", width=5),
        hovertemplate="Day %{x}<br>Forecast: $%{y:.2f}<extra></extra>",
    ))
    
    lower = result.conf_int.iloc[:, 0].values
    upper = result.conf_int.iloc[:, 1].values
    
    x_band = np.concatenate([x, x[::-1]])
    y_band = np.concatenate([lower, upper[::-1]])
    z_band = np.ones_like(x_band)
    
    fig.add_trace(go.Scatter3d(
        x=x_band, y=y_band, z=z_band,
        mode="lines", fill="toself", fillcolor="rgba(245, 158, 11, 0.15)",
        line=dict(color="rgba(245, 158, 11, 0.3)", width=0),
        name="95% Confidence Interval",
    ))
    
    fig.update_layout(
        title=dict(text="ARIMA Forecast vs Actual — 3D View with Confidence Band", font=dict(size=20, weight=700), x=0.5, xanchor="center"),
        scene=dict(
            xaxis=dict(title="Test-set Day →", gridcolor="rgba(148,163,184,0.1)"),
            yaxis=dict(title="Price ($)", gridcolor="rgba(148,163,184,0.1)"),
            zaxis=dict(title="Series", tickvals=[0, 1], ticktext=["Actual", "Forecast"], gridcolor="rgba(148,163,184,0.1)"),
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=600,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0.3)"),
    )
    return fig


# =====================================================================
# Shared UI helpers
# =====================================================================
def render_nav() -> None:
    """A horizontal, website-style navigation bar at the top of every page."""
    if "page" not in st.session_state:
        st.session_state.page = "home"

    cols = st.columns(len(PAGES))
    for col, (key, label, _) in zip(cols, PAGES):
        is_active = st.session_state.page == key
        btn_type = "primary" if is_active else "secondary"
        if col.button(label, key=f"nav_{key}", use_container_width=True, type=btn_type):
            st.session_state.page = key
            st.rerun()
    st.markdown("<hr style='margin-top:0.3rem;margin-bottom:1.5rem;opacity:0.1;'>", unsafe_allow_html=True)


def goto(page_key: str) -> None:
    st.session_state.page = page_key
    st.rerun()


def render_sidebar() -> Tuple[str, datetime, datetime, int, float]:
    st.sidebar.title("⚙️ Configuration")
    
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <div style="font-size: 2.5rem;">📈</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: #e2e8f0;">Stock Trend Lab</div>
            <div style="font-size: 0.7rem; color: #64748b;">Statistical vs ML</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper().strip()
    start_date = st.sidebar.date_input("Start Date", value=datetime(2018, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime.now())
    lookback = st.sidebar.slider("Lookback Window (days)", 5, 60, 20, help="For ML feature engineering")
    test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100

    st.sidebar.markdown("---")
    
    with st.sidebar.expander("📖 Why Probability & Statistics?", expanded=True):
        st.markdown(
            """
            • Markets are **stochastic** → we model uncertainty  
            • **Stationarity** testing (ADF) is essential  
            • **Autocorrelation** reveals hidden structure  
            • **Confidence intervals** quantify uncertainty  
            • **Residual diagnostics** validate models  
            • **Volatility clustering** requires specialized models
            """
        )
    
    st.sidebar.markdown("---")
    
    # Show plotly status
    if PLOTLY_AVAILABLE:
        st.sidebar.success("✅ Plotly available")
    else:
        st.sidebar.warning("⚠️ Plotly not available - using matplotlib fallback")
    
    st.sidebar.caption("Built with ❤️ using Streamlit · yfinance · statsmodels · scikit-learn")
    return ticker, start_date, end_date, lookback, test_size


# =====================================================================
# Pages
# =====================================================================
def render_home_page(df: pd.DataFrame, df_feat: pd.DataFrame, ticker: str) -> None:
    latest_close = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2] if len(df) > 1 else latest_close
    change_pct = (latest_close / prev_close - 1) * 100 if prev_close else 0.0

    st.markdown(
        f"""
        <div class="hero">
            <span class="hero-tag">🎯 COMPARATIVE ANALYSIS</span>
            <h1>Statistical Probability Models <br>vs Machine Learning</h1>
            <p>
                A side-by-side comparison of classical statistical time-series forecasting
                (ARIMA, confidence intervals, stationarity testing) against modern machine
                learning classifiers (Logistic Regression, Random Forest, Gradient Boosting)
                for predicting next-day stock direction — brought to life with interactive
                3D visualizations of price, volume, and feature space.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📌 Ticker", ticker)
    with c2:
        st.metric("💰 Latest Close", f"${latest_close:.2f}", f"{change_pct:+.2f}%")
    with c3:
        st.metric("📅 Trading Days", f"{len(df):,}")
    with c4:
        st.metric("📊 Daily Volatility", f"{df_feat['Return'].std()*100:.2f}%")

    # Show visualization based on plotly availability
    fig = build_3d_price_landscape(df, ticker, window=90)
    if PLOTLY_AVAILABLE:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.pyplot(fig)

    st.markdown("### 🚀 Explore the Project")
    
    cards = [
        ("data", "📊", "Data Explorer", "Explore candlesticks, stationarity tests, and statistics."),
        ("market3d", "🌐", "3D Market View", "Immersive 3D visualizations of price, volume, and features."),
        ("stats", "📐", "Statistical Models", "ARIMA forecasting with confidence intervals & diagnostics."),
        ("ml", "🤖", "Machine Learning", "Logistic Regression, Random Forest & Gradient Boosting."),
        ("compare", "⚖️", "Comparison", "Head-to-head accuracy and model strengths."),
        ("theory", "📚", "Theory", "Why probability and statistics are indispensable."),
    ]
    
    for row_start in range(0, len(cards), 3):
        row = cards[row_start:row_start + 3]
        cols = st.columns(3)
        for col, (key, icon, title, desc) in zip(cols, row):
            with col:
                st.markdown(
                    f"""
                    <div class="nav-card">
                        <span class="icon">{icon}</span>
                        <h4>{title}</h4>
                        <p>{desc}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Explore →", key=f"open_{key}", use_container_width=True):
                    goto(key)


def render_data_page(df: pd.DataFrame, df_feat: pd.DataFrame, ticker: str) -> None:
    st.header("📊 Data Overview & Stationarity")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Days", len(df))
    with c2:
        st.metric("Latest Close", f"${df['Close'].iloc[-1]:.2f}")
    with c3:
        st.metric("Mean Daily Return", f"{df_feat['Return'].mean()*100:.3f}%")
    with c4:
        st.metric("Daily Volatility", f"{df_feat['Return'].std()*100:.3f}%")

    # Candlestick chart
    if PLOTLY_AVAILABLE:
        fig = go.Figure(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC",
                increasing=dict(line=dict(color=UP_COLOR)),
                decreasing=dict(line=dict(color=DOWN_COLOR)),
            )
        )
        fig.update_layout(
            title=dict(text=f"{ticker} Price — Candlestick Chart", font=dict(size=18, weight=700), x=0.5),
            xaxis_rangeslider_visible=False,
            height=500,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = create_matplotlib_candlestick(df, ticker)
        st.pyplot(fig)

    st.subheader("🔬 Augmented Dickey-Fuller Test (Stationarity)")
    st.markdown("*Statistical models like ARIMA require stationarity. Returns are usually stationary → we model returns or difference the price series.*")
    
    adf_price = adfuller(df_feat["Close"])
    adf_return = adfuller(df_feat["Return"].dropna())

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div class="stat-box">
                <strong>On Close Price</strong><br>
                ADF Statistic: {adf_price[0]:.4f}<br>
                p-value: {adf_price[1]:.4f}<br>
                {'✅ Stationary' if adf_price[1] < 0.05 else '❌ Non-stationary (needs differencing)'}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="stat-box">
                <strong>On Returns</strong><br>
                ADF Statistic: {adf_return[0]:.4f}<br>
                p-value: {adf_return[1]:.4f}<br>
                {'✅ Stationary' if adf_return[1] < 0.05 else '❌ Non-stationary'}
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # Show correlation heatmap
    st.subheader("📈 Feature Correlations")
    corr_cols = ["Return", "RSI", "Volatility", "Volume_Change", "Target"]
    corr_df = df_feat[corr_cols].corr()
    
    if PLOTLY_AVAILABLE:
        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.columns,
                colorscale="RdBu",
                zmid=0,
                text=corr_df.values.round(2),
                texttemplate="%{text}",
                textfont=dict(size=10),
            )
        )
        fig_corr.update_layout(
            title="Correlation Matrix of Key Features",
            height=450,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_df.values, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_df.columns)))
        ax.set_yticks(range(len(corr_df.columns)))
        ax.set_xticklabels(corr_df.columns)
        ax.set_yticklabels(corr_df.columns)
        plt.colorbar(im)
        st.pyplot(fig)


def render_market3d_page(df: pd.DataFrame, df_feat: pd.DataFrame, ticker: str, arima_result: ArimaResult) -> None:
    st.header("🌐 3D Market View")
    st.markdown(
        """
        Explore the market from three immersive perspectives: 
        **price as a physical landscape**, **price and volume as a flow through time**, 
        and the **ML feature space** as a terrain of risk and momentum.
        """
    )
    
    if not PLOTLY_AVAILABLE:
        st.warning("⚠️ Plotly is not available. Showing 2D visualizations instead of 3D.")

    window = st.slider("Trading days to render", 30, min(365, len(df)), min(120, len(df)), key="3d_window")

    tab1, tab2, tab3 = st.tabs(["🏔️ Price Landscape", "🌊 Price & Volume Flow", "🎯 Feature Space"])
    
    with tab1:
        fig = build_3d_price_landscape(df, ticker, window=window)
        if PLOTLY_AVAILABLE:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.pyplot(fig)
    
    with tab2:
        fig = build_3d_price_volume_time(df, window=window)
        if PLOTLY_AVAILABLE:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.pyplot(fig)
    
    with tab3:
        fig = build_3d_feature_scatter(df_feat)
        if PLOTLY_AVAILABLE:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.pyplot(fig)

    st.subheader("📐 ARIMA Forecast in 3D")
    fig = build_3d_arima_ribbon(arima_result)
    if PLOTLY_AVAILABLE:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.pyplot(fig)


def render_statistical_page(result: ArimaResult) -> None:
    st.header("📐 Statistical Probability Models — ARIMA")
    st.markdown(
        """
        **ARIMA (AutoRegressive Integrated Moving Average)** is a classic statistical time-series model.
        It relies on the assumption that residuals behave like white noise (i.i.d., approximately normal),
        and it produces **confidence intervals** — genuine probabilistic statements about future values.
        """
    )

    if result.error:
        st.error(f"ARIMA failed to fit: {result.error}")
        return

    st.success(f"ARIMA Directional Accuracy: **{result.directional_accuracy*100:.2f}%**")

    # Main forecast chart
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result.train.index, y=result.train, name="Train", line=dict(color="#60a5fa", width=2)))
        fig.add_trace(go.Scatter(x=result.test.index, y=result.test, name="Actual Test", line=dict(color=UP_COLOR, width=2.5)))
        fig.add_trace(go.Scatter(x=result.test.index, y=result.forecast, name="ARIMA Forecast", line=dict(color="#f59e0b", width=2.5, dash="dash")))
        fig.add_trace(go.Scatter(x=result.test.index, y=result.conf_int.iloc[:, 0], mode="lines", line=dict(color="rgba(245,158,11,0.2)", width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=result.test.index, y=result.conf_int.iloc[:, 1], fill="tonexty", mode="lines", line=dict(color="rgba(245,158,11,0.2)", width=0), name="95% Confidence Interval"))
        fig.update_layout(
            title=dict(text="ARIMA Forecast with 95% Probability Confidence Interval", font=dict(size=18, weight=700), x=0.5),
            height=500,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(result.train.index, result.train.values, color='blue', label='Train', linewidth=2)
        ax.plot(result.test.index, result.test.values, color='green', label='Actual Test', linewidth=2)
        ax.plot(result.test.index, result.forecast.values, color='orange', label='ARIMA Forecast', linestyle='--', linewidth=2)
        ax.fill_between(result.test.index, 
                       result.conf_int.iloc[:, 0].values, 
                       result.conf_int.iloc[:, 1].values, 
                       alpha=0.2, color='orange', label='95% CI')
        ax.set_title('ARIMA Forecast with 95% Probability Confidence Interval')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown(
        """
        **Why the confidence interval?** Under the model's probabilistic assumptions, there is a 
        **95% probability** that the true future value lies inside the shaded band — a form of statistical 
        inference that pure point-forecast ML models rarely provide out-of-the-box.
        """
    )

    st.subheader("🔬 Residual Diagnostics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Ljung-Box Test on Residuals** (should be white noise)")
        lb_test = acorr_ljungbox(result.residuals, lags=[10], return_df=True)
        st.dataframe(lb_test, use_container_width=True, height=100)
    
    with col2:
        if PLOTLY_AVAILABLE:
            fig_resid = go.Figure()
            fig_resid.add_trace(go.Histogram(x=result.residuals.values, nbinsx=30, marker_color="rgba(34,211,238,0.6)"))
            fig_resid.update_layout(title="Residuals Distribution", height=250, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_resid, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(result.residuals.values, bins=30, color='rgba(34,211,238,0.6)', edgecolor='white')
            ax.set_title('Residuals Distribution')
            ax.set_xlabel('Residual Value')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)


def render_ml_page(results: Dict[str, ModelResult], feature_cols: List[str], y_test: pd.Series) -> None:
    st.header("🤖 Machine Learning Models for Trend Classification")
    st.markdown(
        """
        Three machine learning classifiers trained on technical indicators and lagged returns:
        - **Logistic Regression** — Statistical baseline with probabilistic output (sigmoid)
        - **Random Forest** — Ensemble of decision trees capturing non-linear interactions
        - **Gradient Boosting** — Sequential boosting with strong predictive power
        """
    )

    cols = st.columns(len(results))
    for col, (name, res) in zip(cols, results.items()):
        with col:
            short_name = name.split("(")[0].strip()
            st.metric(short_name, f"{res.accuracy*100:.2f}%")

    for name, res in results.items():
        with st.expander(f"📊 {name} — Detailed Results", expanded=False):
            col1, col2 = st.columns([3, 2])
            with col1:
                st.text("Classification Report:")
                st.text(classification_report(y_test, res.predictions, target_names=["Down", "Up"]))
            with col2:
                if PLOTLY_AVAILABLE:
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(y_test, res.predictions)
                    fig_cm = go.Figure(data=go.Heatmap(z=cm, x=["Down", "Up"], y=["Down", "Up"], text=cm, texttemplate="%{text}", textfont=dict(size=16), colorscale="Blues", showscale=False))
                    fig_cm.update_layout(title="Confusion Matrix", height=250, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_cm, use_container_width=True)
                else:
                    st.write("Confusion Matrix:")
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(y_test, res.predictions)
                    st.dataframe(pd.DataFrame(cm, index=["Down", "Up"], columns=["Down", "Up"]))

    if "Random Forest" in results:
        st.subheader("🌟 Feature Importance Analysis (Random Forest)")
        rf = results["Random Forest"].model
        importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(20)
        
        if PLOTLY_AVAILABLE:
            fig = go.Figure(go.Bar(
                x=importance.values, y=importance.index, orientation="h",
                marker=dict(color=importance.values, colorscale="Viridis", showscale=True, colorbar=dict(title="Importance")),
                text=[f"{v:.3f}" for v in importance.values], textposition="outside",
            ))
            fig.update_layout(title="Top 20 Feature Importances (Random Forest)", height=500, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(importance.index, importance.values, color='skyblue')
            ax.set_title('Top 20 Feature Importances (Random Forest)')
            ax.set_xlabel('Importance Score')
            plt.tight_layout()
            st.pyplot(fig)


def render_comparison_page(arima_result: ArimaResult, results: Dict[str, ModelResult]) -> None:
    st.header("⚖️ Head-to-Head Comparison")
    
    st.markdown("### Comparing Statistical vs Machine Learning Approaches")

    rows = [{
        "Model": "ARIMA(5,1,2)",
        "Type": "Statistical / Probability",
        "Directional Accuracy": f"{arima_result.directional_accuracy*100:.2f}%",
        "Strengths": "Interpretable, confidence intervals, residual diagnostics",
        "Weaknesses": "Linear assumptions, struggles with non-linearity",
    }]

    trait_map = {
        "Logistic Regression": ("Statistical + ML", "Probabilistic output, interpretable coefficients", "Linear decision boundary"),
        "Random Forest": ("Machine Learning", "Non-linear interactions, feature importance", "Less interpretable, can overfit"),
        "Gradient Boosting": ("Machine Learning", "Strong predictive power", "Sensitive to hyperparameters"),
    }

    for name, res in results.items():
        model_type, strengths, weaknesses = trait_map.get(name, ("ML", "Good predictive power", "May require tuning"))
        rows.append({"Model": name, "Type": model_type, "Directional Accuracy": f"{res.accuracy*100:.2f}%", "Strengths": strengths, "Weaknesses": weaknesses})

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("📊 Accuracy Comparison")
    acc_values = [float(r["Directional Accuracy"].strip("%")) for r in rows]
    model_names = [r["Model"] for r in rows]
    
    if PLOTLY_AVAILABLE:
        fig = go.Figure(go.Bar(
            x=model_names, y=acc_values,
            marker_color=[ACCENT if "ARIMA" in name else "#f59e0b" if "Logistic" in name else ACCENT_2 for name in model_names],
            text=[f"{v:.1f}%" for v in acc_values], textposition="outside",
        ))
        fig.update_layout(title="Directional Accuracy by Model", yaxis=dict(title="Accuracy (%)", range=[0, max(acc_values) + 10]), height=450, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#22d3ee' if 'ARIMA' in name else '#f59e0b' if 'Logistic' in name else '#a78bfa' for name in model_names]
        bars = ax.bar(model_names, acc_values, color=colors)
        for bar, val in zip(bars, acc_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', ha='center', va='bottom')
        ax.set_title('Directional Accuracy by Model')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, max(acc_values) + 10)
        plt.tight_layout()
        st.pyplot(fig)


def render_theory_page() -> None:
    st.header("📚 Why Probability & Statistics Are Indispensable")
    
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(34,211,238,0.05), rgba(167,139,250,0.05)); 
                    border-radius: 16px; padding: 24px; border: 1px solid rgba(34,211,238,0.1);">
            <p style="font-size: 1.1rem; color: #cbd5e1;">
                <strong style="color: #22d3ee;">Machine learning</strong> supplies powerful function approximators; 
                <strong style="color: #a78bfa;">probability and statistics</strong> supply the language to 
                quantify uncertainty, test assumptions, and decide whether a model is actually trustworthy.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("---")
    
    sections = [
        {
            "title": "1. Stock Prices as Stochastic Processes",
            "content": r"""
            A stock price $S_t$ is a discrete-time stochastic process. The log return
            $r_t = \ln(S_t / S_{t-1})$ is commonly modelled as
            $r_t = \mu + \sigma_t \varepsilon_t$, with $\varepsilon_t \sim \mathcal{N}(0,1)$
            or a heavier-tailed distribution.
            
            **Why this matters:** The probabilistic nature of prices means we can never predict 
            with certainty — only with probability.
            """
        },
        {
            "title": "2. Stationarity (ADF Test)",
            "content": """
            Classical models assume weak stationarity: constant mean, constant variance, and
            autocovariance that depends only on lag. The Augmented Dickey-Fuller test checks the
            null hypothesis of a unit root (non-stationarity).
            
            **Why this matters:** Non-stationary series can produce spurious correlations and 
            unreliable forecasts.
            """
        },
        {
            "title": "3. Autocorrelation & White Noise",
            "content": """
            After fitting a model, the **Ljung-Box** test checks whether residuals are
            independently distributed. Rejecting that null means the model missed linear structure
            still present in the data.
            
            **Why this matters:** Validates whether your model has captured the signal.
            """
        },
        {
            "title": "4. Confidence Intervals",
            "content": """
            ARIMA forecasts come with prediction intervals derived from the estimated forecast-error
            variance — a genuine probabilistic claim ("95% probability the true value falls here"),
            which point-forecast ML models rarely provide natively.
            
            **Why this matters:** Confidence intervals quantify uncertainty in a way that 
            point forecasts cannot.
            """
        },
        {
            "title": "5. Volatility Modelling (GARCH)",
            "content": r"""
            $\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$
            models conditional variance directly — the foundation of modern risk management.
            
            **Why this matters:** Volatility clustering is a stylized fact of financial markets.
            """
        },
        {
            "title": "6. Evaluation as Statistical Inference",
            "content": """
            Even tree-based or deep models are ultimately judged with statistical tools: significance
            tests on accuracy differences, confusion-matrix-derived precision/recall, and
            time-series-aware cross-validation.
            
            **Why this matters:** Statistical rigor ensures that performance differences are real 
            and not due to chance.
            """
        },
    ]
    
    for section in sections:
        with st.expander(section["title"], expanded=False):
            st.markdown(section["content"])
    
    st.markdown("---")
    st.info(
        """
        **The Verdict:** Statistical and machine learning approaches are complementary, not competing.
        Statistical models provide interpretability, uncertainty quantification, and strong 
        theoretical foundations. Machine learning offers flexibility and pattern recognition. 
        The best results come from combining them.
        """
    )


# =====================================================================
# App entry point
# =====================================================================
def main() -> None:
    configure_page()
    inject_css()

    ticker, start_date, end_date, lookback, test_size = render_sidebar()

    with st.spinner(f"📥 Loading {ticker} data..."):
        df = load_price_data(ticker, start_date, end_date)

    render_nav()

    if df.empty:
        st.error(f"❌ No data found for **{ticker}**. Double-check the ticker symbol and date range.")
        st.stop()

    if len(df) < 100:
        st.warning("⚠️ Fewer than 100 trading days — models may be unreliable. Consider widening the date range.")

    df_feat = engineer_features(df, lookback)
    if df_feat.empty:
        st.error("❌ Not enough data remains after feature engineering.")
        st.stop()

    arima_result = fit_arima(df["Close"], test_size)
    ml_results, feature_cols, y_test, _ = train_ml_models(df_feat, test_size)

    page = st.session_state.get("page", "home")
    
    if page == "home":
        render_home_page(df, df_feat, ticker)
    elif page == "data":
        render_data_page(df, df_feat, ticker)
    elif page == "market3d":
        render_market3d_page(df, df_feat, ticker, arima_result)
    elif page == "stats":
        render_statistical_page(arima_result)
    elif page == "ml":
        render_ml_page(ml_results, feature_cols, y_test)
    elif page == "compare":
        render_comparison_page(arima_result, ml_results)
    elif page == "theory":
        render_theory_page()

    st.markdown("---")
    st.caption("📈 Educational Streamlit app · Data from Yahoo Finance · Not financial advice")


if __name__ == "__main__":
    main()
