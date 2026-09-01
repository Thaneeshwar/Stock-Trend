"""
Statistical vs Machine Learning Stock Trend Prediction
========================================================
An educational Streamlit application that compares classical statistical
time-series modelling (ARIMA) against machine learning classifiers
(Logistic Regression, Random Forest, Gradient Boosting) for predicting
next-day stock price direction.

Author : <Your Name>
Purpose: Portfolio project demonstrating data engineering, statistical
         modelling, machine learning, and interactive dashboard design.
Stack  : Python, Streamlit, yfinance, statsmodels, scikit-learn, Plotly

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


# =====================================================================
# Page configuration & styling
# =====================================================================
def configure_page() -> None:
    """Set Streamlit page config and inject light custom CSS for polish."""
    st.set_page_config(
        page_title="Statistical vs ML Stock Trend Prediction",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
            .main > div {padding-top: 1.5rem;}
            h1 {font-weight: 700;}
            div[data-testid="stMetricValue"] {font-size: 1.6rem;}
            .stTabs [data-baseweb="tab-list"] {gap: 6px;}
            .stTabs [data-baseweb="tab"] {
                padding: 10px 16px;
                border-radius: 8px 8px 0 0;
            }
            footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# Data layer
# =====================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_price_data(ticker: str, start, end) -> pd.DataFrame:
    """
    Download OHLCV data for a ticker from Yahoo Finance.

    Returns an empty DataFrame (rather than raising) on failure so the
    caller can show a friendly error message instead of a stack trace.
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
    """Container for everything the ARIMA tab needs to render and reuse."""
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

        # Directional accuracy: day-over-day sign agreement between the
        # actual test series and the forecast series. Both are aligned
        # to len(test), so the [1:] vs [:-1] diffs are always equal
        # length — this avoids the broadcast mismatch of comparing
        # forecast values against a differently-shifted actual series.
        actual_dir = (test.values[1:] > test.values[:-1]).astype(int)
        pred_dir = (forecast.values[1:] > forecast.values[:-1]).astype(int)
        n = min(len(actual_dir), len(pred_dir))
        acc = accuracy_score(actual_dir[:n], pred_dir[:n]) if n > 0 else 0.5

        return ArimaResult(
            train=train, test=test, forecast=forecast, conf_int=conf_int,
            residuals=fitted.resid, directional_accuracy=acc,
        )
    except Exception as exc:  # noqa: BLE001 — surfaced to the UI, not swallowed
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
        "Logistic Regression (Statistical baseline)": LogisticRegression(
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
# UI sections
# =====================================================================
def render_sidebar() -> Tuple[str, datetime, datetime, int, float]:
    st.sidebar.title("⚙️ Configuration")
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper().strip()
    start_date = st.sidebar.date_input("Start Date", value=datetime(2018, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime.now())
    lookback = st.sidebar.slider("Lookback Window (days) for ML features", 5, 60, 20)
    test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Why Probability & Statistics?**\n"
        "- Markets are stochastic → we model uncertainty with distributions.\n"
        "- Stationarity, autocorrelation, volatility clustering.\n"
        "- Confidence intervals & hypothesis testing.\n"
        "- Residual diagnostics (Ljung-Box).\n"
        "- Directional accuracy is a probabilistic concept."
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Built with Streamlit · yfinance · statsmodels · scikit-learn")
    return ticker, start_date, end_date, lookback, test_size


def render_header() -> None:
    st.title("📈 Statistical Models vs Machine Learning for Stock Trend Prediction")
    st.markdown(
        """
        ### Why do we use **Probability and Statistics** here?

        1. **Uncertainty is fundamental** – future prices are random variables; probability lets us quantify the odds of a move.
        2. **Stationarity & Differencing** – ARIMA requires a stationary series, tested with the Augmented Dickey-Fuller (ADF) test.
        3. **Volatility Clustering** – large moves cluster together (the intuition behind GARCH-type models).
        4. **Hypothesis Testing** – residual white-noise checks (Ljung-Box) validate whether a model captured the signal.
        5. **Risk Measures** – confidence intervals and value-at-risk both come from probability distributions.
        6. **Evaluation Metrics** – accuracy, precision, recall, F1 all have probabilistic interpretations.
        7. **Even ML models** are trained and judged with probabilistic loss functions (log-loss, cross-entropy).

        **Takeaway (2020–2026 literature):** no single model dominates. ARIMA/GARCH remain strong short-horizon
        baselines; tree-based ML often wins on non-linear signals but can overfit. Hybrid approaches frequently perform best.
        """
    )


def render_data_tab(df: pd.DataFrame, df_feat: pd.DataFrame, ticker: str) -> None:
    st.header("Data Overview & Stationarity")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Days", len(df))
    c2.metric("Latest Close", f"${df['Close'].iloc[-1]:.2f}")
    c3.metric("Mean Daily Return", f"{df_feat['Return'].mean()*100:.3f}%")
    c4.metric("Daily Volatility", f"{df_feat['Return'].std()*100:.3f}%")

    fig = go.Figure(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC"
        )
    )
    fig.update_layout(title=f"{ticker} Price", xaxis_rangeslider_visible=False, height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Augmented Dickey-Fuller Test (Stationarity)")
    adf_price = adfuller(df_feat["Close"])
    adf_return = adfuller(df_feat["Return"].dropna())

    c1, c2 = st.columns(2)
    with c1:
        st.write("**On Close Price**")
        st.write(f"ADF Statistic: {adf_price[0]:.4f}")
        st.write(f"p-value: {adf_price[1]:.4f}")
        st.write("✅ Stationary" if adf_price[1] < 0.05 else "❌ Non-stationary (needs differencing)")
    with c2:
        st.write("**On Returns**")
        st.write(f"ADF Statistic: {adf_return[0]:.4f}")
        st.write(f"p-value: {adf_return[1]:.4f}")
        st.write("✅ Stationary" if adf_return[1] < 0.05 else "❌ Non-stationary")

    st.info(
        "Statistical models like ARIMA require stationarity. Returns are usually stationary → "
        "we model returns or difference the price series."
    )


def render_arima_tab(result: ArimaResult) -> None:
    st.header("Statistical Probability Models – ARIMA")
    st.markdown(
        """
        **ARIMA (AutoRegressive Integrated Moving Average)** is a classic statistical time-series model.
        It relies on the assumption that residuals behave like white noise (i.i.d., approximately normal),
        and it produces confidence intervals — genuine probabilistic statements about future values.
        """
    )

    if result.error:
        st.error(f"ARIMA failed to fit: {result.error}")
        return

    st.success(f"ARIMA Directional Accuracy (approx): **{result.directional_accuracy*100:.2f}%**")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.train.index, y=result.train, name="Train", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=result.test.index, y=result.test, name="Actual Test", line=dict(color="green")))
    fig.add_trace(
        go.Scatter(x=result.test.index, y=result.forecast, name="ARIMA Forecast", line=dict(color="red", dash="dash"))
    )
    fig.add_trace(
        go.Scatter(
            x=result.test.index, y=result.conf_int.iloc[:, 0],
            mode="lines", line_color="rgba(255,0,0,0.2)", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.test.index, y=result.conf_int.iloc[:, 1],
            fill="tonexty", mode="lines", line_color="rgba(255,0,0,0.2)", name="95% Confidence Interval",
        )
    )
    fig.update_layout(title="ARIMA Forecast with 95% Probability Confidence Interval", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        **Why the confidence interval?** Under the model's probabilistic assumptions, there is a
        **95% probability** that the true future value lies inside the shaded band — a form of statistical
        inference that pure point-forecast ML models rarely provide out-of-the-box.
        """
    )

    st.write("**Ljung-Box Test on Residuals (should be white noise)**")
    lb_test = acorr_ljungbox(result.residuals, lags=[10], return_df=True)
    st.dataframe(lb_test, use_container_width=True)


def render_ml_tab(results: Dict[str, ModelResult], feature_cols: List[str], y_test: pd.Series) -> None:
    st.header("Machine Learning Models for Trend Classification")

    for name, res in results.items():
        st.subheader(name)
        st.metric("Directional Accuracy", f"{res.accuracy*100:.2f}%")
        st.text(classification_report(y_test, res.predictions, target_names=["Down", "Up"]))

    if "Random Forest" in results:
        rf = results["Random Forest"].model
        importance = (
            pd.Series(rf.feature_importances_, index=feature_cols)
            .sort_values(ascending=False)
            .head(15)
        )
        fig = go.Figure(go.Bar(x=importance.values, y=importance.index, orientation="h"))
        fig.update_layout(title="Top 15 Feature Importances (Random Forest)", height=450)
        st.plotly_chart(fig, use_container_width=True)


def render_comparison_tab(arima_result: ArimaResult, results: Dict[str, ModelResult]) -> None:
    st.header("⚖️ Head-to-Head Comparison")

    rows = [
        {
            "Model": "ARIMA(5,1,2)",
            "Type": "Statistical / Probability",
            "Directional Accuracy": f"{arima_result.directional_accuracy*100:.2f}%",
            "Strengths": "Interpretable, confidence intervals, residual diagnostics, works with small data",
            "Weaknesses": "Linear assumptions, struggles with strong non-linearity",
        }
    ]

    trait_map = {
        "Random Forest": (
            "Machine Learning",
            "Captures non-linear interactions, feature importance, robust",
            "Less interpretable, can overfit, no native uncertainty",
        ),
        "Gradient Boosting": (
            "Machine Learning",
            "Strong predictive power on tabular features",
            "Sensitive to hyperparameters, slower training",
        ),
    }

    for name, res in results.items():
        model_type, strengths, weaknesses = trait_map.get(
            name, ("Statistical + ML", "Probabilistic output (sigmoid), highly interpretable coefficients", "Linear decision boundary")
        )
        rows.append(
            {
                "Model": name,
                "Type": model_type,
                "Directional Accuracy": f"{res.accuracy*100:.2f}%",
                "Strengths": strengths,
                "Weaknesses": weaknesses,
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown(
        """
        ### Key Takeaways from Academic Literature (2020–2026)
        - **ARIMA / SARIMA / GARCH** remain competitive baselines, especially for short-horizon forecasts and liquid stocks.
        - **Tree-based models** (RF, XGBoost) often beat pure statistical models when rich technical features are available.
        - **LSTM / Transformers** can win on longer horizons or highly non-linear series, given enough data and careful validation.
        - **No universal winner** – performance is asset-, horizon-, and regime-dependent.
        - **Best practice**: pair statistical models (interpretability + uncertainty) with ML (non-linear pattern capture); hybrids often win.
        """
    )


def render_theory_tab() -> None:
    st.header("📚 Why Probability & Statistics Are Indispensable")
    st.markdown(
        r"""
        ### 1. Stock Prices as Stochastic Processes
        A stock price $S_t$ is a discrete-time stochastic process. The log return
        $r_t = \ln(S_t / S_{t-1})$ is commonly modelled as
        $r_t = \mu + \sigma_t \varepsilon_t$, with $\varepsilon_t \sim \mathcal{N}(0,1)$
        or a heavier-tailed distribution (Student-t, GARCH innovations).

        ### 2. Stationarity (ADF Test)
        Classical models assume weak stationarity: constant mean, constant variance, and
        autocovariance that depends only on lag. The Augmented Dickey-Fuller test checks the
        null hypothesis of a unit root (non-stationarity).

        ### 3. Autocorrelation & White Noise
        After fitting a model, the **Ljung-Box** test checks whether residuals are
        independently distributed. Rejecting that null means the model missed linear structure
        still present in the data.

        ### 4. Confidence Intervals
        ARIMA forecasts come with prediction intervals derived from the estimated forecast-error
        variance — a genuine probabilistic claim ("95% probability the true value falls here"),
        which point-forecast ML models rarely provide natively.

        ### 5. Volatility Modelling (GARCH)
        $\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$
        models conditional variance directly — the foundation of modern risk management.

        ### 6. Evaluation as Statistical Inference
        Even tree-based or deep models are ultimately judged with statistical tools: significance
        tests on accuracy differences (e.g. Diebold-Mariano), confusion-matrix-derived
        precision/recall, and time-series-aware cross-validation.

        **Bottom line:** machine learning supplies powerful function approximators; probability
        and statistics supply the language to quantify uncertainty, test assumptions, and decide
        whether a model is actually trustworthy.
        """
    )


# =====================================================================
# App entry point
# =====================================================================
def main() -> None:
    configure_page()
    ticker, start_date, end_date, lookback, test_size = render_sidebar()
    render_header()

    with st.spinner(f"Downloading {ticker} data..."):
        df = load_price_data(ticker, start_date, end_date)

    if df.empty:
        st.error(
            f"No data found for **{ticker}**. Double-check the ticker symbol and date range, "
            "then try again."
        )
        st.stop()

    if len(df) < 100:
        st.warning(
            "Fewer than 100 trading days in this window — ARIMA and the ML models may be "
            "unreliable. Consider widening the date range."
        )

    st.success(f"Loaded {len(df)} trading days for **{ticker}**")

    df_feat = engineer_features(df, lookback)
    if df_feat.empty:
        st.error("Not enough data remains after feature engineering. Try a longer date range or shorter lookback.")
        st.stop()

    tab_data, tab_arima, tab_ml, tab_compare, tab_theory = st.tabs(
        [
            "📊 Data & Exploration",
            "📐 Statistical Models (ARIMA + Probability)",
            "🤖 Machine Learning Models",
            "⚖️ Comparative Analysis",
            "📚 Theory & Why Probability Matters",
        ]
    )

    with tab_data:
        render_data_tab(df, df_feat, ticker)

    arima_result = fit_arima(df["Close"], test_size)
    with tab_arima:
        render_arima_tab(arima_result)

    ml_results, feature_cols, y_test, _ = train_ml_models(df_feat, test_size)
    with tab_ml:
        render_ml_tab(ml_results, feature_cols, y_test)

    with tab_compare:
        render_comparison_tab(arima_result, ml_results)

    with tab_theory:
        render_theory_tab()

    st.markdown("---")
    st.caption(
        "Educational Streamlit app · Data from Yahoo Finance · Models are for research/demo "
        "purposes only · Not financial advice"
    )


if __name__ == "__main__":
    main()

