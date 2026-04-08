# 🚀 Option Chain Dashboard

A high-performance, real-time option chain analytics platform built with Flask and SocketIO. It specializes in processing intraday and historical option data from **NSE**, **BSE**, and **MCX** using the Upstox API.

---

## ✨ Key Features

- **📊 Real-time Monitoring**: Live streaming of Option Chain metrics (Open Interest, Volume, Price).
- **⏱️ Time-Travel Analysis**: Interactive historical data exploration via an intraday playback slider.
- **🌈 Smart Visualization**:
    - **Color-Coded Moneyness**: ITM (In-the-Money), ATM (At-the-Money), and OTM (Out-of-the-Money) highlighting for Calls (CE) and Puts (PE).
    - **Dynamic OI Metrics**: Real-time Rate of Change (ROC) calculations anchored to the previous trading day's close for accurate trend identification.
- **⚡ Advanced Filtering**:
    - **Scalping Mode**: Intelligent ATM-centric selection for high-frequency analysis.
    - **Intraday Mode**: Broad view of the chain relative to the current spot price.
- **🌍 Multi-Exchange Support**: Advanced processing for NSE (Nifty/BankNifty), BSE (Sensex), and MCX (Commodities like Crude Oil/Natural Gas).
- **🛡️ Robust Pipeline**:
    - **Holiday-Aware Scheduler**: Automatically manages data fetching cycles based on market hours and calendars.
    - **Intelligent Rate Limiting**: Optimized API interaction to prevent 429 errors.
    - **Multi-Layer Caching**: Persistent file and memory storage for fast UI updates.

---

## 🖥️ User Interface Guide

### Interactive Sidebar
- **Exchange & Symbol**: Switch between NSE, BSE, and MCX instruments.
- **Interval Selector**: Choose data granularity from 1 to 300 minutes.
- **Filter Toggles**:
    - `All` / `None`: Mass selection/deselection.
    - `CE` / `PE`: Toggle Calls and Puts visibility.
    - `Scalping`: Automatically selects the 3 nearest ATM strikes (ideal for quick entries).
    - `Intraday`: Selects a wider range of strikes around the spot.

### Main View
- **Time Slider**: Drag to analyze historical snapshots throughout the trading day.
- **Live Toggle**: Enable real-time updates during market hours (09:15 - 15:30 IST for NSE/BSE).
- **Charts**: View interactive Plotly charts with synchronized crosshairs and metric comparisons.

---

## 🚀 Technical Stack

- **Backend**: Python, Flask, Flask-SocketIO.
- **Frontend**: Vanilla JS (ES6+), Plotly.js, CSS3 (Modern Glassmorphism Design).
- **Data Engine**: Pandas, NumPy, Scipy (Greeks calculation), Upstox SDK.
- **Task Runner**: Custom background scheduler with thread-safe locking.

---

## 📂 Project Architecture

The application follows a modular architecture for scalability and maintainability:

- **`run.py`**: The central entry point. Boots the Flask server and background schedulers.
- **`dashboard/`**: Contains API routes and the `SocketIO` real-time broadcasting logic.
- **`fetchers/`**: Specialized Upstox API wrappers (`Intraday`, `Historical`, `Expired`) with built-in retry logic.
- **`resolvers/`**: Instrument and symbol resolution for different exchanges.
- **`strategies/`**: Data normalization, Black-Scholes Greeks calculation, and exchange-specific processing logic.
- **`storage/`**:抽象化的 data persistence layer (supports CSV, JSON, and DB outputs).
- **`core/`**: Configuration management (`config.py`), utilities, and the rate-limiter.
- **`static/`**: Modern frontend with Plotly.js and interactive JS components.

---

## 🛠️ Quick Start

### 1. Prerequisites
- Python 3.8+
- [Upstox API](https://upstox.com/developer/api-documentation/) credentials.

### 2. Installation
```powershell
# Clone and install dependencies
git clone <repo-url>
cd option_app
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file or set the following variables:
- `UPSTOX_ACCESS_TOKEN`: Your valid API access token.
- `SENTRY_DSN` (Optional): For error tracking.

### 4. Launch the Application
```powershell
python run.py
```
Open **[http://localhost:8010](http://localhost:8010)** to view the dashboard.

---

## 📘 Troubleshooting

- **Token Expiry**: If the app fails with authentication errors, ensure your `UPSTOX_ACCESS_TOKEN` is fresh. You can use the `/api/refresh-token` endpoint (if configured) or restart with a new token.
- **Market Hours**: The background scheduler only triggers during market hours (09:15-15:30 for NSE/BSE). For testing outside these hours, use historical data mode via the UI.
- **Rate Limits**: If you encounter 429 errors, check `core/rate_limiter.py` to adjust throttle settings.

---
*Updated April 2026 for the latest architecture and UI enhancements.*
