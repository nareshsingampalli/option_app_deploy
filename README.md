# Option App Deploy

Nifty Option Analysis Dashboard.

## Features
- Fetches real-time and historical option chain data from Upstox API.
- Displays data in a tabular format.
- Handles access token management.
- Deployed on Ubuntu VM.

## Setup
1. Clone the repository.
2. Install dependencies: \pip install -r requirements.txt\
3. Set up environment variables in \.env\.
4. Run the dashboard: \python3 option_dashboard.py\

## Structure
- \option_dashboard.py\: Main Flask application.
- \option_chain.py\: Script to fetch option chain data.
- \candle_fetchers.py\: Helper for fetching historical candle data.
- \	emplates/\: HTML templates for the dashboard.
