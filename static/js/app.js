/**
 * app.js — Orchestrator
 * Wires together all components and services.
 */

// ── Instantiate services & components ────────────────────────────────────────
const apiService = new ApiService();
const dataService = new DataService(apiService);
const symbolSelector = new SymbolSelector('symbol-selector-container');
const timeSelector = new TimeSelector('time-slider', 'time-display');
const instrumentSelector = new InstrumentSelector('instrument-list');
const metricSelector = new MetricSelector('metric-list');
const chartRenderer = new ChartRenderer('charts-area', metricSelector);

// ── State ────────────────────────────────────────────────────────────────────
let isLiveMode = false;
let refreshInterval = null;
let currentRenderedSymbol = null; 
let currentRenderedDate = null; 

// ── Notification Helper ─────────────────────────────────────────────────────
function showNotice(message, duration = 5000) {
    const loader = document.getElementById('loading');
    if (!loader) return;
    
    loader.textContent = message;
    loader.style.display = 'flex';
    loader.classList.add('waiting');
    
    // Clear after duration
    setTimeout(() => {
        if (loader.classList.contains('waiting') && loader.textContent === message) {
            loader.style.display = 'none';
            loader.classList.remove('waiting');
        }
    }, duration);
}

// ── Wiring: Data → UI ────────────────────────────────────────────────────────
dataService.subscribe((records, isInitial) => {
    console.log(`[App] Data received. Records: ${records ? records.length : 0}, Initial: ${isInitial}`);
    if (!records || records.length === 0) {
        console.warn("[App] No records received.");
        return;
    }
    // 1. Update instrument list
    const currentSymbol = symbolSelector.symbol;
    const currentDate = (records[0] && records[0].date) ? records[0].date.split(' ')[0] : '';
    const hasInstruments = document.querySelectorAll('.instrument-cb').length > 0;
    
    // Force re-render if it's initial, or no instruments, or the symbol/date has changed
    const needsRefresh = isInitial || !hasInstruments || 
                         currentRenderedSymbol !== currentSymbol || 
                         currentRenderedDate !== currentDate;

    if (needsRefresh) {
        try {
            const symbols = [...new Set(records.map(r => r.symbol))].sort();
            const instrumentInfo = symbols.map(sym => {
                const row = records.find(r => r.symbol === sym);
                let label = sym;
                let strike = null;
                let type = null;
                if (row && row.strike && row.option_type) {
                    const baseMatch = sym.match(/^[A-Z]+/);
                    const baseSym = baseMatch ? baseMatch[0] : '';
                    label = `${baseSym} ${row.strike} ${row.option_type}`;
                    strike = row.strike;
                    type = row.option_type;
                }
                return { symbol: sym, label: label, strike: strike, type: type };
            });
            if (instrumentInfo.length > 0) {
                // Read spot price so colors reflect ITM/ATM/OTM immediately
                const spotEl = document.getElementById('spot-price-display');
                let spotForRender = null;
                if (spotEl && spotEl.textContent) {
                    const m = spotEl.textContent.match(/[\d.]+/);
                    if (m) spotForRender = parseFloat(m[0]);
                }
                instrumentSelector.render(instrumentInfo, spotForRender);
                currentRenderedSymbol = currentSymbol;
                currentRenderedDate = currentDate;
            }
        } catch (e) {
            console.error("[App] Instrument list render error:", e);
        }
    }

    // 2. Apply active filters (e.g. Scalping default) and render charts
    if (window.applyCurrentFilters) window.applyCurrentFilters();
    renderCharts();
});

// ── Wiring: Symbol Selection → Data Fetch ────────────────────────────────────
symbolSelector.onChange(({ exchange, symbol }) => {
    console.log(`[App] Market changed: ${exchange} - ${symbol}`);
    timeSelector.setExchange(exchange);
    
    // Clear old state immediately so user doesn't see stale data
    chartRenderer.clear ? chartRenderer.clear() : null; 
    dataService.clear(); 
    
    // Always initialize WebSocket and join room to listen for background fetch updates (historical or live)
    dataService.initWebSocket(exchange, symbol);
    
    fetchData();
});

// ── Wiring: Instrument Selection → Local Re-render ───────────────────────────
instrumentSelector.onChange(selected => {
    chartRenderer.render(dataService.rawData, selected);
});

// ── Logic ────────────────────────────────────────────────────────────────────

window.buildParams = function () {
    const datePicker = document.getElementById('date-picker');
    const intervalSelect = document.getElementById('interval-select');
    return {
        exchange: symbolSelector.exchange,
        symbol: symbolSelector.symbol,
        date: datePicker.value,
        time: timeSelector.time,
        interval: intervalSelect.value,
        ...(isLiveMode ? { live: 'true' } : {})
    };
}

function fetchData(silent = false) {
    const datePicker = document.getElementById('date-picker');
    if (!datePicker.value) return;

    // Block fetching yesterday's or today's data between 12:00 AM and 1:00 AM IST
    const now = new Date();
    const ist = new Date(now.toLocaleString("en-US", { timeZone: "Asia/Kolkata" }));
    const hour = ist.getHours();
    
    const yesterday = new Date(ist);
    yesterday.setDate(yesterday.getDate() - 1);
    const yesterdayStr = yesterday.toLocaleDateString('en-CA');
    const todayStr = ist.toLocaleDateString('en-CA');

    if (hour === 0 && (datePicker.value === yesterdayStr || datePicker.value === todayStr)) {
        if (!silent) {
            showNotice("Historical data for the previous day is typically unavailable from the server between 12:00 AM and 1:00 AM IST.");
        }
        return;
    }

    if (!silent) {
        // Clear active UI state immediately to prevent showing yesterday/stale data
        chartRenderer.clear ? chartRenderer.clear() : null;
        instrumentSelector.clear ? instrumentSelector.clear() : null;
        dataService.clearData();
    }
    
    dataService.load(window.buildParams(), silent);
}

function renderCharts() {
    chartRenderer.render(dataService.rawData, instrumentSelector.selected());
}

// ── Controls ─────────────────────────────────────────────────────────────────
const datePicker = document.getElementById('date-picker');
const liveToggle = document.getElementById('live-toggle');

// Init dates
function getInitialDate() {
    const now = new Date();
    const ist = new Date(now.toLocaleString("en-US", { timeZone: "Asia/Kolkata" }));
    const day = ist.getDay(); // 0=Sun, 6=Sat
    
    if (day === 0) { // Sunday -> Last Friday
        ist.setDate(ist.getDate() - 2);
    } else if (day === 6) { // Saturday -> Friday
        ist.setDate(ist.getDate() - 1);
    }
    return ist.toLocaleDateString('en-CA');
}

const todayStr = new Date().toLocaleDateString('en-CA');
const initialDateStr = getInitialDate();

datePicker.max = todayStr;
datePicker.value = initialDateStr;

datePicker.addEventListener('change', () => {
    // Check for weekends (Note: Date picking in localized browsers can be tricky, using 'CA' or splitting ensures local day)
    const dateVal = datePicker.value;
    if (dateVal) {
        const d = new Date(dateVal);
        const day = d.getUTCDay(); // UTC because YYYY-MM-DD input is treated as UTC midnight
        if (day === 0 || day === 6) {
            showNotice(`The selected date (${dateVal}) is a trading holiday (Weekend). No data will be fetched.`);
        }
    }
    fetchData();
});

document.getElementById('time-slider').addEventListener('input', () => {
    timeSelector.updateDisplay();
});

document.getElementById('time-slider').addEventListener('change', fetchData);

document.getElementById('interval-select').addEventListener('change', () => {
    const exchange = symbolSelector.exchange || 'NSE';
    const interval = parseInt(document.getElementById('interval-select').value);
    timeSelector.reconfigure(exchange, interval);
    fetchData();
});

function isMarketOpen(exchange = 'NSE') {
    const now = new Date();
    const istTime = new Date(now.toLocaleString("en-US", { timeZone: "Asia/Kolkata" }));
    const day = istTime.getDay();
    const hour = istTime.getHours();
    const min = istTime.getMinutes();
    const timeVal = hour * 100 + min;

    // Weekend Check
    if (day === 0 || day === 6) return false;

    // Hours Check
    if (exchange === 'NSE') {
        return (timeVal >= 1000 && timeVal <= 1530);
    } else if (exchange === 'MCX') {
        return (timeVal >= 900 && timeVal <= 2330);
    }
    return false;
}

liveToggle.addEventListener('change', () => {
    isLiveMode = liveToggle.checked;
    const exchange = symbolSelector.exchange || 'NSE';

    if (isLiveMode && !isMarketOpen(exchange)) {
        const range = exchange === 'NSE' ? '10:00 AM - 03:30 PM' : '09:00 AM - 11:30 PM';
        showNotice(`Market is closed for ${exchange}. Live mode is only available Mon-Fri, ${range} IST.`);
        liveToggle.checked = false;
        isLiveMode = false;
        return;
    }

    if (isLiveMode) {
        const todayStr = new Date().toLocaleDateString('en-CA'); // YYYY-MM-DD
        datePicker.value = todayStr;
        datePicker.disabled = true;
        // WS is already initialized by SymbolSelector, but we ensure it's correct
        dataService.initWebSocket(symbolSelector.exchange, symbolSelector.symbol);
        fetchData();
    } else {
        datePicker.disabled = false;
        // We keep the socket open to hear about background historical fetches
    }
});

document.getElementById('update-charts-btn').addEventListener('click', fetchData);

async function refreshToken() {
    try {
        const result = await apiService.refreshToken();
        alert('Credentials refreshed successfully');
        fetchData();
    } catch (err) {
        alert('Failed to refresh: ' + err.message);
    }
}

// ── Instrument Selection State Machine ──────────────────────────────────────
window.handleSelectorClick = (type) => {
    const btnId = `btn-${type}`;
    const btn = document.getElementById(btnId);
    if (!btn) return;

    const allButtons = document.querySelectorAll('#selector-toggles .btn');
    
    // Logic: 
    // - 'all' and 'none' (Clear) are mutually exclusive to others.
    // - 'ce', 'pe', 'intraday', 'scalping' are toggles.

    if (type === 'all' || type === 'none') {
        allButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    } else {
        // Deactivate 'all' and 'none' if any other toggle is clicked
        document.getElementById('btn-all').classList.remove('active');
        document.getElementById('btn-none').classList.remove('active');
        
        // Mutual exclusivity between intraday and scalping
        if (type === 'intraday') document.getElementById('btn-scalping').classList.remove('active');
        if (type === 'scalping') document.getElementById('btn-intraday').classList.remove('active');

        btn.classList.toggle('active');
    }

    window.applyCurrentFilters();
};

window.applyCurrentFilters = () => {
    // Extract current state
    const states = {
        all: document.getElementById('btn-all').classList.contains('active'),
        none: document.getElementById('btn-none').classList.contains('active'),
        ce: document.getElementById('btn-ce').classList.contains('active'),
        pe: document.getElementById('btn-pe').classList.contains('active'),
        intraday: document.getElementById('btn-intraday').classList.contains('active'),
        scalping: document.getElementById('btn-scalping').classList.contains('active')
    };

    // Grab spot price
    const spotEl = document.getElementById('spot-price-display');
    let spotPrice = null;
    if (spotEl && spotEl.textContent) {
        const match = spotEl.textContent.match(/[\d.]+/);
        if (match) spotPrice = parseFloat(match[0]);
    }

    instrumentSelector.applySelection(states, spotPrice);
};

// Cleanup old global
delete window.selectInstruments;
delete window.toggleOptionType;

window.refreshToken = refreshToken;

// ── Start ────────────────────────────────────────────────────────────────────
timeSelector.setExchange('NSE');
document.getElementById('loading').style.display = 'none';

if (isMarketOpen(symbolSelector.exchange)) {
    console.log("[App] Market is open (IST). Activating Live Mode automatically...");
    liveToggle.checked = true;
    liveToggle.dispatchEvent(new Event('change'));
} else {
    // Join room anyway to listen for updates to default symbol
    dataService.initWebSocket(symbolSelector.exchange, symbolSelector.symbol);
    fetchData();
}
