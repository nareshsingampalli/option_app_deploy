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
    const hasInstruments = document.querySelectorAll('.instrument-cb').length > 0;
    
    // Force re-render if it's initial, or no instruments, or the symbol has changed
    if (isInitial || !hasInstruments || currentRenderedSymbol !== currentSymbol) {
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
                instrumentSelector.render(instrumentInfo);
                currentRenderedSymbol = currentSymbol;
            }
        } catch (e) {
            console.error("[App] Instrument list render error:", e);
        }
    }

    // 2. Render charts
    renderCharts();
});

// ── Wiring: Symbol Selection → Data Fetch ────────────────────────────────────
symbolSelector.onChange(({ exchange, symbol }) => {
    console.log(`[App] Market changed: ${exchange} - ${symbol}`);
    timeSelector.setExchange(exchange);
    
    // Clear old state immediately so user doesn't see stale data
    chartRenderer.clear ? chartRenderer.clear() : null; 
    dataService.clear(); 
    
    if (isLiveMode) {
        dataService.initWebSocket(exchange, symbol);
    }
    fetchData();
});

// ── Wiring: Instrument Selection → Local Re-render ───────────────────────────
instrumentSelector.onChange(selected => {
    chartRenderer.render(dataService.rawData, selected);
});

// ── Logic ────────────────────────────────────────────────────────────────────

window.buildParams = function () {
    const datePicker = document.getElementById('date-picker');
    return {
        exchange: symbolSelector.exchange,
        symbol: symbolSelector.symbol,
        date: datePicker.value,
        time: timeSelector.time,
        ...(isLiveMode ? { live: 'true' } : {})
    };
}

function fetchData(silent = false) {
    const datePicker = document.getElementById('date-picker');
    if (!datePicker.value) return;
    dataService.load(window.buildParams(), silent);
}

function renderCharts() {
    chartRenderer.render(dataService.rawData, instrumentSelector.selected());
}

// ── Controls ─────────────────────────────────────────────────────────────────
const datePicker = document.getElementById('date-picker');
const liveToggle = document.getElementById('live-toggle');

// Init dates
const todayStr = new Date().toLocaleDateString('en-CA');
const yesterdayObj = new Date();
yesterdayObj.setDate(yesterdayObj.getDate() - 1);
const yesterdayStr = yesterdayObj.toLocaleDateString('en-CA');

datePicker.max = todayStr;
datePicker.value = yesterdayStr;

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

document.getElementById('time-slider').addEventListener('change', () => {
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
        return (timeVal >= 915 && timeVal <= 1530);
    } else if (exchange === 'MCX') {
        return (timeVal >= 900 && timeVal <= 2330);
    }
    return false;
}

liveToggle.addEventListener('change', () => {
    isLiveMode = liveToggle.checked;
    const exchange = symbolSelector.exchange || 'NSE';

    if (isLiveMode && !isMarketOpen(exchange)) {
        const range = exchange === 'NSE' ? '09:15 AM - 03:30 PM' : '09:00 AM - 11:30 PM';
        showNotice(`Market is closed for ${exchange}. Live mode is only available Mon-Fri, ${range} IST.`);
        liveToggle.checked = false;
        isLiveMode = false;
        return;
    }

    if (isLiveMode) {
        const todayStr = new Date().toLocaleDateString('en-CA'); // YYYY-MM-DD
        datePicker.value = todayStr;
        datePicker.disabled = true;
        dataService.initWebSocket(symbolSelector.exchange, symbolSelector.symbol);
        fetchData();
    } else {
        datePicker.disabled = false;
        dataService.stopWebSocket();
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

// Global exposure for non-JS buttons (if any)
window.toggleOptionType = (type) => {
    const btn = document.getElementById(`toggle-${type.toLowerCase()}`);
    if (btn) btn.classList.toggle('active');
};

window.selectInstruments = (type) => {
    // Attempt to grab current spot price from display for Intraday/Scalping filters
    const spotEl = document.getElementById('spot-price-display');
    let spotPrice = null;
    if (spotEl && spotEl.textContent) {
        const match = spotEl.textContent.match(/[\d.]+/);
        if (match) spotPrice = parseFloat(match[0]);
    }
    instrumentSelector.selectAll(type, spotPrice);
};
window.refreshToken = refreshToken;

// ── Start ────────────────────────────────────────────────────────────────────
timeSelector.setExchange('NSE');
document.getElementById('loading').style.display = 'none';

if (isMarketOpen(symbolSelector.exchange)) {
    console.log("[App] Market is open (IST). Activating Live Mode automatically...");
    liveToggle.checked = true;
    liveToggle.dispatchEvent(new Event('change'));
} else {
    fetchData();
}
