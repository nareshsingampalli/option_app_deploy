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
    
    // ── Sync Instrument List — Automatic Side-Bar Update ───────────────────
    const firstRow = records[0];
    const spotPrice = parseFloat(firstRow.spot_price) || null;
    const currentSymbol = symbolSelector.symbol;
    const currentDate = (firstRow && firstRow.date) ? firstRow.date.split(' ')[0] : '';
    
    // Extract candidate instrument info from the data
    const symbols = [...new Set(records.map(r => r.symbol))].sort();
    const currentInstrumentInfo = symbols.map(sym => {
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

    const hasInstruments = document.querySelectorAll('.instrument-cb').length > 0;
    
    // Force re-render if it's initial, or no instruments, OR the set of strikes has changed
    const existingSymbols = (instrumentSelector._lastInstrumentInfo || []).map(x => x.symbol);
    const incomingSymbols = currentInstrumentInfo.map(x => x.symbol);
    const strikesShifted = existingSymbols.length !== incomingSymbols.length || 
                          !incomingSymbols.every(s => existingSymbols.includes(s));

    const needsRefresh = isInitial || !hasInstruments || 
                         currentRenderedSymbol !== currentSymbol || 
                         currentRenderedDate !== currentDate ||
                         strikesShifted;

    if (needsRefresh) {
        try {
            if (currentInstrumentInfo.length > 0) {
                console.log(`[App] Instrument list shifted or new load. Syncing sidebar...`);
                instrumentSelector.render(currentInstrumentInfo, spotPrice);
                currentRenderedSymbol = currentSymbol;
                currentRenderedDate = currentDate;
                
                // Re-apply filters (like Scalping) if any was active
                const allButtons = document.querySelectorAll('#selector-toggles .btn');
                const anyActive = Array.from(allButtons).some(b => b.classList.contains('active') && b.id !== 'btn-all');
                if (anyActive && window.applyCurrentFilters) {
                    window.applyCurrentFilters();
                }
            }
        } catch (e) {
            console.error("[App] Instrument list sync error:", e);
        }
    } else {
        // Just re-calculate ATM/ITM labels/colors in place without re-rendering the whole sidebar
        instrumentSelector.recolor(spotPrice);
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
    
    // Always initialize WebSocket and join room to listen for background fetch updates (historical or live)
    dataService.initWebSocket(exchange, symbol);
    
    updateIntervalAvailability();
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
    updateIntervalAvailability();
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

// ── Market Status (single source of truth: fetched from backend) ─────────────
const _marketStatusCache = {};  // { exchange: { ts, is_open, start, end, reason } }
const MARKET_STATUS_TTL_MS = 60_000; // re-check at most once per minute

async function fetchMarketStatus(exchange = 'NSE') {
    const cached = _marketStatusCache[exchange];
    if (cached && (Date.now() - cached.ts) < MARKET_STATUS_TTL_MS) {
        return cached;
    }
    try {
        const res = await fetch(`/api/market-status?exchange=${exchange}`);
        const data = await res.json();
        _marketStatusCache[exchange] = { ...data, ts: Date.now() };
        return _marketStatusCache[exchange];
    } catch (e) {
        console.warn('[App] Could not reach /api/market-status, falling back to cache:', e);
        return cached || { is_open: false, reason: 'unknown', start: '09:15', end: '15:30' };
    }
}

liveToggle.addEventListener('change', async () => {
    isLiveMode = liveToggle.checked;
    const exchange = symbolSelector.exchange || 'NSE';

    if (isLiveMode) {
        const status = await fetchMarketStatus(exchange);
        if (!status.is_open) {
            showNotice(
                `Market is closed for ${exchange}. ` +
                `Live mode is available Mon-Fri, ${status.start}–${status.end} IST. ` +
                `(Server: ${status.now_ist} IST, reason: ${status.reason})`
            );
            liveToggle.checked = false;
            isLiveMode = false;
            return;
        }
        const todayStr = new Date().toLocaleDateString('en-CA'); // YYYY-MM-DD
        datePicker.value = todayStr;
        datePicker.disabled = true;
        dataService.initWebSocket(symbolSelector.exchange, symbolSelector.symbol);
        fetchData();
    } else {
        datePicker.disabled = false;
        // Keep socket open to hear about background historical fetches
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

// ── Interval Availability Logic ───────────────────────────────────────────
function updateIntervalAvailability() {
    const exchange = symbolSelector.exchange || 'NSE';
    const intervalSelect = document.getElementById('interval-select');
    if (!intervalSelect) return;

    const now = new Date();
    const ist = new Date(now.toLocaleString("en-US", { timeZone: "Asia/Kolkata" }));
    
    // Market open in minutes from midnight
    const marketOpenMinutes = (exchange === 'NSE') ? (9 * 60 + 15) : (9 * 60);
    const currentMinutes = ist.getHours() * 60 + ist.getMinutes();
    const elapsedMinutes = currentMinutes - marketOpenMinutes;

    const datePicker = document.getElementById('date-picker');
    const isToday = datePicker.value === ist.toLocaleDateString('en-CA');

    let firstAvailable = null;
    Array.from(intervalSelect.options).forEach(opt => {
        const intervalVal = parseInt(opt.value);
        
        // If today, only enable if at least one candle has completed
        if (isToday && elapsedMinutes >= 0 && elapsedMinutes < 720) { // Keep logic for morning/day
            if (elapsedMinutes < intervalVal) {
                opt.disabled = true;
                opt.style.backgroundColor = '#f8f8f8';
                opt.style.color = '#bbb';
            } else {
                opt.disabled = false;
                opt.style.backgroundColor = '';
                opt.style.color = '';
                if (!firstAvailable) firstAvailable = opt.value;
            }
        } else {
            opt.disabled = false;
            opt.style.backgroundColor = '';
            opt.style.color = '';
        }
    });

    // Auto-switch if user was on a 15m but it's now disabled (e.g. at 09:20 AM)
    if (intervalSelect.selectedOptions[0] && intervalSelect.selectedOptions[0].disabled) {
        intervalSelect.value = firstAvailable || "1";
        timeSelector.reconfigure(exchange, parseInt(intervalSelect.value));
    }
}

// ── Start ────────────────────────────────────────────────────────────────────
timeSelector.setExchange('NSE');
document.getElementById('loading').style.display = 'none';

(async () => {
    updateIntervalAvailability();
    const status = await fetchMarketStatus(symbolSelector.exchange);
    if (status.is_open) {
        console.log(`[App] Market is open (${status.now_ist} IST). Activating Live Mode automatically...`);
        liveToggle.checked = true;
        liveToggle.dispatchEvent(new Event('change'));
    } else {
        console.log(`[App] Market closed (${status.reason}). Loading historical data...`);
        dataService.initWebSocket(symbolSelector.exchange, symbolSelector.symbol);
        fetchData();
    }
})();

// ── Auto-Open Background Timer ──────────────────────────────────────────────
setInterval(async () => {
    updateIntervalAvailability();
    if (!isLiveMode) {
        const exchange = symbolSelector.exchange || 'NSE';
        const status = await fetchMarketStatus(exchange);
        if (status.is_open) {
            console.log(`[App] Market has opened. Switching to Live Mode...`);
            liveToggle.checked = true;
            liveToggle.dispatchEvent(new Event('change'));
        }
    }
}, 60000);
