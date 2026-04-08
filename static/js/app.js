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

// Expose for cross-module access (e.g. DataService date-shift handling)
window._timeSelector = timeSelector;
window._symbolSelector = symbolSelector;

// ── State ────────────────────────────────────────────────────────────────────
let isLiveMode = false;
let refreshInterval = null;
let currentRenderedSymbol = null; 
let currentRenderedDate = null; 
let currentReferenceSpotPrice = null;

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
    const targetTime = timeSelector.time;
    // Find the specific record for our slider time to get the correct spot price
    const matchingRow = records.find(r => r.date.includes(targetTime)) || records[records.length - 1];
    const spotPrice = parseFloat(matchingRow.spot_price) || null;
    currentReferenceSpotPrice = spotPrice; // Store for selection changes
    
    const currentSymbol = symbolSelector.symbol;
    const currentDate = (matchingRow && matchingRow.date) ? matchingRow.date.split(' ')[0] : '';
    
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
    const strikesShifted = existingSymbols.length !== symbols.length || 
                          !symbols.every(s => existingSymbols.includes(s));
    
    const strikes = currentInstrumentInfo.map(x => parseFloat(x.strike)).filter(s => !isNaN(s));
    
    // Proactive refresh: If spot is within the outer 15% of our strike range, or completely outside
    let isSpotOutsideRange = false;
    if (strikes.length > 0 && spotPrice) {
        const minS = Math.min(...strikes);
        const maxS = Math.max(...strikes);
        const range = maxS - minS;
        const buffer = range * 0.15; // 15% margin
        isSpotOutsideRange = (spotPrice < (minS + buffer)) || (spotPrice > (maxS - buffer));
    }

    const needsRefresh = isInitial || !hasInstruments || 
                         currentRenderedSymbol !== currentSymbol || 
                         currentRenderedDate !== currentDate ||
                         strikesShifted || isSpotOutsideRange;

    if (needsRefresh) {
        try {
            if (currentInstrumentInfo.length > 0) {
                console.log(`[App] Instrument list shifted or new load (Spot: ${spotPrice}). Syncing sidebar...`);
                instrumentSelector.render(currentInstrumentInfo, spotPrice);
                
                // Update Sidebar Labels: Spot Price & Date
                const spotEl = document.getElementById('spot-price-display');
                if (spotEl) {
                    spotEl.innerHTML = `<strong>Spot Price:</strong> ${spotPrice.toFixed(2)}`;
                    spotEl.style.color = '#1e88e5'; // Highlight when syncing
                }
                
                currentRenderedSymbol = currentSymbol;
                currentRenderedDate = currentDate;
                
                // After full render, always apply current filters (Scalping, etc.)
                if (window.applyCurrentFilters) {
                    window.applyCurrentFilters(false);
                }
            }
        } catch (e) {
            console.error("[App] Instrument list sync error:", e);
        }
    } else {
        // Just re-calculate ATM/ITM labels/colors in place without re-rendering the whole sidebar
        instrumentSelector.recolor(spotPrice);
        
        // Even if list didn't change, the SELECTION might need to move (e.g. Scalping ATM changed)
        if (window.applyCurrentFilters) {
            window.applyCurrentFilters(false);
        }
    }

    // 2. Render charts
    renderCharts();
});

// ── Wiring: Symbol Selection → Data Fetch ────────────────────────────────────
symbolSelector.onChange(async ({ exchange, symbol }) => {
    console.log(`[App] Market changed: ${exchange} - ${symbol}`);
    timeSelector.setExchange(exchange);
    
    // Clear old state immediately so user doesn't see stale data
    chartRenderer.clear ? chartRenderer.clear() : null; 
    dataService.clear(); 
    
    // ── Handle Auto-Live Toggle on Exchange Switch ───────────────────────
    const status = await fetchMarketStatus(exchange);
    if (!status.is_open && isLiveMode) {
        console.log(`[App] ${exchange} is closed. Switching off Live Mode.`);
        isLiveMode = false;
        liveToggle.checked = false;
        datePicker.disabled = false;
    } else if (status.is_open && !isLiveMode) {
        console.log(`[App] ${exchange} is open. Activating Live Mode.`);
        isLiveMode = true;
        liveToggle.checked = true;
        datePicker.disabled = true;
        const todayStr = new Date().toLocaleDateString('en-CA');
        datePicker.value = todayStr;
    }

    // Always initialize WebSocket and join room to listen for background fetch updates (historical or live)
    dataService.initWebSocket(exchange, symbol);
    
    updateIntervalAvailability();
    fetchData();
});

// ── Wiring: Instrument Selection → Local Re-render ───────────────────────────
instrumentSelector.onChange(selected => {
    chartRenderer.render(dataService.rawData, selected, currentReferenceSpotPrice);
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
        // Show pulse states in header and sidebars
        const spotEl = document.getElementById('spot-price-display');
        if (spotEl) {
            spotEl.innerHTML = `<strong>Spot Price:</strong> <span class="pulse-text" style="width: 80px; height: 18px; vertical-align: middle;"></span>`;
        }
        
        const expiryEl = document.getElementById('expiry-display');
        if (expiryEl) {
            expiryEl.innerHTML = `<strong>Expiry:</strong> <span class="pulse-text" style="width: 100px; height: 18px; vertical-align: middle;"></span>`;
        }

        const lastUpdatedEl = document.getElementById('last-updated-display');
        if (lastUpdatedEl) {
            lastUpdatedEl.innerHTML = `<span class="pulse-text" style="width: 180px; height: 14px;"></span>`;
        }

        chartRenderer.clear ? chartRenderer.clear() : null;
        
        // Show skeleton list in sidebar
        if (instrumentSelector.container) {
            instrumentSelector.container.innerHTML = Array(12).fill(0).map(() => 
                `<div class="skeleton-sidebar"></div>`
            ).join('');
        }
        
        dataService.clearData();
    }
    
    dataService.load(window.buildParams(), silent);
}

function renderCharts() {
    chartRenderer.render(dataService.rawData, instrumentSelector.selected(), currentReferenceSpotPrice);
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

// ── Time Slider Debounce Logic ──────────────────────────────────────────────
let sliderDebounceTimer = null;
document.getElementById('time-slider').addEventListener('input', () => {
    timeSelector.updateDisplay();
    
    // Clear existing timer
    if (sliderDebounceTimer) clearTimeout(sliderDebounceTimer);
    
    // Wait for 1 second of stillness before fetching
    sliderDebounceTimer = setTimeout(() => {
        console.log(`[App] Slider stopped at ${timeSelector.time}. Fetching time-slice...`);
        fetchData();
    }, 1000);
});

// Remove the old 'change' listener since we now use debounced 'input'
// document.getElementById('time-slider').addEventListener('change', fetchData);

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
        
        // Auto-reset slider to the end when turning Live Mode back ON
        const slider = document.getElementById('time-slider');
        if (slider) {
            slider.value = slider.max;
            timeSelector.updateDisplay();
        }

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
        // Deactivate 'all' and 'none' master switches if any specific filter is toggled
        document.getElementById('btn-all').classList.remove('active');
        document.getElementById('btn-none').classList.remove('active');
        
        // Exclusivity: CE vs PE
        if (type === 'ce') document.getElementById('btn-pe').classList.remove('active');
        if (type === 'pe') document.getElementById('btn-ce').classList.remove('active');

        // Exclusivity: Intraday vs Scalping
        if (type === 'intraday') document.getElementById('btn-scalping').classList.remove('active');
        if (type === 'scalping') document.getElementById('btn-intraday').classList.remove('active');

        btn.classList.toggle('active');
    }

    window.applyCurrentFilters();
};

window.applyCurrentFilters = (notify = true) => {
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

    instrumentSelector.applySelection(states, spotPrice, notify);
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
