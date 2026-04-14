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
const nextExpiryChk = document.getElementById('next-expiry-chk');

// ── Real-time Instrument List Observer (Subscriber Pattern) ──────────────────
dataService.onInstrumentsChanged((instruments) => {
    console.log("[App] Updating strike selector from resolved list...");
    instrumentSelector.render(instruments, currentReferenceSpotPrice);
});

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
dataService.subscribe((records, isInitial, status, errorCode) => {
    console.log(`[App] Data event. Records: ${records ? records.length : 0}, Status: ${status}, Error: ${errorCode}`);
    
    // 1. Handle Auth Error (UDAPI100050)
    const refreshBtn = document.querySelector('button[onclick="refreshToken()"]');
    if (status === "auth_error") {
        if (refreshBtn) refreshBtn.classList.add('btn-error-pulse');
        showNotice("Invalid API Token. Please click 'Refresh Token' to re-authenticate.");
    } else if (refreshBtn) {
        refreshBtn.classList.remove('btn-error-pulse');
    }

    // 2. Handle Holiday Fallback — auto-turn off Live mode if fallback occured or server reports holiday
    if (status === "holiday") {
        if (isLiveMode) {
            console.log("[App] Holiday detected — disabling Live Mode.");
            showNotice("Market is closed today (Holiday). Live mode disabled. Showing previous trading session.", 8000);
            isLiveMode = false;
            const liveToggle = document.getElementById('live-toggle');
            if (liveToggle) liveToggle.checked = false;
            const datePicker = document.getElementById('date-picker');
            if (datePicker) datePicker.disabled = false;
        }
    }

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
    
    // ── Aggressive Refresh: If the ATM strike shifts, re-center the whole list ─
    let isAtmShifted = false;
    if (strikes.length > 0 && spotPrice) {
        // Find the strike in our current list that is closest to the spot
        const currentAtm = strikes.reduce((prev, curr) => 
            Math.abs(curr - spotPrice) < Math.abs(prev - spotPrice) ? curr : prev
        );
        // Find the strike that is currently at the center of our list
        const centerIdx = Math.floor(strikes.length / 2);
        const centerStrike = strikes[centerIdx];
        
        // If the price has moved so much that a different strike is now the center-point
        // of our resolution, we trigger a re-fetch to get new OTM/ITMs around it.
        isAtmShifted = (currentAtm !== centerStrike);
    }

    const needsRefresh = isInitial || !hasInstruments || 
                         currentRenderedSymbol !== currentSymbol || 
                         currentRenderedDate !== currentDate ||
                         strikesShifted || isAtmShifted;

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
    // Logic removed: Let the user or server handle live mode state.
    
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
        next_expiry: document.getElementById('next-expiry-chk').checked ? 'true' : 'false',
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

nextExpiryChk.addEventListener('change', () => {
    dataService.clear();
    fetchData();
});



liveToggle.addEventListener('change', async () => {
    isLiveMode = liveToggle.checked;
    const exchange = symbolSelector.exchange || 'NSE';

    if (isLiveMode) {
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
    console.log(`[App] App initialized. Fetching initial data...`);
    dataService.initWebSocket(symbolSelector.exchange, symbolSelector.symbol);
    fetchData();
})();

// ── Auto-Open Background Timer ──────────────────────────────────────────────
setInterval(async () => {
    updateIntervalAvailability();
}, 60000);
