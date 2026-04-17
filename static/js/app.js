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
let isLiveMode      = false;
let refreshInterval = null;
let currentRenderedSymbol    = null;
let currentRenderedDate      = null;
let currentReferenceSpotPrice = null;
let _liveToggleBlocked = false; // true when market is closed/holiday — live toggle must stay off

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

    // 2. Handle Holiday / No-data — auto-fallback (NSE/BSE only)
    if (status === "holiday") {
        const currentExchange = symbolSelector.exchange || 'NSE';
        if (currentExchange === 'MCX') {
            console.log("[App] MCX data missing - rollback disabled.");
            showNotice("No MCX data available for this session.", 6000);
            if (isLiveMode) {
                isLiveMode = false;
                const liveToggle = document.getElementById('live-toggle');
                if (liveToggle) liveToggle.checked = false;
                const datePicker = document.getElementById('date-picker');
                if (datePicker) datePicker.disabled = false;
            }
            return;
        }

        // Only turn off live toggle if it was on
        if (isLiveMode) {
            console.log("[App] Holiday detected — disabling Live Mode.");
            isLiveMode = false;
            const liveToggle = document.getElementById('live-toggle');
            if (liveToggle) liveToggle.checked = false;
        }

        const datePicker = document.getElementById('date-picker');
        if (datePicker) {
            datePicker.disabled = false;

            // Prefer server-provided fallback_date (already weekend-safe + pre-fetched)
            // errorCode holds fallback_date passed from DataService._notify
            if (errorCode) {
                console.log(`[App] Using server fallback date: ${errorCode}`);
                datePicker.value = errorCode;
                showNotice(`Holiday / no data. Showing ${errorCode}.`, 6000);
            } else {
                // Client-side fallback: roll back 1 calendar day
                const current = new Date(datePicker.value);
                current.setDate(current.getDate() - 1);
                datePicker.value = current.toLocaleDateString('en-CA');
                showNotice("Market is closed today (Holiday). Showing previous trading session.", 8000);
            }
            fetchData();
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

    // ── Track the actively watched symbol ─────────────────────────────────
    // This ensures WebSocket data_updated events only fire for the new symbol.
    dataService.setActiveSymbol(symbol);

    // Clear old state immediately so user doesn't see stale data
    chartRenderer.clear ? chartRenderer.clear() : null;
    dataService.clear();

    // Re-join the WS room for the new symbol (leaves old room automatically on server)
    dataService.initWebSocket(symbol);

    // If live mode was on for the previous symbol, cancel it —
    // the user must re-enable live manually for the new symbol.
    if (isLiveMode) {
        console.log(`[App] Symbol switched while live — pausing live mode for new symbol.`);
        isLiveMode = false;
        const liveToggle = document.getElementById('live-toggle');
        if (liveToggle) liveToggle.checked = false;
        const datePicker = document.getElementById('date-picker');
        if (datePicker) datePicker.disabled = false;
    }

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

async function fetchData(silent = false) {
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
        // Show pulse states only for initial/forced loads
        const spotEl = document.getElementById('spot-price-display');
        if (spotEl) {
            spotEl.innerHTML = `<strong>Spot Price:</strong> <span class="pulse-text" style="width: 80px; height: 18px; vertical-align: middle; display: inline-block;"></span>`;
        }
        
        const expiryEl = document.getElementById('expiry-display');
        if (expiryEl) {
            expiryEl.innerHTML = `<strong>Expiry:</strong> <span class="pulse-text" style="width: 100px; height: 18px; vertical-align: middle; display: inline-block;"></span>`;
        }

        const lastUpdatedEl = document.getElementById('last-updated-display');
        if (lastUpdatedEl) {
            lastUpdatedEl.innerHTML = `<span class="pulse-text" style="width: 180px; height: 14px; display: inline-block;"></span>`;
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
    
    await dataService.load(window.buildParams(), silent);
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
    const now = new Date();
    const istToday = new Date(now.toLocaleString("en-US", { timeZone: "Asia/Kolkata" })).toLocaleDateString('en-CA');
    const selectedDate = datePicker.value;

    // ── Expired / Historical date selected ────────────────────────────────
    // If the user picks any date that is not today, turn off live mode and
    // let the backend route automatically:
    //   - ExpiredCandleFetcher  (NSE: date ≤ last expired expiry)
    //   - HistoricalCandleFetcher (everything else past)
    if (selectedDate !== istToday && isLiveMode) {
        console.log(`[App] Date changed to ${selectedDate} — disabling live mode.`);
        isLiveMode = false;
        liveToggle.checked  = false;
        datePicker.disabled = false;
        // Notify the user when the date looks like an expired contract window
        showNotice(`Historical mode: fetching data for ${selectedDate}`, 4000);
        // stopLiveMode leaves the WS room cleanly
        dataService.stopLiveMode(symbolSelector.symbol);
    }

    // Default slider to end-of-day for any non-today date
    if (selectedDate !== istToday) {
        const slider = document.getElementById('time-slider');
        if (slider) {
            slider.value = slider.max;
            if (window._timeSelector) window._timeSelector.updateDisplay();
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
    
    // Wait for 200ms of stillness before fetching (MUCH snappier)
    sliderDebounceTimer = setTimeout(() => {
        fetchData(true); // silent = true to prevent clearing/skeletons during slide
    }, 200);
});

// Remove the old 'change' listener since we now use debounced 'input'
// document.getElementById('time-slider').addEventListener('change', fetchData);

document.getElementById('interval-select').addEventListener('change', (e) => {
    e.target.dataset.userHardSet = "true";
    const exchange = symbolSelector.exchange || 'NSE';
    const symbol   = symbolSelector.symbol || 'NIFTY';
    const interval = parseInt(document.getElementById('interval-select').value);
    
    timeSelector.reconfigure(exchange, interval);

    // If live mode is enabled, we must re-join the symbol room so the 
    // server-side scheduler knows we've switched our interval interest.
    if (isLiveMode) {
        console.log(`[App] Interval changed while Live — notifying scheduler of ${interval}m interest.`);
        dataService.initWebSocket(symbol);
    }
    
    fetchData();
});

nextExpiryChk.addEventListener('change', () => {
    const symbol = symbolSelector.symbol || 'NIFTY';
    
    // Notify scheduler of new expiry track interest if alive
    if (isLiveMode) {
        console.log(`[App] Expiry track changed while Live — notifying scheduler.`);
        dataService.initWebSocket(symbol);
    }

    dataService.clear();
    fetchData();
});



liveToggle.addEventListener('change', async () => {
    const exchange = symbolSelector.exchange || 'NSE';
    const symbol   = symbolSelector.symbol   || 'NIFTY';

    if (liveToggle.checked) {
        // ── LIVE ON: probe spot price first to detect holidays ─────────────
        console.log(`[App] Live toggle ON — probing spot for ${symbol} (${exchange})…`);
        showNotice(`Checking market status for ${symbol}…`, 4000);

        let probe = { is_holiday: false, spot_price: null, reason: '' };
        try {
            probe = await apiService.spotProbe(exchange, symbol);
        } catch (e) {
            console.warn('[App] Spot probe failed, proceeding cautiously.', e);
        }

        if (probe.reason === 'market_closed') {
            // Market hasn't opened yet — refuse live mode, use historical
            console.log('[App] Market not open yet. Blocking live mode.');
            liveToggle.checked = false;
            isLiveMode = false;
            showNotice('Market is not open yet. Showing historical data.', 7000);
            datePicker.disabled = false;
            return;
        }

        if (probe.is_holiday) {
            // Market hours but NO data → this is a holiday
            console.log('[App] Holiday detected via spot probe.');
            liveToggle.checked = false;
            isLiveMode = false;
            datePicker.disabled = false;

            // Roll date picker back to previous trading day
            const datePart = datePicker.value;
            const prev = new Date(datePart);
            prev.setDate(prev.getDate() - 1);
            // Skip weekends
            while (prev.getDay() === 0 || prev.getDay() === 6) prev.setDate(prev.getDate() - 1);
            datePicker.value = prev.toLocaleDateString('en-CA');

            showNotice(`Today is a holiday/no-data day. Showing ${datePicker.value}.`, 8000);
            fetchData();
            return;
        }

        // ── Market is open AND data exists → go live ──────────────────────
        isLiveMode = true;
        const todayStr = new Date().toLocaleDateString('en-CA');
        datePicker.value = todayStr;
        datePicker.disabled = true;

        // Auto-reset slider to the end (current time)
        const slider = document.getElementById('time-slider');
        if (slider) {
            slider.value = slider.max;
            timeSelector.updateDisplay();
        }

        // Track active symbol for WS filtering, then fetch and join
        dataService.setActiveSymbol(symbol);
        await fetchData();
        dataService.initWebSocket(symbol);

        console.log(`[App] Live mode ON for ${symbol}.`);
        showNotice(`Live mode ON — ${symbol}`, 4000);

    } else {
        // ── LIVE OFF: stop live room subscription ─────────────────────────
        console.log('[App] Live toggle OFF — stopping live mode.');
        isLiveMode = false;
        datePicker.disabled = false;

        // Auto-reset slider to the end for historical analysis
        const slider = document.getElementById('time-slider');
        if (slider) {
            slider.value = slider.max;
            timeSelector.updateDisplay();
        }

        // Leave the WS room so the server stops pushing to us,
        // but keep the socket alive for background data_updated events.
        dataService.stopLiveMode(symbol);
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

    // After 10:00 AM, if the interval wasn't manually set, default to 15m for stable viewing.
    const isLateMorning = ist.getHours() > 10 || (ist.getHours() === 10 && ist.getMinutes() >= 0);
    const currentVal = intervalSelect.value;
    
    if (isToday && isLateMorning && currentVal !== "15" && !intervalSelect.dataset.userHardSet) {
        console.log("[App] Defaulting to 15m interval (Post-10AM session).");
        intervalSelect.value = "15";
        timeSelector.reconfigure(exchange, 15);
    } else if (intervalSelect.selectedOptions[0] && intervalSelect.selectedOptions[0].disabled) {
        // Auto-switch if current is disabled (e.g. 15m at 09:15 AM)
        intervalSelect.value = firstAvailable || "1";
        timeSelector.reconfigure(exchange, parseInt(intervalSelect.value));
    }
}

// ── Start ────────────────────────────────────────────────────────────────────
timeSelector.setExchange('NSE');
document.getElementById('loading').style.display = 'none';

(async () => {
    updateIntervalAvailability();

    const initialSymbol   = symbolSelector.symbol   || 'NIFTY';
    const initialExchange = symbolSelector.exchange  || 'NSE';
    console.log(`[App] Initializing dashboard layout for ${initialSymbol} (${initialExchange})...`);

    // ── Step 1: Pre-market check ─────────────────────────────────────────
    // Ask the server what date/mode to use before we start fetching.
    // This handles: pre-market, post-market, weekends, and holiday chains.
    try {
        const preStatus = await apiService.getPreMarketStatus(initialExchange);
        console.log('[App] Pre-market status:', preStatus);

        if (preStatus.use_historical && preStatus.date) {
            // Market is closed / pre-market / holiday
            // → set date picker to the advised date, keep live toggle OFF
            datePicker.value    = preStatus.date;
            liveToggle.checked  = false;
            liveToggle.disabled = false; // still allow user to try when they think market opens
            isLiveMode          = false;

            // Reconfigure slider AFTER setting the date — this recalculates
            // slider.max with isToday=false so 15:30 becomes the correct max.
            const exchange = symbolSelector.exchange || 'NSE';
            const interval = parseInt(document.getElementById('interval-select').value) || 15;
            timeSelector.reconfigure(exchange, interval);
            // reconfigure() already snaps slider.value = slider.max (15:30),
            // but call updateDisplay() explicitly to ensure label is refreshed.
            timeSelector.updateDisplay();


            const msgMap = {
                pre_market:  `Market opens later. Showing ${preStatus.date}.`,
                after_close: `Market closed. Showing ${preStatus.date}.`,
                weekend:     `Weekend — showing last session (${preStatus.date}).`,
            };
            const msg = msgMap[preStatus.reason] || `Showing ${preStatus.date}.`;
            showNotice(msg, 7000);
        }
        // If use_historical=false the market is OPEN — normal flow below
    } catch (e) {
        console.warn('[App] Pre-market status check failed, using defaults.', e);
    }

    // ── Step 2: Track active symbol for WS filtering ──────────────────────
    dataService.setActiveSymbol(initialSymbol);

    // ── Step 3: Init WebSocket (always — needed for background fetch events) ──
    // The socket is kept alive; the active symbol filter in DataService ensures
    // only updates for the current symbol trigger re-renders.
    try {
        // 3. Trigger initial data fetch
        await fetchData();

        // 4. Then initialize WebSocket room join
        dataService.initWebSocket(initialSymbol);
    } catch (e) {
        console.error("[App] Socket initialization failed:", e);
    }
})();

// ── Background Housekeeping ──────────────────────────────────────────────
const housekeepingTask = setInterval(() => {
    updateIntervalAvailability();

    // Stop the task after 10:30 AM IST as all intervals will be unlocked by then.
    const now = new Date();
    const ist = new Date(now.toLocaleString("en-US", { timeZone: "Asia/Kolkata" }));
    if (ist.getHours() > 10 || (ist.getHours() === 10 && ist.getMinutes() >= 30)) {
        console.log("[App] Housekeeping cycle complete for today. Stopping timer.");
        clearInterval(housekeepingTask);
    }
}, 60000);
