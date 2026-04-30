/**
 * app.js — Orchestrator
 * Wires together all components and services.
 */

// -- Instantiate services & components with safety guards --------------------
let apiService, dataService, symbolSelector, timeSelector, instrumentSelector, metricSelector, chartRenderer, signalsPanel;
let currentReferenceSpotPrice = null;
let isLiveMode = false;
let currentRenderedSymbol = null;
let currentRenderedDate = null;

document.addEventListener('DOMContentLoaded', async () => {
    try {
        apiService = new ApiService();
        dataService = new DataService(apiService);
        symbolSelector = new SymbolSelector('symbol-selector-container');
        timeSelector = new TimeSelector('time-slider', 'time-display');
        instrumentSelector = new InstrumentSelector('instrument-list');
        metricSelector = new MetricSelector('metric-list');
        chartRenderer = new ChartRenderer('charts-area'); 
        signalsPanel = new SignalsPanel('signals-panel-container');
        
        window._timeSelector = timeSelector;
        window._symbolSelector = symbolSelector;

        setupApp();
    } catch (e) {
        console.error("[App] Initialization failed:", e);
    }
});

function setupApp() {
    const nextExpiryChk = document.getElementById('next-expiry-chk');
    const datePicker = document.getElementById('date-picker');
    const liveToggle = document.getElementById('live-toggle');
    const intervalSelect = document.getElementById('interval-select');

    // -- Real-time Instrument List Observer ------------------
    if (dataService && instrumentSelector) {
        dataService.onInstrumentsChanged((instruments) => {
            if (instrumentSelector.container) {
                instrumentSelector.render(instruments, currentReferenceSpotPrice);
            }
        });
    }

    // -- Data Subscription ------------------
    if (dataService) {
        dataService.subscribe((records, isInitial, status, errorCode) => {
            if (!records || records.length === 0) return;

            const targetTime = timeSelector && timeSelector.container ? timeSelector.time : '';
            const matchingRow = records.find(r => r.date.includes(targetTime)) || records[records.length - 1];
            const spotPrice = parseFloat(matchingRow.spot_price) || null;
            currentReferenceSpotPrice = spotPrice;

            // Render signals
            if (signalsPanel && signalsPanel.container) signalsPanel.update(records);
            updateHeaderMetrics(records, spotPrice);
            renderCharts();

            // Sync sidebar
            if (instrumentSelector && instrumentSelector.container) {
                const currentSymbol = symbolSelector.symbol;
                const currentDate = (matchingRow && matchingRow.date) ? matchingRow.date.split(' ')[0] : '';
                if (currentRenderedSymbol !== currentSymbol || currentRenderedDate !== currentDate) {
                    const symbols = [...new Set(records.map(r => r.symbol))].sort();
                    const info = symbols.map(sym => {
                        const row = records.find(r => r.symbol === sym);
                        return { symbol: sym, label: sym, strike: row.strike, type: row.option_type };
                    });
                    instrumentSelector.render(info, spotPrice);
                    currentRenderedSymbol = currentSymbol;
                    currentRenderedDate = currentDate;
                } else {
                    instrumentSelector.recolor(spotPrice);
                }
            }
        });
    }

    // -- Event Listeners ------------------
    if (symbolSelector && symbolSelector.container) {
        symbolSelector.onChange(async ({ exchange, symbol }) => {
            if (timeSelector && timeSelector.container) timeSelector.setExchange(exchange);
            dataService.setActiveSymbol(symbol);
            dataService.clear();
            dataService.initWebSocket(symbol);
            fetchData();
        });
    }

    if (instrumentSelector && instrumentSelector.container) {
        instrumentSelector.onChange(selected => {
            renderCharts();
        });
    }

    if (datePicker) {
        datePicker.addEventListener('change', () => {
            updateIntervalAvailability();
            fetchData();
        });
    }

    const timeSlider = document.getElementById('time-slider');
    if (timeSlider) {
        timeSlider.addEventListener('input', () => {
            if (timeSelector && timeSelector.container) {
                timeSelector.updateDisplay();
                fetchData(true);
            }
        });
    }

    if (intervalSelect) {
        intervalSelect.addEventListener('change', () => {
            if (timeSelector && timeSelector.container) {
                timeSelector.reconfigure(symbolSelector.exchange, parseInt(intervalSelect.value));
            }
            fetchData();
        });
    }

    if (liveToggle) {
        liveToggle.addEventListener('change', async () => {
            isLiveMode = liveToggle.checked;
            if (isLiveMode) {
                if (datePicker) datePicker.disabled = true;
                dataService.initWebSocket(symbolSelector.symbol);
            } else {
                if (datePicker) datePicker.disabled = false;
                dataService.stopLiveMode(symbolSelector.symbol);
            }
        });
    }

    // Initial load
    initLoad();
}

async function fetchData(silent = false) {
    await dataService.load(buildParams(), silent);
}

function buildParams() {
    const datePicker = document.getElementById('date-picker');
    const intervalSelect = document.getElementById('interval-select');
    const nextExpiryChk = document.getElementById('next-expiry-chk');
    
    const dateValue = datePicker ? datePicker.value : new Date().toLocaleDateString('en-CA');
    const intervalValue = intervalSelect ? intervalSelect.value : '15';
    
    return {
        exchange: symbolSelector.exchange,
        symbol: symbolSelector.symbol,
        date: dateValue,
        time: timeSelector && timeSelector.container ? timeSelector.time : '',
        interval: intervalValue,
        next_expiry: nextExpiryChk?.checked ? 'true' : 'false',
        live: isLiveMode ? 'true' : 'false'
    };
}

function renderCharts() {
    if (chartRenderer && chartRenderer.container && instrumentSelector) {
        const selected = instrumentSelector.container ? instrumentSelector.selected() : [];
        const metrics = metricSelector && metricSelector.container ? metricSelector.selected() : null;
        chartRenderer.render(dataService.rawData, selected, metrics, currentReferenceSpotPrice);
    }
}

function updateHeaderMetrics(records, spotPrice) {
    const spotEl = document.getElementById('header-spot-price');
    if (spotEl) spotEl.textContent = spotPrice ? spotPrice.toFixed(2) : '--';
}

function updateIntervalAvailability() {
    const intervalSelect = document.getElementById('interval-select');
    if (!intervalSelect) return;
    Array.from(intervalSelect.options).forEach(opt => {
        opt.disabled = false;
        opt.style.backgroundColor = '';
        opt.style.color = '';
    });
}

async function initLoad() {
    const datePicker = document.getElementById('date-picker');
    if (datePicker) {
        datePicker.value = new Date().toLocaleDateString('en-CA');
    }
    
    // Ensure WebSocket is initialized for the default symbol
    if (dataService && symbolSelector) {
        dataService.initWebSocket(symbolSelector.symbol);
    }

    updateIntervalAvailability();
    fetchData();
    
    // Hide loading overlay once initial fetch is triggered
    const loadingEl = document.getElementById('loading');
    if (loadingEl) {
        setTimeout(() => loadingEl.style.display = 'none', 1000);
    }
}

async function refreshToken() {
    try {
        const result = await apiService.refreshToken();
        if (result.status === 'success') {
            alert('Token reloaded successfully!');
            fetchData(); // Retry fetching with new token
        } else {
            alert('Failed: ' + result.message);
        }
    } catch (err) {
        alert('Error: ' + err.message);
    }
}

window.refreshToken = refreshToken;

window.app = {
    currentTab: 'overview',
    switchTab(tabId) {
        this.currentTab = tabId;
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.innerText.toLowerCase().includes(tabId));
        });
        chartRenderer.setTab(tabId);
        renderCharts();
    },
    focusStrike(strike, type) {
        this.switchTab('overview');
        // Logic to select strike...
    },
    filterInstruments(type) {
        if (!instrumentSelector) return;
        const states = {
            all: type === 'all',
            none: type === 'clear',
            ce: type === 'ce',
            pe: type === 'pe',
            intraday: type === 'intraday',
            scalping: type === 'scalping'
        };
        instrumentSelector.applySelection(states, currentReferenceSpotPrice);
        
        // Update button active states
        document.querySelectorAll('.instrument-filters .btn').forEach(btn => {
            btn.classList.toggle('active', btn.innerText.toLowerCase().includes(type));
        });
    }
};
