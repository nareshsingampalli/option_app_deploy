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
    dataService.onInstrumentsChanged((instruments) => {
        instrumentSelector.render(instruments, currentReferenceSpotPrice);
    });

    // -- Data Subscription ------------------
    dataService.subscribe((records, isInitial, status, errorCode) => {
        if (!records || records.length === 0) return;

        const targetTime = timeSelector.time;
        const matchingRow = records.find(r => r.date.includes(targetTime)) || records[records.length - 1];
        const spotPrice = parseFloat(matchingRow.spot_price) || null;
        currentReferenceSpotPrice = spotPrice;

        // Render signals
        signalsPanel.update(records);
        updateHeaderMetrics(records, spotPrice);
        renderCharts();

        // Sync sidebar
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
    });

    // -- Event Listeners ------------------
    symbolSelector.onChange(async ({ exchange, symbol }) => {
        timeSelector.setExchange(exchange);
        dataService.setActiveSymbol(symbol);
        dataService.clear();
        dataService.initWebSocket(symbol);
        fetchData();
    });

    instrumentSelector.onChange(selected => {
        renderCharts();
    });

    datePicker.addEventListener('change', () => {
        updateIntervalAvailability();
        fetchData();
    });

    document.getElementById('time-slider').addEventListener('input', () => {
        timeSelector.updateDisplay();
        fetchData(true);
    });

    intervalSelect.addEventListener('change', () => {
        timeSelector.reconfigure(symbolSelector.exchange, parseInt(intervalSelect.value));
        fetchData();
    });

    liveToggle.addEventListener('change', async () => {
        isLiveMode = liveToggle.checked;
        if (isLiveMode) {
            datePicker.disabled = true;
            dataService.initWebSocket(symbolSelector.symbol);
        } else {
            datePicker.disabled = false;
            dataService.stopLiveMode(symbolSelector.symbol);
        }
    });

    // Initial load
    initLoad();
}

async function fetchData(silent = false) {
    await dataService.load(buildParams(), silent);
}

function buildParams() {
    const datePicker = document.getElementById('date-picker');
    const intervalSelect = document.getElementById('interval-select');
    return {
        exchange: symbolSelector.exchange,
        symbol: symbolSelector.symbol,
        date: datePicker.value,
        time: timeSelector.time,
        interval: intervalSelect.value,
        next_expiry: document.getElementById('next-expiry-chk')?.checked ? 'true' : 'false',
        live: isLiveMode ? 'true' : 'false'
    };
}

function renderCharts() {
    chartRenderer.render(dataService.rawData, instrumentSelector.selected(), currentReferenceSpotPrice);
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
    datePicker.value = new Date().toLocaleDateString('en-CA');
    updateIntervalAvailability();
    fetchData();
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
    }
};
