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

// ── Wiring: Data → UI ────────────────────────────────────────────────────────
dataService.subscribe((records, isInitial) => {
    console.log(`[App] Data received. Records: ${records ? records.length : 0}, Initial: ${isInitial}`);
    if (!records || records.length === 0) {
        console.warn("[App] No records received.");
        return;
    }
    // 1. Update instrument list
    const hasInstruments = document.querySelectorAll('.instrument-cb').length > 0;
    if (isInitial || !hasInstruments) {
        try {
            const symbols = [...new Set(records.map(r => r.symbol))].sort();
            const instrumentInfo = symbols.map(sym => {
                const row = records.find(r => r.symbol === sym);
                let label = sym;
                if (row && row.strike && row.option_type) {
                    const baseMatch = sym.match(/^[A-Z]+/);
                    const baseSym = baseMatch ? baseMatch[0] : '';
                    label = `${baseSym} ${row.strike} ${row.option_type}`;
                }
                return { symbol: sym, label: label };
            });
            if (instrumentInfo.length > 0) {
                instrumentSelector.render(instrumentInfo);
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
            alert(`The selected date (${dateVal}) is a trading holiday (Weekend). No data will be fetched.`);
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

liveToggle.addEventListener('change', () => {
    isLiveMode = liveToggle.checked;

    // Check for weekends (0 = Sunday, 6 = Saturday)
    const day = new Date().getDay();
    if (isLiveMode && (day === 0 || day === 6)) {
        alert("Today is a trading holiday (Weekend). Live mode is unavailable.");
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
window.selectInstruments = (type) => instrumentSelector.selectAll(type);
window.refreshToken = refreshToken;

// ── Auto-Live Logic ──────────────────────────────────────────────────────────
function shouldAutoStartLive() {
    const now = new Date();
    const istTime = new Date(now.toLocaleString("en-US", { timeZone: "Asia/Kolkata" }));
    const day = istTime.getDay(); // 0=Sun, 1=Mon, ..., 6=Sat
    const hour = istTime.getHours();
    const min = istTime.getMinutes();
    const timeVal = hour * 100 + min;

    // Mon-Fri and after 09:15 AM IST
    return (day >= 1 && day <= 5 && timeVal >= 915);
}

function activateLiveMode() {
    liveToggle.checked = true;
    liveToggle.dispatchEvent(new Event('change'));
}

// ── Start ────────────────────────────────────────────────────────────────────
timeSelector.setExchange('NSE');
document.getElementById('loading').style.display = 'none';

if (shouldAutoStartLive()) {
    console.log("[App] Market is open (IST). Activating Live Mode automatically...");
    activateLiveMode();
} else {
    fetchData();
}
