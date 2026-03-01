/**
 * DataService — Behavioral: Observer Pattern (Subject)
 * Fetches data from the API and notifies all registered UI components.
 * UI components never call fetch() directly — they subscribe here.
 */
class DataService {
    constructor(apiService) {
        this._api = apiService;
        this._observers = [];
        this._rawData = [];
        this._currentParams = null; // Track last loaded params
    }

    // ── Observer registration ────────────────────────────────────────────────

    /** Subscribe a callback to receive new data. */
    subscribe(fn) {
        this._observers.push(fn);
    }

    _notify(data, isInitial) {
        this._observers.forEach(fn => fn(data, isInitial));
    }

    // ── Public API ───────────────────────────────────────────────────────────

    get rawData() {
        return this._rawData;
    }

    async load(params, silent = false) {
        // Simple check: if params haven't changed and not in live mode, maybe skip?
        // But usually we want the server to decide freshness.

        const loader = document.getElementById('loading');
        if (!silent && loader) {
            loader.textContent = `Fetching data for ${params.symbol}…`;
            loader.style.display = 'flex';
        }

        try {
            const data = await this._api.getOptionData(params);

            if (data.error) {
                if (!silent) alert(data.error);
                this._updateMeta(data.meta || {});
                return;
            }

            const records = Array.isArray(data) ? data : (data.data || []);
            this._rawData = records;
            const meta = data.meta || {};

            this._updateMeta(meta);
            this._notify(records, !silent);

        } catch (err) {
            console.error('[DataService] fetch error:', err);
            if (!silent) alert('Error loading data. Check console.');
        } finally {
            if (loader) loader.style.display = 'none';
        }
    }

    // ── WebSocket ────────────────────────────────────────────────────────────

    initWebSocket(prefix, symbol) {
        this._currentSymbol = symbol;
        if (this.socket && this.socket.connected) {
            console.log("[DataService] WS Re-joining room:", symbol);
            this.socket.emit("join_symbol", { symbol: symbol });
            return;
        }

        try {
            this.socket = io();
            this.socket.on('connect', () => {
                console.log("[DataService] WS Connected. Joining:", this._currentSymbol);
                this.socket.emit("join_symbol", { symbol: this._currentSymbol });
            });

            this.socket.on('data_updated', (data) => {
                const params = window.buildParams ? window.buildParams() : null;
                const activeSym = params ? params.symbol : this._currentSymbol;

                if (data.symbol === activeSym || (data.prefix === prefix && !data.symbol)) {
                    console.log(`[DataService] WebSocket update for ${activeSym}`);
                    if (params) this.load(params, true);
                }
            });
        } catch (e) {
            console.error("[DataService] WS Init failed", e);
        }
    }

    stopWebSocket() {
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
        }
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    _updateMeta(meta) {
        const spotEl = document.getElementById('spot-price-display');
        const expiryEl = document.getElementById('expiry-date-display');
        const updatedEl = document.getElementById('last-updated');

        if (spotEl) spotEl.textContent = meta.spot_price ? `Spot Price: ${meta.spot_price}` : '';
        if (expiryEl) expiryEl.textContent = meta.expiry_date ? `Expiry: ${meta.expiry_date}` : '';
        if (updatedEl && meta.fetched_at) {
            updatedEl.textContent = `Last Updated: ${new Date(meta.fetched_at).toLocaleTimeString()}`;
        }
    }
}
