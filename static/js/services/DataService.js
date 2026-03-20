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

    clear() {
        this._rawData = [];
        this._currentParams = null;
    }

    async load(params, silent = false) {
        // Simple check: if params haven't changed and not in live mode, maybe skip?
        // But usually we want the server to decide freshness.

        const loader = document.getElementById('loading');
        if (!silent && loader) {
            const datePart = params.date && params.date !== new Date().toLocaleDateString('en-CA') ? ` on ${params.date}` : '';
            loader.textContent = `Loading ${params.symbol}${datePart}\u2026`;
            loader.style.display = 'flex';
            loader.classList.remove('waiting'); // Reset any prior waiting state
        }

        try {
            const data = await this._api.getOptionData(params);

            if (data.error) {
                if (!silent) {
                    if (data.error.includes("Waiting for data")) {
                        // Non-blocking: Show as a hint in the UI instead of alert
                        if (loader) {
                            loader.textContent = data.error;
                            loader.style.display = 'flex';
                            loader.classList.add('waiting'); // Style this to look like a notice
                        }
                    } else {
                        if (window.showNotice) {
                            window.showNotice(data.error);
                        } else {
                            alert(data.error);
                        }
                        if (loader) loader.style.display = 'none';
                    }
                }
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
            if (!silent) {
                if (window.showNotice) {
                    window.showNotice('Error loading data. Check console.');
                } else {
                    alert('Error loading data. Check console.');
                }
            }
        } finally {
            if (loader) {
                const isWaitingBanner = loader.classList.contains('waiting');
                const hasDataNow = this._rawData && this._rawData.length > 0;

                // Hide if:
                // 1. Data actually arrived (hasDataNow)
                // 2. OR it wasn't a "waiting" banner to begin with (regular fetch finished)
                if (hasDataNow || !isWaitingBanner) {
                    loader.style.display = 'none';
                    loader.classList.remove('waiting');
                }
            }
        }
    }

    // ── WebSocket ────────────────────────────────────────────────────────────

    initWebSocket(prefix, symbol) {
        this._currentSymbol = symbol;
        this._currentPrefix = prefix;
        if (this.socket && this.socket.connected) {
            console.log("[DataService] WS Re-joining room:", symbol);
            this.socket.emit("join_symbol", { symbol: symbol, exchange: prefix });
            return;
        }

        try {
            this.socket = io();
            this.socket.on('connect', () => {
                console.log("[DataService] WS Connected. Joining:", this._currentSymbol);
                this.socket.emit("join_symbol", { symbol: this._currentSymbol, exchange: this._currentPrefix });
            });

            this.socket.on('data_updated', (data) => {
                const params = window.buildParams ? window.buildParams() : null;
                const activeSym = params ? params.symbol : this._currentSymbol;

                if (data.symbol === activeSym || (data.prefix === prefix && !data.symbol)) {
                    console.log(`[DataService] WebSocket update for ${activeSym}`);
                    
                    // If the server shifted the date (holiday fallback), update our UI state
                    if (data.date && params && data.date !== params.date) {
                        console.log(`[DataService] Server shifted date to ${data.date} (Holiday fallback)`);
                        const datePicker = document.getElementById('date-picker');
                        if (datePicker) {
                            datePicker.value = data.date;
                            // Re-build params with the new date
                            const newParams = window.buildParams();
                            this.load(newParams, true);
                            return;
                        }
                    }

                    if (params) this.load(params, true);
                }
            });

            // ── Rejoining client: a fetch is already in progress server-side ──
            this.socket.on('data_fetching', (data) => {
                console.log(`[DataService] Server: fetch in progress for ${data.symbol}`);
                const loader = document.getElementById('loading');
                if (loader) {
                    loader.textContent = data.message || `Fetching latest data for ${data.symbol}\u2026`;
                    loader.style.display = 'flex';
                    loader.classList.add('waiting');
                }
            });

            // ── Rejoining client: market is closed / weekend / holiday ─────────
            this.socket.on('market_status', (data) => {
                console.log(`[DataService] Server market_status: ${data.status} for ${data.symbol}`);
                const loader = document.getElementById('loading');

                // If we already have data loaded, market_status is just informational — don't interrupt
                if (this._rawData && this._rawData.length > 0) {
                    console.log('[DataService] Ignoring market_status — already have data.');
                    return;
                }

                const msg = data.message || 'Market is currently closed.';
                if (loader) {
                    loader.textContent = msg;
                    loader.style.display = 'flex';
                    loader.classList.add('waiting');
                }
                // Also show a non-blocking notice if the helper exists
                if (window.showNotice) window.showNotice(msg);
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
