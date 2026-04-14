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
        this._isLoading = false;    // Guard against re-entrant calls
        this._lastLoadTime = 0;     // Cooling down to avoid loops
    }

    // ── Observer registration ────────────────────────────────────────────────

    /** Subscribe a callback to receive new data. */
    subscribe(fn) {
        this._observers.push(fn);
    }

    _notify(data, isInitial, status = "success", errorCode = null) {
        this._observers.forEach(fn => fn(data, isInitial, status, errorCode));
    }

    onInstrumentsChanged(fn) {
        this._onInstrumentsChanged = fn;
    }

    // ── Public API ───────────────────────────────────────────────────────────

    get rawData() {
        return this._rawData;
    }

    clear() {
        this.clearData();
        this.clearMeta();
    }

    clearData() {
        this._rawData = [];
        this._currentParams = null;
    }

    clearMeta() {
        const spotEl = document.getElementById('spot-price-display');
        const expiryEl = document.getElementById('expiry-date-display');
        const updatedEl = document.getElementById('last-updated');
        if (spotEl) spotEl.textContent = '';
        if (expiryEl) expiryEl.textContent = '';
        if (updatedEl) updatedEl.textContent = '';
    }

    async load(params, silent = false) {
        if (this._isLoading) {
            console.log("[DataService] Load already in progress, skipping...");
            return;
        }
        
        // Cooldown: skip if triggered within 2 seconds of last fetch (prevent loops)
        const now = Date.now();
        if (silent && (now - this._lastLoadTime) < 2000) {
            console.log("[DataService] Cooling down, skipping background fetch.");
            return;
        }

        this._isLoading = true;
        this._lastLoadTime = now;

        const loader = document.getElementById('loading');
        if (!silent && loader) {
            const datePart = params.date && params.date !== new Date().toLocaleDateString('en-CA') ? ` on ${params.date}` : '';
            loader.textContent = `Loading ${params.symbol}${datePart}\u2026`;
            loader.style.display = 'flex';
            loader.classList.remove('waiting'); // Reset any prior waiting state
        }

        try {
            const data = await this._api.getOptionData(params);
            const status = data.status || "success";
            const errorCode = (data.meta && data.meta.error_code) ? data.meta.error_code : null;

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
                this._notify([], !silent, status, errorCode);
                return;
            }

            const records = Array.isArray(data) ? data : (data.data || []);
            this._rawData = records;
            const meta = data.meta || {};

            this._updateMeta(meta);
            this._notify(records, !silent, status, errorCode);

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
            this._isLoading = false;
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
            const params = window.buildParams ? window.buildParams() : { interval: 15 };
            this.socket.emit("join_symbol", { symbol: symbol, exchange: prefix, interval: params.interval });
            return;
        }

        try {
            this.socket = io();
            this.socket.on('connect', () => {
                const params = window.buildParams ? window.buildParams() : { interval: 15 };
                this.socket.emit("join_symbol", { symbol: this._currentSymbol, exchange: this._currentPrefix, interval: params.interval });
            });

            this.socket.on('data_updated', (data) => {
                const params = window.buildParams ? window.buildParams() : null;
                const activeSym = params ? params.symbol : this._currentSymbol;

                if (data.symbol === activeSym || (data.prefix === prefix && !data.symbol)) {
                    console.log(`[DataService] WebSocket update for ${activeSym}`, data);
                    
                    // If the server shifted the date (e.g. holiday fallback), sync our UI state
                    if (data.date && params && data.date !== params.date) {
                        console.log(`[DataService] Server provided date ${data.date} (Syncing from ${params.date})`);
                        const datePicker = document.getElementById('date-picker');
                        if (datePicker) {
                            datePicker.value = data.date;
                            // Reconfigure time slider for the new (past) date so it shows full market range
                            if (window._timeSelector) {
                                const exchange = window._symbolSelector ? window._symbolSelector.exchange : 'NSE';
                                const interval = parseInt(document.getElementById('interval-select').value) || 15;
                                window._timeSelector.reconfigure(exchange, interval);
                            }
                            // Re-fetch with the newly synchronized parameters
                            const newParams = window.buildParams();
                            this.load(newParams, true);
                            return;
                        }
                    }

                    if (params) {
                         this.load(params, true);
                         if (data.status) {
                             this._notify(this._rawData, false, data.status, data.error_code);
                         }
                    }
                }
            });

            this.socket.on('instruments_changed', (data) => {
                console.log("[DataService] Instruments changed event:", data);
                if (this._onInstrumentsChanged) {
                    this._onInstrumentsChanged(data.instruments);
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
        const headerExpiryVal = document.getElementById('header-expiry-val');
        const updatedEl = document.getElementById('last-updated');
        const datePicker = document.getElementById('date-picker');

        // Sync Date Picker if the server shifted or provided a different date (e.g. Holiday/Weekend fallback)
        if (datePicker && meta.date && datePicker.value !== meta.date) {
            console.log(`[DataService] Syncing UI date picker to: ${meta.date}`);
            datePicker.value = meta.date;
            // Reconfigure time slider for the shifted (past) date — full market range, slider at end
            if (window._timeSelector) {
                const exchange = window._symbolSelector ? window._symbolSelector.exchange : 'NSE';
                const interval = parseInt(document.getElementById('interval-select').value) || 15;
                window._timeSelector.reconfigure(exchange, interval);
            }
        }

        if (spotEl && meta.spot_price) {
            spotEl.textContent = `Spot Price: ${meta.spot_price}`;
        }
        if (meta.expiry_date) {
            if (expiryEl) expiryEl.textContent = `Expiry: ${meta.expiry_date}`;
            
            // Format for the header segment as requested in image mockup: e.g. (14- Apr)
            if (headerExpiryVal) {
                try {
                    const dt = new Date(meta.expiry_date);
                    const month = dt.toLocaleString('default', { month: 'short' });
                    headerExpiryVal.textContent = `${dt.getDate()}- ${month}`;
                } catch (e) {
                    headerExpiryVal.textContent = meta.expiry_date;
                }
            }
        }
        if (updatedEl && meta.fetched_at) {
            updatedEl.textContent = `Last Updated: ${new Date(meta.fetched_at).toLocaleTimeString()}`;
        }
    }
}
