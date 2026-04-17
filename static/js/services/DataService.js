/**
 * DataService — Behavioral: Observer Pattern (Subject)
 * Fetches data from the API and notifies all registered UI components.
 * UI components never call fetch() directly — they subscribe here.
 *
 * Active symbol tracking
 * ─────────────────────
 * _activeSymbol : the symbol the user is currently watching (room we are in)
 * _liveMode     : true when live toggle is ON and market confirmed open+not-holiday
 *
 * WS room lifecycle
 * ─────────────────
 * - Symbol switch  → emit join_symbol (server-side handler leaves old room)
 * - Live OFF       → emit leave_symbol, set _liveMode=false
 * - Full teardown  → stopWebSocket() disconnects the socket
 */
class DataService {
    constructor(apiService) {
        this._api = apiService;
        this._observers = [];
        this._activeSymbol = null;   // currently watched symbol
        this._liveMode     = false;  // is live mode currently active?
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
        
        // Cooldown removed for maximum responsiveness.

        this._isLoading = true;
        this._lastLoadTime = Date.now();

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
                    if (data.error.includes("Processing") || data.status === "fetching") {
                        if (loader) {
                            const dateLabel = params.date || "Today's";
                            loader.innerHTML = `<span class="spinner"></span> Catching up on ${dateLabel} data... Please wait.`;
                            loader.style.display = 'flex';
                            loader.classList.add('waiting');
                        }
                        // Wait for server to finish fetching
                        if (!this.socket) {
                            console.log("[DataService] Server is fetching... awaiting result.");
                        }
                    } else if (data.error.includes("Waiting for data")) {
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

            // Auto-sync UI live toggle if present in meta
            const liveToggle = document.getElementById('live-toggle');
            if (liveToggle && meta.live !== undefined) {
                liveToggle.checked = (meta.live === true || meta.live === "true");
            }

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

    /**
     * setActiveSymbol — tracks which symbol is currently being viewed.
     * Any symbol switch must call this so data_updated events are filtered correctly.
     */
    setActiveSymbol(symbol) {
        this._activeSymbol = symbol ? symbol.toUpperCase() : null;
    }

    /**
     * initWebSocket — creates the socket once per page load and joins the symbol's room.
     * Subsequent calls (symbol switch) just switch rooms via join_symbol.
     */
    initWebSocket(symbol) {
        if (!symbol) return;
        const upperSym = symbol.toUpperCase();
        this._activeSymbol = upperSym;
        this._currentSymbol = upperSym; // Keep for backwards compat
        const interval = parseInt(document.getElementById('interval-select').value) || 15;
        const nextExp = document.getElementById('next-expiry-chk')?.checked || false;

        // If socket exists and is connected, just switch the room
        if (this.socket && this.socket.connected) {
            console.log(`[DataService] Switching WebSocket room: ${this._activeSymbol} → ${upperSym} (int: ${interval}m, next: ${nextExp})`);
            this.socket.emit("join_symbol", { symbol: upperSym, interval: interval, next_expiry: nextExp });
            return;
        }

        // Only create a NEW connection if none exists or it's disconnected
        if (!this.socket) {
            try {
                console.log(`[DataService] Establishing single WebSocket for tab...`);
                this.socket = io({
                    reconnection: true,
                    reconnectionAttempts: 5,
                    transports: ['websocket', 'polling']
                });

                this.socket.on('connect', () => {
                    const currentInterval = parseInt(document.getElementById('interval-select').value) || 15;
                    const isNext = document.getElementById('next-expiry-chk')?.checked || false;
                    console.log(`[DataService] WS connected. Joining room: ${this._activeSymbol} (int: ${currentInterval}m, next: ${isNext})`);
                    this.socket.emit("join_symbol", { 
                        symbol: this._activeSymbol, 
                        interval: currentInterval,
                        next_expiry: isNext
                    });
                });

                // ── data_updated: only react to the ACTIVELY watched symbol & interval ──
                this.socket.on('data_updated', (data) => {
                    const params = window.buildParams ? window.buildParams() : null;
                    if (!params || !data.symbol) return;

                    const incomingSymbol = data.symbol.toUpperCase();
                    const activeSymbol   = params.symbol.toUpperCase();
                    
                    // Filter by symbol
                    if (incomingSymbol !== activeSymbol) {
                        console.log(`[WS] Symbol mismatch: ${incomingSymbol} vs active ${activeSymbol}`);
                        return;
                    }

                    // Filter by interval
                    const incomingInterval = parseInt(data.interval);
                    const activeInterval   = parseInt(params.interval);
                    if (!isNaN(incomingInterval) && incomingInterval !== activeInterval) {
                        console.log(`[WS] Interval mismatch: ${incomingInterval}m vs active ${activeInterval}m`);
                        return; 
                    }

                    // Filter by expiry track (Handle 'true'/'false' strings from buildParams)
                    const incomingNext = !!data.next_expiry;
                    const activeNext   = (params.next_expiry === true || params.next_expiry === 'true');
                    
                    if (incomingNext !== activeNext) {
                        console.log(`[WS] Expiry mismatch: Next=${incomingNext} vs active Next=${activeNext}`);
                        return;
                    }

                    console.log(`[WS] Update accepted for ${incomingSymbol} (${incomingInterval}m, next=${incomingNext}). Refreshing UI...`);
                    this.load(params, true);
                });

                // ── holiday_detected: only react to the ACTIVELY watched symbol ──
                this.socket.on('holiday_detected', (data) => {
                    if (!data.symbol) return;
                    const incoming = data.symbol.toUpperCase();
                    if (incoming !== this._activeSymbol) return;

                    console.warn(`[DataService] ${data.date} is a holiday/no-data. Fallback: ${data.fallback_date || 'client-side'}`);
                    const loader = document.getElementById('loading');
                    if (loader) {
                        loader.style.display = 'none';
                        loader.classList.remove('waiting');
                    }
                    this._notify([], true, "holiday", data.fallback_date || null);
                });

                // ── data_fetching: background fetch progress indicator ──
                this.socket.on('data_fetching', (data) => {
                    if (!data.symbol) return;
                    const incoming = data.symbol.toUpperCase();
                    if (incoming !== this._activeSymbol) return;

                    const loader = document.getElementById('loading');
                    if (loader) {
                        loader.textContent = data.message || `Fetching ${incoming}…`;
                        loader.style.display = 'flex';
                        loader.classList.add('waiting');
                    }
                });

                this.socket.on('disconnect', () => {
                    console.log('[DataService] WebSocket disconnected.');
                });

            } catch (e) {
                console.error("[DataService] WS Connection failed", e);
            }
        }
    }

    /**
     * stopLiveMode — called when user turns off the Live toggle.
     * Leaves the live room on the server but keeps the socket alive
     * so background data_updated events (for non-live requests) still work.
     */
    stopLiveMode(symbol) {
        this._liveMode = false;
        if (this.socket && this.socket.connected && symbol) {
            console.log(`[DataService] Live OFF → leaving room: ${symbol}`);
            this.socket.emit("leave_symbol", { symbol: symbol.toUpperCase() });
        }
    }

    /**
     * stopWebSocket — full teardown. Disconnects the socket entirely.
     * Only call this when the user navigates away or the page is closed.
     */
    stopWebSocket() {
        this._activeSymbol = null;
        this._liveMode = false;
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
