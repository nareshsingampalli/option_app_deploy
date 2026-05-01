/**
 * TimeSelector — Structural: Component
 * Manages the time slider and displays different ranges for NSE vs MCX.
 */
class TimeSelector extends UIComponent {
    constructor(containerId, displayId) {
        super(containerId);
        this.display = document.getElementById(displayId);
        this._exchange = 'NSE';
        this._dataEndMins = null; // Set by setDataEndTime() after data loads
    }

    setExchange(exch) {
        const interval = parseInt(document.getElementById('interval-select').value) || 15;
        this.reconfigure(exch, interval);
    }

    reconfigure(exch, interval) {
        this._exchange = exch;
        this._interval = interval;
        this._dataEndMins = null; // Reset when exchange/interval changes
        
        const now = new Date();
        const ist = new Date(now.toLocaleString("en-US", { timeZone: "Asia/Kolkata" }));
        const currentMins = ist.getHours() * 60 + ist.getMinutes();
        
        const datePicker = document.getElementById('date-picker');
        const isToday = datePicker ? datePicker.value === ist.toLocaleDateString('en-CA') : false;

        let marketStart, marketEnd;
        if (exch === 'NSE') {
            marketStart = 9 * 60 + 15;
            marketEnd = 15 * 60 + 30;
        } else {
            marketStart = 9 * 60;
            marketEnd = 23 * 60 + 50;  // MCX_MARKET_END = 23:50
        }

        const start = marketStart + interval;
        // Cap the end of the slider at the current time if looking at today's data
        const end = isToday ? Math.min(marketEnd, currentMins) : marketEnd;
        
        this.container.max = Math.max(0, Math.floor((end - start) / interval));
        this.container.value = this.container.max;
        this.updateDisplay();
    }

    get time() {
        const val = parseInt(this.container.value);
        const interval = this._interval || 15;
        if (this._exchange === 'NSE') {
            const startMinutes = 9 * 60 + 15 + interval; // First completed candle
            const totalMinutes = startMinutes + (val * interval);
            return this._minutesToHHMM(totalMinutes);
        } else {
            const startMinutes = 9 * 60 + interval;
            const totalMinutes = startMinutes + (val * interval);
            return this._minutesToHHMM(totalMinutes);
        }
    }

    get isAtMax() {
        return parseInt(this.container.value) >= parseInt(this.container.max);
    }

    /**
     * setDataEndTime — called after data loads to sync slider end to the
     * actual last candle timestamp returned by the server.
     * @param {string} isoDate - e.g. "2026-04-30 15:15:00+05:30"
     */
    setDataEndTime(isoDate) {
        try {
            // Support "YYYY-MM-DD HH:MM:SS+05:30" and ISO "T" separator
            const timeStr = isoDate.replace('T', ' ').split(' ')[1]; // "HH:MM:SS+05:30"
            const parts = timeStr.split(':');
            const hh = parseInt(parts[0]);
            const mm = parseInt(parts[1]);
            if (isNaN(hh) || isNaN(mm)) return;

            const lastDataMins = hh * 60 + mm;
            const interval = this._interval || 15;
            const marketStart = this._exchange === 'NSE' ? (9 * 60 + 15) : (9 * 60);
            const start = marketStart + interval;

            const newMax = Math.max(0, Math.floor((lastDataMins - start) / interval));
            const wasAtMax = parseInt(this.container.value) >= parseInt(this.container.max);

            this._dataEndMins = lastDataMins;
            this.container.max = newMax;
            if (wasAtMax) {
                this.container.value = newMax; // Follow end if already pinned there
            }
            this.updateDisplay();
        } catch (e) {
            console.warn('[TimeSelector] setDataEndTime failed:', e);
        }
    }

    updateDisplay() {
        if (this.display) {
            this.display.textContent = this.time;

            // Update the min/max labels beneath the slider
            const labelsGrid = this.container.nextElementSibling;
            if (labelsGrid && labelsGrid.children.length >= 3) {
                const interval = this._interval || 15;
                const now = new Date();
                const ist = new Date(now.toLocaleString("en-US", { timeZone: "Asia/Kolkata" }));
                const currentMins = ist.getHours() * 60 + ist.getMinutes();
                const datePicker = document.getElementById('date-picker');
                const isToday = datePicker ? datePicker.value === ist.toLocaleDateString('en-CA') : false;

                if (this._exchange === 'NSE') {
                    const startM = 9 * 60 + 15 + interval;
                    // Prefer actual last-candle time; fall back to market close
                    const endM = this._dataEndMins ||
                        (isToday ? Math.min(15 * 60 + 30, currentMins) : 15 * 60 + 30);
                    labelsGrid.children[0].textContent = this._minutesToHHMM(startM);
                    labelsGrid.children[1].textContent = this._minutesToHHMM(startM + (endM - startM) / 2);
                    labelsGrid.children[2].textContent = this._minutesToHHMM(endM);
                } else {
                    const startM = 9 * 60 + interval;
                    // Prefer actual last-candle time; fall back to MCX_MARKET_END 23:50
                    const endM = this._dataEndMins ||
                        (isToday ? Math.min(23 * 60 + 50, currentMins) : 23 * 60 + 50);
                    labelsGrid.children[0].textContent = this._minutesToHHMM(startM);
                    labelsGrid.children[1].textContent = this._minutesToHHMM(startM + (endM - startM) / 2);
                    labelsGrid.children[2].textContent = this._minutesToHHMM(endM);
                }
            }
        }
    }

    _minutesToHHMM(totalMinutes) {
        const hours = Math.floor(totalMinutes / 60);
        const minutes = totalMinutes % 60;
        return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
    }
}
