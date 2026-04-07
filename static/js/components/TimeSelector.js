/**
 * TimeSelector — Structural: Component
 * Manages the time slider and displays different ranges for NSE vs MCX.
 */
class TimeSelector extends UIComponent {
    constructor(containerId, displayId) {
        super(containerId);
        this.display = document.getElementById(displayId);
        this._exchange = 'NSE';
    }

    setExchange(exch) {
        const interval = parseInt(document.getElementById('interval-select').value) || 15;
        this.reconfigure(exch, interval);
    }

    reconfigure(exch, interval) {
        this._exchange = exch;
        this._interval = interval;
        
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
            marketEnd = 23 * 60 + 30;
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

    updateDisplay() {
        if (this.display) {
            this.display.textContent = this.time;

            // Update the min/max labels beneath the slider
            const labelsGrid = this.container.nextElementSibling;
            if (labelsGrid && labelsGrid.children.length >= 3) {
                const interval = this._interval || 15;
                if (this._exchange === 'NSE') {
                    labelsGrid.children[0].textContent = this._minutesToHHMM(9*60+15 + interval);
                    // Dynamic middle label
                    const totalT = (15 * 60 + 30) - (9 * 60 + 15 + interval);
                    labelsGrid.children[1].textContent = this._minutesToHHMM(9*60+15 + interval + totalT/2);
                    labelsGrid.children[2].textContent = '15:30';
                } else {
                    labelsGrid.children[0].textContent = this._minutesToHHMM(9*60 + interval);
                    const totalT = (23 * 60 + 30) - (9 * 60 + interval);
                    labelsGrid.children[1].textContent = this._minutesToHHMM(9*60 + interval + totalT/2);
                    labelsGrid.children[2].textContent = '23:30';
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
