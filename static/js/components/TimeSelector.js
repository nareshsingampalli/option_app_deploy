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
        
        if (exch === 'NSE') {
            const start = 9 * 60 + 15;
            const end = 15 * 60 + 30;
            this.container.max = Math.floor((end - start) / interval);
        } else {
            const start = 9 * 60;
            const end = 23 * 60 + 30;
            this.container.max = Math.floor((end - start) / interval);
        }
        this.container.value = this.container.max;
        this.updateDisplay();
    }

    get time() {
        const val = parseInt(this.container.value);
        const interval = this._interval || 15;
        if (this._exchange === 'NSE') {
            const startMinutes = 9 * 60 + 15;
            const totalMinutes = startMinutes + (val * interval);
            return this._minutesToHHMM(totalMinutes);
        } else {
            const startMinutes = 9 * 60;
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
                if (this._exchange === 'NSE') {
                    labelsGrid.children[0].textContent = '09:15';
                    // Dynamic middle label
                    const totalT = (15 * 60 + 30) - (9 * 60 + 15);
                    labelsGrid.children[1].textContent = this._minutesToHHMM(9*60+15 + totalT/2);
                    labelsGrid.children[2].textContent = '15:30';
                } else {
                    labelsGrid.children[0].textContent = '09:00';
                    const totalT = (23 * 60 + 30) - (9 * 60);
                    labelsGrid.children[1].textContent = this._minutesToHHMM(9*60 + totalT/2);
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
