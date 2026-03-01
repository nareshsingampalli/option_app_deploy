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
        this._exchange = exch;
        if (exch === 'NSE') {
            this.container.max = 25; // 09:15 to 15:30 (15 min steps)
        } else {
            this.container.max = 29; // 09:00 to 23:30 (30 min steps)
        }
        this.container.value = this.container.max;
        this.updateDisplay();
    }

    get time() {
        const val = parseInt(this.container.value);
        if (this._exchange === 'NSE') {
            const startMinutes = 9 * 60 + 15;
            const totalMinutes = startMinutes + (val * 15);
            return this._minutesToHHMM(totalMinutes);
        } else {
            const startMinutes = 9 * 60;
            const totalMinutes = startMinutes + (val * 30);
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
                    labelsGrid.children[1].textContent = '12:15';
                    labelsGrid.children[2].textContent = '15:30';
                } else {
                    labelsGrid.children[0].textContent = '09:00';
                    labelsGrid.children[1].textContent = '16:00';
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
