/**
 * InstrumentSelector — Structural: Composite Pattern
 * Manages the instrument checkbox list and filter buttons.
 * Notifies subscribers via Observer pattern when selection changes.
 */
class InstrumentSelector extends UIComponent {
    constructor(containerId) {
        super(containerId);
        this._onChange = [];
    }

    onChange(fn) {
        this._onChange.push(fn);
    }

    _notify() {
        this._onChange.forEach(fn => fn(this.selected()));
    }

    render(instrumentInfo) {
        if (!this.container) return;
        this.container.innerHTML = '';
        instrumentInfo.forEach(inst => {
            const div = document.createElement('div');
            div.className = 'control-group';
            div.innerHTML = `
                <label>
                    <input type="checkbox" class="instrument-cb" value="${inst.symbol}" checked>
                    ${inst.label || inst.symbol}
                </label>`;
            this.container.appendChild(div);
        });
        // Notify on any checkbox change
        this.container.querySelectorAll('.instrument-cb').forEach(cb => {
            cb.addEventListener('change', () => this._notify());
        });
    }

    selectAll(type) {
        const checkboxes = this.container.querySelectorAll('.instrument-cb');
        checkboxes.forEach(cb => {
            if (type === 'all') cb.checked = true;
            else if (type === 'none') cb.checked = false;
            else if (type === 'ce') cb.checked = cb.value.includes('CE');
            else if (type === 'pe') cb.checked = cb.value.includes('PE');
        });
        this._notify();
    }

    selected() {
        return Array.from(
            this.container.querySelectorAll('.instrument-cb:checked')
        ).map(cb => cb.value);
    }
}
