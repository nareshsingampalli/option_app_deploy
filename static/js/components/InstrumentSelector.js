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
                    <input type="checkbox" class="instrument-cb" 
                           value="${inst.symbol}" 
                           data-strike="${inst.strike || ''}"
                           data-type="${inst.type || ''}"
                           checked>
                    ${inst.label || inst.symbol}
                </label>`;
            this.container.appendChild(div);
        });
        // Notify on any checkbox change
        this.container.querySelectorAll('.instrument-cb').forEach(cb => {
            cb.addEventListener('change', () => this._notify());
        });
    }

    applySelection(states, spotPrice = null) {
        const checkboxes = Array.from(this.container.querySelectorAll('.instrument-cb'));
        
        // Clear all first
        checkboxes.forEach(cb => cb.checked = false);

        if (states.all) {
            checkboxes.forEach(cb => cb.checked = true);
        } else if (states.none) {
            // Already cleared
        } else {
            const isIntra = states.intraday;
            const isScalp = states.scalping;
            // If neither is true, checking All CE/PE should pick ALL strikes of that type.
            // If either IS true, we use the ATM/OTM/ITM strike count.
            const isDynamic = isIntra || isScalp;
            const countMap = isScalp ? { atm: 1, otm: 1, itm: 1 } : { atm: 1, otm: 2, itm: 2 };

            // Group by type (CE/PE)
            const groups = {
                'CE': checkboxes.filter(cb => cb.dataset.type === 'CE' && states.ce),
                'PE': checkboxes.filter(cb => cb.dataset.type === 'PE' && states.pe)
            };

            // If neither CE nor PE toggle is active, act as if BOTH are active (default)
            if (!states.ce && !states.pe) {
                groups.CE = checkboxes.filter(cb => cb.dataset.type === 'CE');
                groups.PE = checkboxes.filter(cb => cb.dataset.type === 'PE');
            }

            Object.keys(groups).forEach(optType => {
                const group = groups[optType];
                if (group.length === 0) return;

                if (!isDynamic) {
                    // Just select the whole group
                    group.forEach(cb => cb.checked = true);
                } else if (spotPrice) {
                    // Select relative to spot
                    group.sort((a,b) => Math.abs(parseFloat(a.dataset.strike) - spotPrice) - Math.abs(parseFloat(b.dataset.strike) - spotPrice));

                    const atm = group[0]; 
                    if (atm) atm.checked = true;

                    const others = group.slice(1);
                    const itmList = others.filter(cb => {
                        const k = parseFloat(cb.dataset.strike);
                        return optType === 'CE' ? k < spotPrice : k > spotPrice;
                    });
                    const otmList = others.filter(cb => {
                        const k = parseFloat(cb.dataset.strike);
                        return optType === 'CE' ? k > spotPrice : k < spotPrice;
                    });

                    itmList.forEach((cb, i) => { if (i < countMap.itm) cb.checked = true; });
                    otmList.forEach((cb, i) => { if (i < countMap.otm) cb.checked = true; });
                }
            });
        }
        
        this._notify();
    }

    selected() {
        return Array.from(
            this.container.querySelectorAll('.instrument-cb:checked')
        ).map(cb => cb.value);
    }
}
