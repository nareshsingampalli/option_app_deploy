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

    selectAll(type, spotPrice = null) {
        const checkboxes = Array.from(this.container.querySelectorAll('.instrument-cb'));
        
        // Simple filters
        if (['all', 'none', 'ce', 'pe'].includes(type)) {
            checkboxes.forEach(cb => {
                if (type === 'all') cb.checked = true;
                else if (type === 'none') cb.checked = false;
                else if (type === 'ce') cb.checked = cb.dataset.type === 'CE';
                else if (type === 'pe') cb.checked = cb.dataset.type === 'PE';
            });
        } 
        // Complex filters (Intraday/Scalping)
        else if ((type === 'intraday' || type === 'scalping') && spotPrice) {
            // Group by type (CE/PE)
            const groups = {
                'CE': checkboxes.filter(cb => cb.dataset.type === 'CE'),
                'PE': checkboxes.filter(cb => cb.dataset.type === 'PE')
            };

            const countMap = type === 'intraday' ? { atm: 1, otm: 2, itm: 2 } : { atm: 1, otm: 1, itm: 1 };

            Object.keys(groups).forEach(optType => {
                const group = groups[optType];
                if (group.length === 0) return;

                // Sort strikes by distance from spot
                group.sort((a,b) => Math.abs(parseFloat(a.dataset.strike) - spotPrice) - Math.abs(parseFloat(b.dataset.strike) - spotPrice));

                // Clear all first
                group.forEach(cb => cb.checked = false);

                const atm = group[0]; // Nearest strike is ATM
                if (atm) atm.checked = true;

                const others = group.slice(1);
                
                // Categorize into OTM/ITM based on spot
                const itmList = others.filter(cb => {
                    const k = parseFloat(cb.dataset.strike);
                    return optType === 'CE' ? k < spotPrice : k > spotPrice;
                });
                const otmList = others.filter(cb => {
                    const k = parseFloat(cb.dataset.strike);
                    return optType === 'CE' ? k > spotPrice : k < spotPrice;
                });

                // Re-sort ITM/OTM lists by closeness to spot (already mostly sorted by distance)
                itmList.forEach((cb, i) => { if (i < countMap.itm) cb.checked = true; });
                otmList.forEach((cb, i) => { if (i < countMap.otm) cb.checked = true; });
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
