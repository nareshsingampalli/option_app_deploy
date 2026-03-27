/**
 * InstrumentSelector — Structural: Composite Pattern
 * Manages the instrument checkbox list and filter buttons.
 * Notifies subscribers via Observer pattern when selection changes.
 *
 * Color source: InstrumentColorService (shared with ChartRenderer)
 *   – instrument name text  → same color
 *   – circle dot border     → same color (fill at 18% opacity)
 *   – ATM chip stays golden, ITM/OTM chip is neutral (monochrome)
 */
class InstrumentSelector extends UIComponent {
    constructor(containerId) {
        super(containerId);
        this._onChange = [];
        this._lastInstrumentInfo = null;
    }

    onChange(fn) { this._onChange.push(fn); }
    _notify()    { this._onChange.forEach(fn => fn(this.selected())); }

    // ── Render ───────────────────────────────────────────────────────────────
    /**
     * @param {Array}       instrumentInfo  [{symbol, label, strike, type}]
     * @param {number|null} spotPrice
     */
    render(instrumentInfo, spotPrice = null) {
        if (!this.container) return;
        this._lastInstrumentInfo = instrumentInfo;

        // Build / rebuild the color map so all three components are in sync
        InstrumentColorService.build(instrumentInfo, spotPrice);

        this.container.innerHTML = '';
        instrumentInfo.forEach(inst => {
            const color = InstrumentColorService.get(inst.symbol);

            // ── Moneyness chip — color from InstrumentColorService ───────────
            let mTag = '';
            if (inst.strike != null && spotPrice && inst.type) {
                const k       = parseFloat(inst.strike);
                const isCE    = inst.type === 'CE';
                const isATM   = Math.abs(k - spotPrice) / spotPrice < 0.0015;
                const isITM   = !isATM && (isCE ? k < spotPrice : k > spotPrice);
                let mLabel;
                if (isATM)      mLabel = 'ATM';
                else if (isITM) mLabel = 'ITM';
                else            mLabel = 'OTM';

                // Badge uses the moneyness color as its text + border color
                let bgOverlay = color.replace(/,\s*[\d.]+\s*\)$/, ', 0.12)');
                if (bgOverlay === color) { bgOverlay = color + '1a'; } // fallback if not rgba
                mTag = `<span style="font-size:9px;font-weight:700;padding:1px 5px;border-radius:3px;margin-left:3px;letter-spacing:0.04em;color:${color};border:1px solid ${color};background:${bgOverlay}">${mLabel}</span>`;
            }

            // ── Row ──────────────────────────────────────────────────────────
            const div = document.createElement('div');
            div.className = 'control-group';

            div.innerHTML = `
                <label style="display:flex;align-items:center;gap:3px;cursor:pointer;">
                    <input type="checkbox" class="instrument-cb"
                           value="${inst.symbol}"
                           data-strike="${inst.strike ?? ''}"
                           data-type="${inst.type ?? ''}"
                           checked style="margin-right:3px;flex-shrink:0;">
                    <span style="white-space:nowrap;">${inst.label || inst.symbol}</span>
                    ${mTag}
                </label>`;
            this.container.appendChild(div);
        });

        this.container.querySelectorAll('.instrument-cb').forEach(cb => {
            cb.addEventListener('change', () => this._notify());
        });
    }

    // ── Re-color in place (when spot updates without a new instrument load) ──
    recolor(spotPrice) {
        if (!this._lastInstrumentInfo) return;
        InstrumentColorService.build(this._lastInstrumentInfo, spotPrice);

        this.container.querySelectorAll('.instrument-cb').forEach(cb => {
            const sym   = cb.value;
            const color = InstrumentColorService.get(sym);
            const label = cb.closest('label');
            if (!label) return;

            const isCE       = cb.dataset.type === 'CE';
            const moneyColor = isCE ? '#26a641' : '#df3333';
            
            // Revert name color to default (remove inline style)
            const nameSpan = label.querySelector('span:not([style*="font-weight:700"])');
            if (nameSpan) nameSpan.style.color = '';

            // Update badge (if it exists)
            const badge = label.querySelector('span[style*="font-weight:700"]');
            if (badge) {
                let bgOverlay = color.replace(/,\s*[\d.]+\s*\)$/, ', 0.12)');
                if (bgOverlay === color) { bgOverlay = color + '1a'; }
                badge.style.color           = color;
                badge.style.borderColor     = color;
                badge.style.backgroundColor = bgOverlay;
                
                // Redetermine label 
                const strike    = parseFloat(cb.dataset.strike);
                if (!isNaN(strike)) {
                    const isATM = Math.abs(strike - spotPrice) / spotPrice < 0.0015;
                    const isITM = !isATM && (isCE ? strike < spotPrice : strike > spotPrice);
                    badge.textContent = isATM ? 'ATM' : (isITM ? 'ITM' : 'OTM');
                }
            }
        });
    }

    // ── Selection helpers ────────────────────────────────────────────────────
    applySelection(states, spotPrice = null) {
        const checkboxes = Array.from(this.container.querySelectorAll('.instrument-cb'));

        checkboxes.forEach(cb => cb.checked = false);

        if (states.all) {
            checkboxes.forEach(cb => cb.checked = true);
        } else if (states.none) {
            if (spotPrice) {
                const ceGroup = checkboxes.filter(cb => cb.dataset.type === 'CE');
                const peGroup = checkboxes.filter(cb => cb.dataset.type === 'PE');
                [ceGroup, peGroup].forEach(group => {
                    if (group.length > 0) {
                        group.sort((a, b) =>
                            Math.abs(parseFloat(a.dataset.strike) - spotPrice) -
                            Math.abs(parseFloat(b.dataset.strike) - spotPrice)
                        );
                        group[0].checked = true;
                    }
                });
            }
        } else {
            const isIntra  = states.intraday;
            const isScalp  = states.scalping;
            const isDynamic = isIntra || isScalp;
            const countMap  = isScalp ? { atm: 1, otm: 1, itm: 1 } : { atm: 1, otm: 2, itm: 2 };

            const groups = {
                CE: checkboxes.filter(cb => cb.dataset.type === 'CE' && states.ce),
                PE: checkboxes.filter(cb => cb.dataset.type === 'PE' && states.pe),
            };

            if (!states.ce && !states.pe) {
                groups.CE = checkboxes.filter(cb => cb.dataset.type === 'CE');
                groups.PE = checkboxes.filter(cb => cb.dataset.type === 'PE');
            }

            Object.keys(groups).forEach(optType => {
                const group = groups[optType];
                if (group.length === 0) return;

                if (!isDynamic) {
                    group.forEach(cb => cb.checked = true);
                } else if (spotPrice) {
                    group.sort((a, b) =>
                        Math.abs(parseFloat(a.dataset.strike) - spotPrice) -
                        Math.abs(parseFloat(b.dataset.strike) - spotPrice)
                    );
                    const atm    = group[0];
                    if (atm) atm.checked = true;
                    const others  = group.slice(1);
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
