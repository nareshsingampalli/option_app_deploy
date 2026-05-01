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
    _notify() { this._onChange.forEach(fn => fn(this.selected())); }

    clear() {
        if (this.container) this.container.innerHTML = '<div style="text-align:center; padding:10px; color:#999; font-size:12px;">Loading instruments...</div>';
        this._lastInstrumentInfo = null;
    }

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
                const k = parseFloat(inst.strike);
                const isCE = inst.type === 'CE';

                // ATM detection logic (CE=down, PE=up)
                // We calculate this relative to all strikes in the current info set
                const allStrikes = [...new Set(instrumentInfo.map(x => parseFloat(x.strike)).filter(s => !isNaN(s)))].sort((a, b) => a - b);
                const atmCE = allStrikes.filter(s => s <= spotPrice).pop() || allStrikes[0];
                const atmPE = allStrikes.filter(s => s >= spotPrice).shift() || allStrikes[allStrikes.length - 1];

                const targetAtm = isCE ? atmCE : atmPE;
                const isATM = k === targetAtm;
                const isITM = !isATM && (isCE ? k < targetAtm : k > targetAtm);
                let mLabel;
                if (isATM) mLabel = 'ATM';
                else if (isITM) mLabel = 'ITM';
                else mLabel = 'OTM';

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
                   <span style="white-space:nowrap;">${inst.label} </span>
                    ${mTag}
                </label>`;
            this.container.appendChild(div);
        });

        this.container.querySelectorAll('.instrument-cb').forEach(cb => {
            cb.addEventListener('change', () => this._notify());
        });

        // Center ATM strike on initial load/refresh
        if (spotPrice) {
            // Small timeout to ensure layout is complete for offsetTop calculation
            setTimeout(() => this._scrollToAtm(spotPrice), 100);
        }
    }

    // ── Re-color in place (when spot updates without a new instrument load) ──
    recolor(spotPrice) {
        if (!this._lastInstrumentInfo) return;
        InstrumentColorService.build(this._lastInstrumentInfo, spotPrice);

        this.container.querySelectorAll('.instrument-cb').forEach(cb => {
            const sym = cb.value;
            const color = InstrumentColorService.get(sym);
            const label = cb.closest('label');
            if (!label) return;

            const isCE = cb.dataset.type === 'CE';
            const moneyColor = isCE ? '#26a641' : '#df3333';

            // Revert name color to default (remove inline style)
            const nameSpan = label.querySelector('span:not([style*="font-weight:700"])');
            if (nameSpan) nameSpan.style.color = '';

            // Update badge (if it exists)
            const badge = label.querySelector('span[style*="font-weight:700"]');
            if (badge) {
                let bgOverlay = color.replace(/,\s*[\d.]+\s*\)$/, ', 0.12)');
                if (bgOverlay === color) { bgOverlay = color + '1a'; }
                badge.style.color = color;
                badge.style.borderColor = color;
                badge.style.backgroundColor = bgOverlay;

                // Redetermine label (Type-specific: CE=down, PE=up)
                const strike = parseFloat(cb.dataset.strike);
                if (!isNaN(strike)) {
                    const allStrikes = [...new Set(this._lastInstrumentInfo.map(x => parseFloat(x.strike)).filter(s => !isNaN(s)))].sort((a, b) => a - b);
                    const atmCE = allStrikes.filter(s => s <= spotPrice).pop() || allStrikes[0];
                    const atmPE = allStrikes.filter(s => s >= spotPrice).shift() || allStrikes[allStrikes.length - 1];

                    const targetAtm = isCE ? atmCE : atmPE;
                    const isATM = strike === targetAtm;
                    const isITM = !isATM && (isCE ? strike < targetAtm : strike > targetAtm);
                    badge.textContent = isATM ? 'ATM' : (isITM ? 'ITM' : 'OTM');
                }
            }
        });
    }

    // ── Selection helpers ────────────────────────────────────────────────────
    applySelection(states, spotPrice = null, notify = true) {
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
            const isIntra = states.intraday;
            const isScalp = states.scalping;
            const isDynamic = isIntra || isScalp;
            // nSide: how many strikes to pick on each side of ATM
            // Scalping: ATM ± 1 = 3 strikes  |  Intraday: ATM ± 2 = 5 strikes
            const nSide = isScalp ? 1 : 2;

            // Determine which types are active
            const useCE = states.ce || (!states.ce && !states.pe);
            const usePE = states.pe || (!states.ce && !states.pe);

            if (!isDynamic) {
                // Non-dynamic: check everything in the active type groups
                checkboxes.forEach(cb => {
                    if (cb.dataset.type === 'CE' && !useCE) return;
                    if (cb.dataset.type === 'PE' && !usePE) return;
                    cb.checked = true;
                });
            } else if (spotPrice) {
                // ── Strike-centric selection ─────────────────────────────────
                const allStrikes = [...new Set(
                    checkboxes.map(cb => parseFloat(cb.dataset.strike)).filter(s => !isNaN(s))
                )].sort((a, b) => a - b);

                if (allStrikes.length > 0) {
                    const atmCE = allStrikes.filter(s => s <= spotPrice).pop() || allStrikes[0];
                    const atmPE = allStrikes.filter(s => s >= spotPrice).shift() || allStrikes[allStrikes.length - 1];

                    const idxCE = allStrikes.indexOf(atmCE);
                    const idxPE = allStrikes.indexOf(atmPE);

                    const ceFrom = Math.max(0, idxCE - nSide);
                    const ceTo = Math.min(allStrikes.length - 1, idxCE + nSide);
                    const ceStrikes = new Set(allStrikes.slice(ceFrom, ceTo + 1));

                    const peFrom = Math.max(0, idxPE - nSide);
                    const peTo = Math.min(allStrikes.length - 1, idxPE + nSide);
                    const peStrikes = new Set(allStrikes.slice(peFrom, peTo + 1));

                    checkboxes.forEach(cb => {
                        const type = cb.dataset.type;
                        const k = parseFloat(cb.dataset.strike);

                        if (type === 'CE') {
                            if (useCE && ceStrikes.has(k)) cb.checked = true;
                        } else if (type === 'PE') {
                            if (usePE && peStrikes.has(k)) cb.checked = true;
                        }
                    });
                }
            }
        }

        if (notify) this._notify();
    }

    selected() {
        return Array.from(
            this.container.querySelectorAll('.instrument-cb:checked')
        ).map(cb => cb.value);
    }

    /** Center the closest strike to spotPrice in the sidebar viewport */
    _scrollToAtm(spotPrice) {
        if (!this.container || !spotPrice) return;
        const checkboxes = Array.from(this.container.querySelectorAll('.instrument-cb'));
        if (checkboxes.length === 0) return;

        // Find closest strike row
        const targetCb = checkboxes.reduce((prev, curr) => {
            const pS = parseFloat(prev.dataset.strike) || 0;
            const cS = parseFloat(curr.dataset.strike) || 0;
            return Math.abs(cS - spotPrice) < Math.abs(pS - spotPrice) ? curr : prev;
        });

        const row = targetCb.closest('.control-group');
        if (row) {
            row.scrollIntoView({ block: 'center', behavior: 'smooth' });
        }
    }
}
