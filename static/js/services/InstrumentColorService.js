/**
 * InstrumentColorService — Singleton
 *
 * Two clearly distinct base colors per option type:
 *
 *   CE ITM  → DARK  green  hsl(130, 72%, 28%)  — opacity: 0.90 / 0.65 / 0.42
 *   CE OTM  → LIGHT green  hsl(130, 50%, 60%)  — opacity: 0.72 / 0.46 / 0.22
 *   PE ITM  → DARK  red    hsl(  0, 72%, 33%)  — opacity: 0.90 / 0.65 / 0.42
 *   PE OTM  → LIGHT pink   hsl(  0, 55%, 66%)  — opacity: 0.72 / 0.46 / 0.22
 *   ATM     → golden yellow hsla(45, 88%, 42%, 0.95)
 *
 * rank-1 = nearest ATM, rank-3 = deepest/farthest.
 * Within ITM: deeper = higher opacity (easier to see).
 * Within OTM: farther = lower opacity (fades away).
 */
window.InstrumentColorService = (() => {

    const ATM_COLOR = 'hsla(45, 88%, 42%, 0.95)';

    // Two distinct bases per family
    const PALETTE = {
        CE: {
            ITM: { h: 130, s: 72, l: 28 },   // dark green
            OTM: { h: 130, s: 50, l: 60 },   // light green
        },
        PE: {
            ITM: { h: 0,   s: 72, l: 33 },   // dark red
            OTM: { h: 0,   s: 55, l: 66 },   // light pink
        },
    };

    // Opacity per rank (index 0 = rank-1 nearest ATM, index 2 = rank-3 deepest/farthest)
    const ITM_ALPHA = [0.42, 0.65, 0.90];   // gets MORE opaque deeper in
    const OTM_ALPHA = [0.72, 0.46, 0.22];   // gets LESS opaque farther out

    const FALLBACK = {
        CE: `hsla(130, 65%, 38%, 0.70)`,
        PE: `hsla(0,   65%, 40%, 0.70)`,
    };

    let _map = {};

    function _c({ h, s, l }, alpha) {
        return `hsla(${h}, ${s}%, ${l}%, ${alpha})`;
    }

    function build(instrumentInfo, spotPrice) {
        _map = {};

        if (!spotPrice || isNaN(spotPrice)) {
            instrumentInfo.forEach(i => { _map[i.symbol] = FALLBACK[i.type] || '#888'; });
            return;
        }

        // ── Find ATM strike = the single strike closest to spot (same logic as ChartRenderer)
        const allStrikes = [...new Set(
            instrumentInfo
                .filter(i => i.strike != null)
                .map(i => parseFloat(i.strike))
                .filter(k => !isNaN(k))
        )];
        const atmStrike = allStrikes.length
            ? allStrikes.reduce((best, k) =>
                Math.abs(k - spotPrice) < Math.abs(best - spotPrice) ? k : best
            )
            : null;
        const isATM = (strike) => atmStrike !== null && parseFloat(strike) === atmStrike;

        ['CE', 'PE'].forEach(type => {
            const isCE  = type === 'CE';
            const group = instrumentInfo.filter(i => i.type === type && i.strike != null);

            const atm = group.filter(i => isATM(i.strike));
            const itm = group.filter(i => {
                const k = parseFloat(i.strike);
                return !isATM(i.strike) && (isCE ? k < spotPrice : k > spotPrice);
            });
            const otm = group.filter(i => {
                const k = parseFloat(i.strike);
                return !isATM(i.strike) && (isCE ? k > spotPrice : k < spotPrice);
            });

            // ATM → yellow (both CE and PE)
            atm.forEach(i => { _map[i.symbol] = ATM_COLOR; });

            // ITM — sorted closest→deepest, opacity increases
            const itmSorted = [...itm].sort((a, b) =>
                isCE
                    ? parseFloat(b.strike) - parseFloat(a.strike)
                    : parseFloat(a.strike) - parseFloat(b.strike)
            );
            itmSorted.forEach((inst, idx) => {
                const alpha = ITM_ALPHA[Math.min(idx, ITM_ALPHA.length - 1)];
                _map[inst.symbol] = _c(PALETTE[type].ITM, alpha);
            });

            // OTM — sorted closest→farthest, opacity decreases
            const otmSorted = [...otm].sort((a, b) =>
                isCE
                    ? parseFloat(a.strike) - parseFloat(b.strike)
                    : parseFloat(b.strike) - parseFloat(a.strike)
            );
            otmSorted.forEach((inst, idx) => {
                const alpha = OTM_ALPHA[Math.min(idx, OTM_ALPHA.length - 1)];
                _map[inst.symbol] = _c(PALETTE[type].OTM, alpha);
            });
        });

        // Fallback for anything without type/strike
        instrumentInfo.forEach(i => { if (!_map[i.symbol]) _map[i.symbol] = '#888'; });
    }

    function get(symbol)  { return _map[symbol] || '#888'; }
    function getMap()     { return { ..._map }; }

    return { build, get, getMap };
})();
