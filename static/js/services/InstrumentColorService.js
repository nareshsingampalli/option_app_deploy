/**
 * InstrumentColorService — Redesign Theme
 * Neon Green for CE, Neon Red for PE, Neon Yellow for ATM.
 */
window.InstrumentColorService = (() => {
    const ATM_COLOR = '#facc15'; // Neon Yellow
    const CE_COLOR = '#4ade80';  // Neon Green
    const PE_COLOR = '#fb7185';  // Neon Red

    let _map = {};

    function build(instrumentInfo, spotPrice) {
        _map = {};
        if (!spotPrice || isNaN(spotPrice)) {
            instrumentInfo.forEach(i => { _map[i.symbol] = i.type === 'CE' ? CE_COLOR : PE_COLOR; });
            return;
        }

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

        instrumentInfo.forEach(i => {
            if (i.strike != null && parseFloat(i.strike) === atmStrike) {
                _map[i.symbol] = ATM_COLOR;
            } else {
                _map[i.symbol] = i.type === 'CE' ? CE_COLOR : PE_COLOR;
            }
        });
    }

    function get(symbol)  { return _map[symbol] || '#888'; }
    function getMap()     { return { ..._map }; }

    return { build, get, getMap };
})();
