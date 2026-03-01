/**
 * MetricSelector — manages metric (criteria) checkboxes.
 */
class MetricSelector {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this._LABELS = {
            'ltp': 'Last Traded Price',
            'change_in_ltp': 'Change in LTP',
            'roc_oi': 'Rate of Change in Open Interest (%)',
            'roc_volume': 'Rate of Change in Volume (%)',
            'roc_iv': 'Rate of Change in IV (%)',
            'coi_vol_ratio': 'Change in OI / Volume Ratio',
            'spot_price': 'Spot Price'
        };
    }

    label(key) {
        return this._LABELS[key] || key;
    }

    selected() {
        return Array.from(
            document.querySelectorAll('.metric-cb:checked')
        ).map(cb => cb.value);
    }
}
