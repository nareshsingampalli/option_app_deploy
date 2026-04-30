/**
 * DecisionEngine.js
 * 
 * The intelligence layer of the dashboard.
 * Responsible for scoring signals, cluster detection, and trade suggestions.
 */
class DecisionEngine {
    constructor() {
        this.weights = {
            price_roc: 0.40,
            oi_roc: 0.25,
            vol_roc: 0.20,
            iv_roc: 0.15
        };
    }

    /**
     * Calculates a breakout score (0-100) for a given strike record.
     */
    calculateScore(record) {
        const pROC = parseFloat(record.change_in_ltp) || 0;
        const oROC = parseFloat(record.roc_oi) || 0;
        const vROC = parseFloat(record.roc_volume) || 0;
        const iROC = parseFloat(record.roc_iv) || 0;

        // Simple linear combination (normalized)
        // Note: Real-world logic would be more complex (e.g. non-linear scaling)
        let score = (
            (pROC * this.weights.price_roc) +
            (oROC * this.weights.oi_roc) +
            (vROC * this.weights.vol_roc) +
            (iROC * this.weights.iv_roc)
        );

        // Map to 0-100 range (example scaling factor)
        score = Math.max(0, Math.min(100, score * 5));
        return Math.round(score);
    }

    /**
     * Ranks the top signals from the current data set.
     */
    getTopSignals(data, limit = 5) {
        if (!data || data.length === 0) return [];

        const signals = data.map(record => ({
            ...record,
            score: this.calculateScore(record),
            signalType: this.determineSignalType(record)
        }));

        // Sort by score descending
        return signals
            .filter(s => s.score > 20) // Filter out low-confidence noise
            .sort((a, b) => b.score - a.score)
            .slice(0, limit);
    }

    determineSignalType(record) {
        if (record.roc_oi > 10 && record.change_in_ltp > 2) return 'Breakout';
        if (record.roc_iv > 5 && record.change_in_ltp < -2) return 'IV Spike / Reversal';
        if (record.roc_volume > 20) return 'Volume Momentum';
        return 'Trend Following';
    }

    /**
     * Detects clusters where multiple strikes show synchronized rising metrics.
     */
    detectClusters(data) {
        const clusters = [];
        // Group by 100-point ranges if needed, or look for adjacent strikes
        // For now, identifying individual strikes with high confluence
        data.forEach(record => {
            if (record.roc_oi > 8 && record.roc_volume > 15 && record.roc_iv > 3) {
                clusters.push({
                    strike: record.strike,
                    type: 'Cluster',
                    reason: 'Confluence of OI + Vol + IV'
                });
            }
        });
        return clusters;
    }

    /**
     * Generates a trade suggestion for a high-scoring signal.
     */
    generateSuggestion(signal, spotPrice) {
        const isCE = signal.instrument_type === 'CE';
        const entry = signal.ltp;
        const stopLoss = isCE ? entry * 0.85 : entry * 0.85; // Simple 15% SL
        const target = isCE ? entry * 1.30 : entry * 1.30;   // Simple 30% Target

        return {
            strike: `${signal.strike} ${signal.instrument_type}`,
            entry: entry.toFixed(2),
            sl: stopLoss.toFixed(2),
            target: target.toFixed(2),
            reason: `Breakout detected with Score ${signal.score}. ${signal.signalType} momentum.`
        };
    }
}

// Global instance
const decisionEngine = new DecisionEngine();
