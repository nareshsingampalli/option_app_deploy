/**
 * SignalsPanel.js
 * 
 * Renders the high-confidence signal cards at the top of the dashboard.
 */
class SignalsPanel extends BaseComponent {
    constructor(containerId) {
        super(containerId);
        this.signals = [];
    }

    render() {
        if (!this.container) return;

        if (this.signals.length === 0) {
            this.container.innerHTML = `
                <div class="no-signals">
                    Monitoring market for high-probability setups...
                </div>
            `;
            return;
        }

        const html = this.signals.map(s => this.createSignalCard(s)).join('');
        this.container.innerHTML = `
            <div class="signals-grid">
                ${html}
            </div>
        `;
    }

    update(data) {
        // Use DecisionEngine to get top 4 signals
        this.signals = decisionEngine.getTopSignals(data, 4);
        this.render();
    }

    createSignalCard(signal) {
        const isCE = signal.instrument_type === 'CE';
        const colorClass = isCE ? 'signal-ce' : 'signal-pe';
        const confidence = signal.score > 80 ? 'High' : (signal.score > 50 ? 'Medium' : 'Low');
        const confidenceClass = `conf-${confidence.toLowerCase()}`;

        return `
            <div class="signal-card ${colorClass}" onclick="app.focusStrike('${signal.strike}', '${signal.instrument_type}')">
                <div class="signal-header">
                    <span class="signal-strike">${signal.strike} ${signal.instrument_type}</span>
                    <span class="signal-score">${signal.score}</span>
                </div>
                <div class="signal-type">${signal.signalType}</div>
                <div class="signal-meta">
                    <span class="confidence ${confidenceClass}">${confidence} Confidence</span>
                    <span class="signal-ltp">LTP: ${signal.ltp}</span>
                </div>
            </div>
        `;
    }
}
