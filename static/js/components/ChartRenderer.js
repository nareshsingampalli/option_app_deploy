/**
 * ChartRenderer — Structural: Composite Pattern
 * Renders Plotly charts for each selected metric × instrument.
 * Completely isolated rendering logic.
 */
class ChartRenderer extends UIComponent {
    constructor(containerId, metricSelector) {
        super(containerId);
        this._metrics = metricSelector;
    }

    _colorFor(symbol) {
        let hash = 0;
        for (let i = 0; i < symbol.length; i++) {
            hash = symbol.charCodeAt(i) + ((hash << 5) - hash);
        }
        const isCE = symbol.includes('CE');
        const isPE = symbol.includes('PE');
        if (isCE) {
            const l = 25 + (Math.abs(hash) % 40);
            return { color: `hsl(120, 100%, ${l}%)`, dash: 'solid' };
        } else if (isPE) {
            const l = 35 + (Math.abs(hash) % 30);
            return { color: `hsl(0, 100%, ${l}%)`, dash: 'dash' };
        }
        return { color: '#888888', dash: 'solid' };
    }

    render(rawData, selectedInstruments) {
        if (!this.container) return;
        this.container.innerHTML = '';

        const selectedMetrics = this._metrics.selected();

        if (selectedInstruments.length === 0) {
            this.container.innerHTML = '<div style="text-align:center; color:#666;">No instruments selected.</div>';
            return;
        }
        if (selectedMetrics.length === 0) {
            this.container.innerHTML = '<div style="text-align:center; color:#666;">No metrics selected.</div>';
            return;
        }

        // Group raw data by symbol
        const grouped = {};
        rawData.forEach(row => {
            if (selectedInstruments.includes(row.symbol)) {
                if (!grouped[row.symbol]) {
                    grouped[row.symbol] = { x: [], rows: [] };
                }
                grouped[row.symbol].x.push(row.date);
                grouped[row.symbol].rows.push(row);
            }
        });

        // One chart per metric
        selectedMetrics.forEach(metric => {
            const chartDiv = document.createElement('div');
            chartDiv.className = 'chart-container';
            const chartId = `chart-${metric}`;
            chartDiv.innerHTML = `
                <div class="chart-header">${this._metrics.label(metric)}</div>
                <div id="${chartId}"></div>`;
            this.container.appendChild(chartDiv);

            const traces = selectedInstruments
                .filter(sym => grouped[sym])
                .map(sym => {
                    const firstRow = grouped[sym].rows[0];
                    const lastRow = grouped[sym].rows[grouped[sym].rows.length - 1];

                    // Build pretty label: e.g. "NIFTY 22400 CE"
                    let prettyName = sym;
                    try {
                        const baseMatch = sym.match(/^[A-Z]+/);
                        const baseSym = baseMatch ? baseMatch[0] : '';
                        if (firstRow && firstRow.strike && firstRow.option_type) {
                            prettyName = `${baseSym} ${firstRow.strike} ${firstRow.option_type}`;
                        }
                    } catch (e) {
                        prettyName = sym;
                    }

                    const { color, dash } = this._colorFor(sym);
                    return {
                        x: grouped[sym].x,
                        y: grouped[sym].rows.map(r => r[metric]),
                        mode: 'lines',
                        name: prettyName,
                        line: { dash, width: 2, color }
                    };
                });

            const layout = {
                margin: { t: 20, r: 20, l: 50, b: 40 },
                height: 400,
                xaxis: { title: 'Time' },
                yaxis: { title: this._metrics.label(metric) },
                hovermode: 'x unified',
                showlegend: true,
                legend: { orientation: 'h', y: -0.2 }
            };

            Plotly.newPlot(chartId, traces, layout);
        });
    }
}
