/**
 * ChartRenderer
 * Renders Plotly charts based on the active tab and selected instruments.
 */
class ChartRenderer extends UIComponent {
    constructor(containerId) {
        super(containerId);
        this.currentTab = 'overview';
        this.metricsMap = {
            'overview': ['spot_price', 'ltp'],
            'oi': ['roc_oi'],
            'vol': ['roc_volume'],
            'iv': ['roc_iv'],
            'ltp': ['ltp', 'change_in_ltp']
        };
        this.labelsMap = {
            'spot_price': 'Spot Price',
            'ltp': 'Last Traded Price (LTP)',
            'roc_oi': 'Rate of Change: Open Interest (%)',
            'roc_volume': 'Rate of Change: Volume (%)',
            'roc_iv': 'Rate of Change: IV (%)',
            'change_in_ltp': 'LTP Change (%)'
        };
    }

    setTab(tabId) {
        this.currentTab = tabId;
        console.log(`[ChartRenderer] Active Tab: ${tabId}`);
    }

    _colorFor(symbol) {
        const color = (window.InstrumentColorService && InstrumentColorService.get(symbol))
            || (symbol.includes('CE') ? 'var(--neon-green)' : 'var(--neon-red)');
        return { color, dash: 'solid' };
    }

    render(rawData, selectedInstruments, referenceSpotPrice = null) {
        if (!this.container || !rawData || rawData.length === 0) {
            if (this.container) this.container.innerHTML = '<div style="text-align:center; color:#94a3b8; padding: 40px;">Monitoring market data...</div>';
            return;
        }

        const metrics = this.metricsMap[this.currentTab] || ['ltp'];
        this.container.innerHTML = '';

        // Standardize Date Helper
        const parseDate = (dStr) => {
            if (!dStr) return null;
            const isoStr = dStr.includes('T') ? dStr : dStr.replace(' ', 'T');
            const d = new Date(isoStr);
            return isNaN(d.getTime()) ? null : d;
        };

        const sortedRaw = [...rawData].sort((a, b) => parseDate(a.date) - parseDate(b.date));
        const lastRow = sortedRaw[sortedRaw.length - 1];
        const targetDay = lastRow.date.split(' ')[0];

        metrics.forEach(metric => {
            const chartDiv = document.createElement('div');
            chartDiv.className = 'chart-container';
            const chartId = `chart-${metric}`;
            chartDiv.innerHTML = `
                <div class="chart-header">${this.labelsMap[metric] || metric}</div>
                <div id="${chartId}"></div>
            `;
            this.container.appendChild(chartDiv);

            let traces = [];

            if (metric === 'spot_price') {
                const dateToSpot = new Map();
                rawData.forEach(row => {
                    if (row.spot_price) dateToSpot.set(row.date, row.spot_price);
                });
                const xSorted = Array.from(dateToSpot.keys()).sort();
                traces.push({
                    x: xSorted.filter(x => x.startsWith(targetDay)),
                    y: xSorted.filter(x => x.startsWith(targetDay)).map(d => dateToSpot.get(d)),
                    mode: 'lines',
                    name: 'Spot Price',
                    line: { width: 3, color: 'var(--accent-blue)' },
                    fill: 'tozeroy',
                    fillcolor: 'rgba(56, 189, 248, 0.05)'
                });
            } else {
                selectedInstruments.forEach(sym => {
                    const rows = rawData.filter(r => r.symbol === sym && r.date.startsWith(targetDay));
                    if (rows.length > 0) {
                        const { color } = this._colorFor(sym);
                        traces.push({
                            x: rows.map(r => r.date),
                            y: rows.map(r => r[metric]),
                            mode: 'lines+markers',
                            marker: { size: 4 },
                            name: sym,
                            line: { width: 2, color }
                        });
                    }
                });
            }

            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                margin: { t: 20, r: 20, l: 60, b: 40 },
                height: 450,
                xaxis: {
                    gridcolor: 'rgba(255,255,255,0.05)',
                    tickfont: { color: '#94a3b8' },
                    type: 'date',
                    tickformat: '%H:%M'
                },
                yaxis: {
                    gridcolor: 'rgba(255,255,255,0.05)',
                    tickfont: { color: '#94a3b8' },
                    automargin: true,
                    fixedrange: false
                },
                hovermode: 'x unified',
                hoverlabel: { bgcolor: '#1e293b', font: { color: '#f8fafc' } },
                showlegend: true,
                legend: { orientation: 'h', y: -0.2, font: { color: '#94a3b8' } }
            };

            Plotly.newPlot(chartId, traces, layout, { responsive: true, displayModeBar: false });
        });
    }
}
