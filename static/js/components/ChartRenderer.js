/**
 * ChartRenderer — Structural: Composite Pattern
 * Renders Plotly charts for each selected metric × instrument.
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
        const absHash = Math.abs(hash);
        const isCE = symbol.includes('CE');
        const isPE = symbol.includes('PE');

        if (isCE) {
            // Broad hue range: 85 (yellow-green) to 155 (spring green)
            const h = 85 + (absHash % 71);
            // Saturation: 60 to 100
            const s = 60 + (absHash % 41);
            // Lightness: 30 to 65
            const l = 30 + (absHash % 36);
            return { color: `hsl(${h}, ${s}%, ${l}%)`, dash: 'solid' };
        } else if (isPE) {
            // Broad hue range: 340 (pinks) to 30 (oranges)
            const h = (340 + (absHash % 51)) % 360;
            const s = 60 + (absHash % 41);
            const l = 35 + (absHash % 36);
            return { color: `hsl(${h}, ${s}%, ${l}%)`, dash: 'dash' };
        }
        return { color: '#888888', dash: 'solid' };
    }

    clear() {
        if (this.container) this.container.innerHTML = '';
    }

    render(rawData, selectedInstruments) {
        if (!this.container || !rawData || rawData.length === 0) {
            if (this.container) this.container.innerHTML = '<div style="text-align:center; color:#666; padding: 20px;">Waiting for data...</div>';
            return;
        }

        const selectedMetrics = this._metrics.selected();
        if (selectedInstruments.length === 0 || selectedMetrics.length === 0) {
            this.container.innerHTML = '<div style="text-align:center; color:#666; padding: 20px;">Select instruments and metrics to view charts.</div>';
            return;
        }

        this.container.innerHTML = '';

        // Standardize Date Helper
        const parseDate = (dStr) => {
            if (!dStr) return null;
            const isoStr = dStr.includes('T') ? dStr : dStr.replace(' ', 'T');
            const d = new Date(isoStr);
            return isNaN(d.getTime()) ? null : d;
        };

        // Determine the "Reference Day" from the LATEST data point
        const sortedRaw = [...rawData].sort((a, b) => parseDate(a.date) - parseDate(b.date));
        const lastRow = sortedRaw[sortedRaw.length - 1];
        const targetDay = lastRow.date.split(' ')[0];

        // Group data by symbol and determine clean labels
        const grouped = {};
        const symbolLabels = {};
        rawData.forEach(row => {
            if (selectedInstruments.includes(row.symbol)) {
                if (!grouped[row.symbol]) {
                    grouped[row.symbol] = { rows: [] };

                    // Generate clean label: NAME + STRIKE (Removing Date, CE, PE as requested)
                    const baseMatch = row.symbol.match(/^[A-Z]+/);
                    const baseSym = baseMatch ? baseMatch[0] : '';
                    const strike = row.strike || '';
                    symbolLabels[row.symbol] = `${baseSym} ${strike}`.trim();
                }
                grouped[row.symbol].rows.push(row);
            }
        });

        selectedMetrics.forEach(metric => {
            const chartDiv = document.createElement('div');
            chartDiv.className = 'chart-container';
            const chartId = `chart-${metric}`;
            chartDiv.innerHTML = `
                <div class="chart-header">${this._metrics.label(metric)}</div>
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
                    x: xSorted,
                    y: xSorted.map(d => dateToSpot.get(d)),
                    mode: 'lines',
                    name: 'Spot Price',
                    line: { width: 3, color: '#1f77b4' }
                });
            } else {
                traces = selectedInstruments
                    .filter(sym => grouped[sym])
                    .map(sym => {
                        const { color, dash } = this._colorFor(sym);
                        const isCE = sym.includes('CE');
                        return {
                            x: grouped[sym].rows.map(r => r.date),
                            y: grouped[sym].rows.map(r => r[metric]),
                            mode: 'lines',
                            name: symbolLabels[sym] || sym,
                            isCE: isCE, // Store for sorting
                            line: { dash, width: 2, color }
                        };
                    });

                // Group: CE first, then PE. Sort by max value (high) descending within groups
                traces.sort((a, b) => {
                    if (a.isCE && !b.isCE) return -1;
                    if (!a.isCE && b.isCE) return 1;

                    const aMax = Math.max(...a.y.filter(v => v !== null && !isNaN(v)), -Infinity);
                    const bMax = Math.max(...b.y.filter(v => v !== null && !isNaN(v)), -Infinity);
                    return bMax - aMax;
                });
            }

            // Filter Traces to TODAY only
            traces.forEach(t => {
                const filtered = t.x.map((x, i) => ({ x, y: t.y[i] }))
                    .filter(pt => pt.x.startsWith(targetDay));
                t.x = filtered.map(p => p.x);
                t.y = filtered.map(p => p.y);
            });

            // Calculate Bounds
            const isMCX = selectedInstruments.some(s => ['CRUDEOIL', 'NATURALGAS', 'SILVER', 'GOLD'].some(m => s.includes(m)));
            const mStartObj = parseDate(`${targetDay}T${isMCX ? '09:00:00' : '09:15:00'}`);

            const allTimes = traces.flatMap(t => t.x).map(x => parseDate(x).getTime()).sort((a, b) => a - b);
            const maxTime = allTimes.length > 0 ? allTimes[allTimes.length - 1] : Date.now();
            const maxTimeObj = new Date(maxTime);

            // X-Axis Window (Initial Zoom: Last 3 hours)
            const windowMs = 3 * 60 * 60 * 1000;
            const viewStart = Math.max(mStartObj.getTime(), maxTime - windowMs);

            // Robust Y-Axis Range logic
            let yRange = null;
            const allY = traces.flatMap(t => t.y).filter(v => typeof v === 'number' && !isNaN(v));
            if (allY.length > 1) {
                let min, max;
                if (allY.length > 10) {
                    const sortedY = [...allY].sort((a, b) => a - b);
                    min = sortedY[Math.floor(sortedY.length * 0.10)];
                    max = sortedY[Math.floor(sortedY.length * 0.90)];
                } else {
                    min = Math.min(...allY);
                    max = Math.max(...allY);
                }
                const diff = max - min;
                const sens = metric.includes('roc') || metric.includes('ratio') ? 0.05 : 10;
                const padding = Math.max(diff * 0.3, sens);
                yRange = [min - padding, max + padding];
                if (['ltp', 'spot_price', 'coi_vol_ratio'].includes(metric)) {
                    yRange[0] = Math.max(0, yRange[0]);
                }
            }

            const layout = {
                margin: { t: 40, r: 30, l: 60, b: 80 },
                height: 550,
                xaxis: {
                    range: [new Date(viewStart), maxTimeObj],
                    minallowed: mStartObj,
                    maxallowed: maxTimeObj,
                    autorange: false,
                    automargin: true,
                    rangeslider: { visible: true, thickness: 0.15, range: [mStartObj, maxTimeObj] },
                    type: 'date',
                    tickformat: '%H:%M',
                    hoverformat: '%H:%M',
                    title: ''
                },
                yaxis: {
                    title: this._metrics.label(metric),
                    automargin: true,
                    fixedrange: false,
                    range: yRange,
                    tickformat: '.2f',
                    hoverformat: '.2f'
                },
                hovermode: 'x unified',
                dragmode: 'pan',
                showlegend: false,
                legend: { orientation: 'h', y: -0.4, x: 0 }
            };

            const config = {
                responsive: true,
                displayModeBar: true, // Shows zoom/pan/autoscale buttons
                scrollZoom: true,
                editable: false, // Disables the "Click to enter" placeholders
                doubleClick: 'reset+autosize' // Double-tap to reset view
            };

            Plotly.newPlot(chartId, traces, layout, config);
        });
    }
}
