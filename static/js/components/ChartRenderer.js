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

    clear() {
        if (this.container) this.container.innerHTML = '';
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

            let traces = [];

            if (metric === 'spot_price') {
                const dateToSpot = new Map();
                rawData.forEach(row => {
                    if (row.spot_price !== undefined && row.spot_price !== null && row.spot_price !== "") {
                        if (!dateToSpot.has(row.date)) {
                            dateToSpot.set(row.date, row.spot_price);
                        }
                    }
                });

                const xDates = Array.from(dateToSpot.keys()).sort();
                const yPrices = xDates.map(date => dateToSpot.get(date));

                let underlyingName = 'Underlying';
                if (selectedInstruments.length > 0) {
                    const match = selectedInstruments[0].match(/^[A-Z]+/);
                    if (match) underlyingName = match[0];
                }

                traces.push({
                    x: xDates,
                    y: yPrices,
                    mode: 'lines',
                    name: `${underlyingName} Spot`,
                    line: { width: 3, color: '#1f77b4' }
                });
            } else {
                // 1. Separate CE and PE groups
                const ces = selectedInstruments.filter(s => s.includes('CE'));
                const pes = selectedInstruments.filter(s => s.includes('PE'));

                // 2. Sorting function by strike
                const sortByStrike = (a, b) => {
                    const getStrike = (s) => parseInt(s.match(/\d+/)?.[0] || 0);
                    return getStrike(a) - getStrike(b);
                };
                ces.sort(sortByStrike);
                pes.sort(sortByStrike);

                // 3. Global high comparison
                const getGlobalMaxHigh = (syms) => {
                    let max = 0;
                    syms.forEach(sym => {
                        if (grouped[sym]) {
                            grouped[sym].rows.forEach(r => {
                                // Prefer 'high', fallback to 'ltp' or 'close'
                                const val = parseFloat(r.high || r.ltp || r.close || 0);
                                if (val > max) max = val;
                            });
                        }
                    });
                    return max;
                };

                const maxCe = getGlobalMaxHigh(ces);
                const maxPe = getGlobalMaxHigh(pes);

                // 4. Final ordered list: higher group first
                const orderedInstruments = (maxPe > maxCe) ? [...pes, ...ces] : [...ces, ...pes];

                traces = orderedInstruments
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
            }

            // Determine exchange and fixed start times
            const isMCX = selectedInstruments.some(s => 
                ['CRUDEOIL', 'NATURALGAS', 'SILVER', 'GOLD'].some(m => s.includes(m))
            );
            
            // Get today's date from the data and normalize to a consistent Date object
            const firstDateStr = rawData[0]?.date || new Date().toISOString();
            const targetDate   = firstDateStr.split(' ')[0];
            
            // Fixed Market Opening (The beginning of the slider)
            const mStartObj = isMCX ? new Date(`${targetDate}T09:00:00`) : new Date(`${targetDate}T09:15:00`);
            
            // Data Bounds
            const allTimes = traces.flatMap(t => t.x).map(x => new Date(x).getTime());
            const maxTime  = allTimes.length > 0 ? Math.max(...allTimes) : null;
            const maxTimeObj = maxTime ? new Date(maxTime) : null;
            
            // Mobile Sliding Window (Initial Zoom: last 2-4 hours)
            const windowHours = isMCX ? 4 : 2;
            const windowMs    = windowHours * 60 * 60 * 1000;
            const windowStart = maxTime ? maxTime - windowMs : null;
            
            // Ensure zoom doesn't go before market start
            const effectiveZoomStart = Math.max(mStartObj.getTime(), windowStart || 0);

            const layout = {
                margin: { t: 30, r: 20, l: 60, b: 60 }, // More padding for bottom slider/legend
                height: 500, // Taller to prevent overlap
                xaxis: { 
                    title: 'Time',
                    range: maxTime ? [new Date(effectiveZoomStart), maxTimeObj] : null,
                    rangeslider: { 
                        visible: true, 
                        thickness: 0.1,
                        range: [mStartObj, maxTimeObj || new Date()] 
                    },
                    type: 'date'
                },
                yaxis: { 
                    title: metric === 'spot_price' ? 'Price' : this._metrics.label(metric),
                    automargin: true 
                },
                hovermode: 'x unified',
                dragmode: 'pan',
                showlegend: true,
                legend: { 
                    orientation: 'h', 
                    yanchor: 'bottom', 
                    y: 1.02, // Move legend ABOVE the chart to avoid slider overlap entirely
                    x: 0 
                }
            };

            const config = {
                responsive: true,
                displayModeBar: false,
                scrollZoom: true
            };

            Plotly.newPlot(chartId, traces, layout, config);
        });
    }
}
