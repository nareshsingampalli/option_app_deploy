/**
 * ChartRenderer — Structural: Composite Pattern
 * Renders Plotly charts for each selected metric × instrument.
 */
class ChartRenderer extends UIComponent {
    constructor(containerId, metricSelector) {
        super(containerId);
        this._metrics = metricSelector;
    }

    /**
     * Returns { color, dash } for a given instrument symbol.
     * Delegates to InstrumentColorService so the chart line color
     * is identical to the sidebar name text and dot circle.
     */
    _colorFor(symbol) {
        const color = (window.InstrumentColorService && InstrumentColorService.get(symbol))
            || (symbol.includes('CE') ? 'hsla(130,65%,35%,0.85)' : 'hsla(0,65%,40%,0.85)');
        return { color, dash: 'solid' };
    }

    clear() {
        if (this.container) {
            const selectedMetrics = this._metrics.selected();
            if (selectedMetrics.length === 0) {
                this.container.innerHTML = '';
                return;
            }
            this.container.innerHTML = selectedMetrics.map(metric => `
                <div class="chart-container">
                    <div class="chart-header">${this._metrics.label(metric)}</div>
                    <div class="skeleton-chart"></div>
                </div>
            `).join('');
        }
    }

    render(rawData, selectedInstruments, referenceSpotPrice = null) {
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

        // Determine the reference spot price for ATM/ITM labels
        let useSpotPrice = referenceSpotPrice;
        if (useSpotPrice === null) {
            for (let i = sortedRaw.length - 1; i >= 0; i--) {
                if (sortedRaw[i].spot_price) {
                    useSpotPrice = sortedRaw[i].spot_price;
                    break;
                }
            }
        }

        let atmStrike = null;
        if (useSpotPrice !== null) {
            const allStrikes = [...new Set(rawData.map(r => parseFloat(r.strike)).filter(s => !isNaN(s)))];
            if (allStrikes.length > 0) {
                atmStrike = allStrikes.reduce((prev, curr) => 
                    Math.abs(curr - useSpotPrice) < Math.abs(prev - useSpotPrice) ? curr : prev
                );
            }
        }

        // Group data by symbol and determine clean labels
        const grouped = {};
        const symbolLabels = {};
        rawData.forEach(row => {
            if (selectedInstruments.includes(row.symbol)) {
                if (!grouped[row.symbol]) {
                    grouped[row.symbol] = { rows: [] };

                    const baseMatch = row.symbol.match(/^[A-Z]+/);
                    const baseSym = baseMatch ? baseMatch[0] : '';
                    const strikeStr = row.strike || '';
                    const strikeVal = parseFloat(strikeStr);
                    
                    let moneyNess = '';
                    if (atmStrike !== null && !isNaN(strikeVal)) {
                        const isCE = row.symbol.includes('CE');
                        const isPE = row.symbol.includes('PE');
                        
                        if (strikeVal === atmStrike) {
                            moneyNess = ' (A)';
                        } else if (isCE) {
                            moneyNess = strikeVal < atmStrike ? ' (I)' : ' (O)';
                        } else if (isPE) {
                            moneyNess = strikeVal > atmStrike ? ' (I)' : ' (O)';
                        }
                    }

                    // Generate clean label: NAME + STRIKE + MONEYNESS
                    symbolLabels[row.symbol] = `${baseSym} ${strikeStr}${moneyNess}`.trim();
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
            let chartAnnotations = [];
            traces.forEach(t => {
                const filtered = t.x.map((x, i) => ({ x, y: t.y[i] }))
                    .filter(pt => pt.x.startsWith(targetDay));
                t.x = filtered.map(p => p.x);
                t.y = filtered.map(p => p.y);
                if (t.name !== 'Spot Price' && t.x.length > 0) {
                    // Shorten the on-chart text and apply specific colors ONLY to the moneyness indicator.
                    const moneyColor = t.isCE ? '#26a641' : '#df3333';
                    const lineColor = t.line.color;
                    
                    // One label per 30-min slot, placed at the MID-POINT of each slot
                    let lastLabeledSlot = -1;
                    t.x.forEach((dtStr, idx) => {
                        const dtObj = new Date(dtStr);
                        if (!isNaN(dtObj.getTime())) {
                            const hr  = dtObj.getHours();
                            const mn  = dtObj.getMinutes();
                            // 30-min slot id (0,1,2,... across the day)
                            const slot     = hr * 2 + (mn >= 30 ? 1 : 0);
                            const mnInSlot = mn % 30;
                            const slotParity = t.isCE ? 0 : 1;
                            const isTargetSlot = (slot % 2) === slotParity;
                            const isMid = mnInSlot >= 12 && mnInSlot <= 18;

                            if (slot !== lastLabeledSlot && isTargetSlot && isMid) {
                                // Extract strike and moneyness indicator from t.name (e.g., 'CRUDEOIL 8500 (A)')
                                // Result: 8500 in line color, (A) in green/red
                                const parts = t.name.split(' ');
                                let strikePart = parts[parts.length - 2];
                                let moneyPart = parts[parts.length - 1];
                                
                                // Fallback if name is differently formatted
                                if (!moneyPart.includes('(')) {
                                    strikePart = parts[parts.length - 1];
                                    moneyPart = '';
                                }

                                let finalHtml = t.name;
                                if (moneyPart && /^\([\d.]+\)$/.test(strikePart.replace(/[^\d.]/g, ''))) {
                                    finalHtml = `<span style="color:${lineColor}">${strikePart}</span> <span style="color:${moneyColor}">${moneyPart}</span>`;
                                } else {
                                    // Robust fallback: just color any (A/I/O) in the string
                                    finalHtml = t.name.replace(/^[A-Z0-9 ]+\s+([\d.]+)/, `<span style="color:${lineColor}">$1</span>`);
                                    finalHtml = finalHtml.replace(/(\((A|I|O)\))/, `<span style="color:${moneyColor}">$1</span>`);
                                }

                                chartAnnotations.push({
                                    x: dtStr,
                                    y: t.y[idx],
                                    text: finalHtml,
                                    showarrow: false,
                                    font: { size: 10, weight: 'bold' },
                                    yshift: 10
                                });
                                lastLabeledSlot = slot;
                            }
                        }
                    });
                }
            });

            // Calculate Bounds
            const allTimes = traces.flatMap(t => t.x).map(x => parseDate(x).getTime()).sort((a, b) => a - b);
            const minTime = allTimes.length > 0 ? allTimes[0] : Date.now();
            const maxTime = allTimes.length > 0 ? allTimes[allTimes.length - 1] : Date.now();
            
            const mStartObj = new Date(minTime);
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
                annotations: chartAnnotations,
                margin: { t: 40, r: 30, l: 60, b: 50 },
                height: 550,
                xaxis: {
                    range: [new Date(viewStart), maxTimeObj],
                    minallowed: mStartObj,
                    maxallowed: maxTimeObj,
                    autorange: false,
                    automargin: true,
                    // rangeslider restored per user request.
                    // Note: to zoom Y-axis on mobile, pinch directly on the Y-axis numbers/labels
                    rangeslider: { visible: true, thickness: 0.12, range: [mStartObj, maxTimeObj] },
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
                    // Use SI prefix for Spot Price or any large values (e.g. 25343 -> 25.343k)
                    tickformat: (metric === 'spot_price' || (yRange && yRange[1] > 1000)) ? '.5~s' : '.2f',
                    hoverformat: '.2f'
                },
                hovermode: 'closest',
                dragmode: 'pan',
                showlegend: false
            };

            const config = {
                responsive: true,
                displayModeBar: true, // Shows zoom/pan/autoscale buttons
                scrollZoom: true,
                editable: false, // Disables the "Click to enter" placeholders
                doubleClick: 'reset+autosize' // Double-tap to reset view
            };

            const existingDiv = document.getElementById(chartId);
            if (existingDiv && existingDiv.data) {
                // Preserve user's zoom/pan state on live updates
                Plotly.react(chartId, traces, layout, config);
            } else {
                // First render — full init
                Plotly.newPlot(chartId, traces, layout, config);
            }
        });
    }
}
