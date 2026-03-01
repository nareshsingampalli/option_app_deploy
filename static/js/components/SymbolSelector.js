/**
 * SymbolSelector — Exchange toggle + symbol dropdown
 * 
 * Keyboard shortcuts (first letter):
 *   NSE : n = Nifty 50, b = BankNifty, f = FinNifty
 *   MCX : c = CrudeOil,  n = NaturalGas, s = Silver,  g = Gold
 * 
 * Behavioral: Observer — notifies subscribers on change.
 */
class SymbolSelector {

    static SYMBOLS = {
        NSE: [
            { key: 'NIFTY', label: 'Nifty 50', shortcut: 'n' },
            { key: 'BANKNIFTY', label: 'BankNifty', shortcut: 'b' },
            { key: 'FINNIFTY', label: 'FinNifty', shortcut: 'f' },
        ],
        MCX: [
            { key: 'CRUDEOIL', label: 'Crude Oil', shortcut: 'c' },
            { key: 'NATURALGAS', label: 'Natural Gas', shortcut: 'n' },
            { key: 'SILVER', label: 'Silver', shortcut: 's' },
            { key: 'GOLD', label: 'Gold', shortcut: 'g' },
        ]
    };

    constructor(containerId) {
        this._container = document.getElementById(containerId);
        this._exchange = 'NSE';
        this._symbol = 'NIFTY';
        this._observers = [];
        this._render();
        this._bindKeyboard();
    }

    // ── Observer ──────────────────────────────────────────────────────────────
    onChange(fn) { this._observers.push(fn); }

    _notify() {
        this._observers.forEach(fn => fn({ exchange: this._exchange, symbol: this._symbol }));
    }

    // ── Public ────────────────────────────────────────────────────────────────
    get exchange() { return this._exchange; }
    get symbol() { return this._symbol; }

    // ── Render ────────────────────────────────────────────────────────────────
    _render() {
        this._container.innerHTML = `
            <div class="symbol-selector">
                <!-- Exchange toggle buttons -->
                <div class="exchange-toggle" id="exchange-toggle">
                    <button class="exch-btn active" data-exch="NSE" title="Press N/B/F for symbols">
                        NSE
                    </button>
                    <button class="exch-btn" data-exch="MCX" title="Press C/N/S/G for symbols">
                        MCX
                    </button>
                </div>

                <!-- Symbol dropdown -->
                <div class="symbol-dropdown-wrap">
                    <select id="symbol-select" class="symbol-select">
                        ${this._optionsHTML()}
                    </select>
                    <span class="shortcut-hint" id="shortcut-hint"></span>
                </div>
            </div>`;

        // Wire exchange buttons
        this._container.querySelectorAll('.exch-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this._exchange = btn.dataset.exch;
                this._symbol = SymbolSelector.SYMBOLS[this._exchange][0].key;
                this._container.querySelectorAll('.exch-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this._refreshDropdown();
                this._notify();
            });
        });

        // Wire dropdown
        this._container.querySelector('#symbol-select').addEventListener('change', e => {
            this._symbol = e.target.value;
            this._notify();
        });
    }

    _optionsHTML() {
        return SymbolSelector.SYMBOLS[this._exchange].map(s =>
            `<option value="${s.key}" ${s.key === this._symbol ? 'selected' : ''}>
                [${s.shortcut.toUpperCase()}]  ${s.label}
            </option>`
        ).join('');
    }

    _refreshDropdown() {
        const sel = this._container.querySelector('#symbol-select');
        if (sel) sel.innerHTML = this._optionsHTML();
    }

    // ── Keyboard shortcuts ────────────────────────────────────────────────────
    _bindKeyboard() {
        document.addEventListener('keydown', e => {
            // Don't hijack input/select/textarea focus
            const tag = document.activeElement.tagName;
            if (['INPUT', 'SELECT', 'TEXTAREA'].includes(tag)) return;

            const pressed = e.key.toLowerCase();
            const match = SymbolSelector.SYMBOLS[this._exchange]
                .find(s => s.shortcut === pressed);
            if (!match) return;

            this._symbol = match.key;
            const sel = this._container.querySelector('#symbol-select');
            if (sel) sel.value = match.key;

            // Flash hint
            const hint = document.getElementById('shortcut-hint');
            if (hint) {
                hint.textContent = `⌨ ${match.label}`;
                hint.classList.add('visible');
                setTimeout(() => hint.classList.remove('visible'), 1500);
            }
            this._notify();
        });
    }
}
