/**
 * ApiService - Behavioral: Strategy Pattern
 * Encapsulates all data fetching logic.
 */
class ApiService {
    constructor() {
        this.baseUrl = '/api';
    }

    async getOptionData(params) {
        const query = new URLSearchParams(params).toString();
        const response = await fetch(`${this.baseUrl}/option-data?${query}`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return await response.json();
    }

    async getMarketStatus(exchange) {
        const response = await fetch(`${this.baseUrl}/market-status?exchange=${exchange}`);
        return await response.json();
    }

    /**
     * Probes live spot price for a symbol.
     * Returns { is_holiday: bool, spot_price: float|null, reason: str }
     * Used to confirm market is truly open (not a holiday) before enabling live mode.
     */
    async spotProbe(exchange, symbol) {
        const response = await fetch(`${this.baseUrl}/spot-probe?exchange=${exchange}&symbol=${symbol}`);
        if (!response.ok) return { is_holiday: false, spot_price: null };
        return await response.json();
    }

    /**
     * Called on page load to determine what date/mode to use.
     * Returns { use_historical: bool, date: 'YYYY-MM-DD', reason: str }
     */
    async getPreMarketStatus(exchange) {
        const response = await fetch(`${this.baseUrl}/pre-market-status?exchange=${exchange}`);
        if (!response.ok) return { use_historical: false, date: null };
        return await response.json();
    }

    async refreshToken() {
        const response = await fetch(`${this.baseUrl}/refresh-token`, { method: 'POST' });
        return await response.json();
    }
}



