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

    async refreshToken() {
        const response = await fetch(`${this.baseUrl}/refresh-token`, { method: 'POST' });
        return await response.json();
    }
}


