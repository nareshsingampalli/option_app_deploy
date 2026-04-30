/**
 * Base Component Class - Structural: Composite/Template Pattern
 */
class BaseComponent {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
    }

    render(data) {
        throw new Error("Render method must be implemented");
    }

    showLoading(message = "Loading...") {
        const loader = document.getElementById('loading');
        if (loader) {
            loader.textContent = message;
            loader.style.display = 'flex';
        }
    }

    hideLoading() {
        const loader = document.getElementById('loading');
        if (loader) loader.style.display = 'none';
    }
}

// Alias for components that were using the old name
window.UIComponent = BaseComponent;
