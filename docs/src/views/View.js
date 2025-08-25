// View.js - Base class for all views
export class View {
  constructor(element, eventBus) {
    this.element = element;
    this.eventBus = eventBus;
    this.eventHandlers = new Map();
  }

  /**
   * Render the view - to be implemented by subclasses
   */
  render() {
    throw new Error('render() must be implemented by subclass');
  }

  /**
   * Bind DOM events
   */
  bindEvents() {
    // To be overridden by subclasses
  }

  /**
   * Unbind DOM events
   */
  unbindEvents() {
    // Remove all event listeners
    this.eventHandlers.forEach((handler, key) => {
      const [element, event] = key.split(':');
      const el = element === 'this' ? this.element : this.element.querySelector(element);
      if (el) {
        el.removeEventListener(event, handler);
      }
    });
    this.eventHandlers.clear();
  }

  /**
   * Helper to add event listener and track it
   */
  addEventListener(selector, event, handler) {
    const element = selector === 'this' ? this.element : this.element.querySelector(selector);
    if (!element) return;
    
    element.addEventListener(event, handler);
    this.eventHandlers.set(`${selector}:${event}`, handler);
  }

  /**
   * Emit an event through the event bus
   */
  emit(event, data) {
    this.eventBus.emit(event, data);
  }

  /**
   * Clean up the view
   */
  destroy() {
    this.unbindEvents();
    if (this.element && this.element.parentNode) {
      this.element.parentNode.removeChild(this.element);
    }
  }
}
