// Controller.js - Base class for all controllers
export class Controller {
  constructor(eventBus) {
    this.eventBus = eventBus;
    this.subscriptions = [];
  }

  /**
   * Initialize the controller
   */
  initialize() {
    this.bindEvents();
  }

  /**
   * Bind event listeners - to be implemented by subclasses
   */
  bindEvents() {
    // To be overridden by subclasses
  }

  /**
   * Subscribe to an event and track the subscription
   */
  subscribe(event, handler) {
    const boundHandler = handler.bind(this);
    const unsubscribe = this.eventBus.on(event, boundHandler);
    this.subscriptions.push(unsubscribe);
    return unsubscribe;
  }

  /**
   * Emit an event
   */
  emit(event, data) {
    this.eventBus.emit(event, data);
  }

  /**
   * Clean up the controller
   */
  destroy() {
    // Unsubscribe from all events
    this.subscriptions.forEach(unsubscribe => unsubscribe());
    this.subscriptions = [];
  }
}
