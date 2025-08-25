// EventBus.js - Simple publish-subscribe pattern for decoupled communication
export class EventBus {
  constructor() {
    this.events = {};
  }

  /**
   * Subscribe to an event
   * @param {string} event - Event name
   * @param {Function} callback - Callback function
   * @returns {Function} Unsubscribe function
   */
  on(event, callback) {
    if (!this.events[event]) {
      this.events[event] = [];
    }
    this.events[event].push(callback);
    
    // Return unsubscribe function
    return () => this.off(event, callback);
  }

  /**
   * Unsubscribe from an event
   * @param {string} event - Event name
   * @param {Function} callback - Callback function to remove
   */
  off(event, callback) {
    if (!this.events[event]) return;
    
    const index = this.events[event].indexOf(callback);
    if (index > -1) {
      this.events[event].splice(index, 1);
    }
  }

  /**
   * Emit an event
   * @param {string} event - Event name
   * @param {*} data - Data to pass to callbacks
   */
  emit(event, data) {
    if (!this.events[event]) return;
    
    // Create a copy to avoid issues if callbacks modify the array
    const callbacks = [...this.events[event]];
    callbacks.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error(`Error in event handler for ${event}:`, error);
      }
    });
  }

  /**
   * Subscribe to an event only once
   * @param {string} event - Event name
   * @param {Function} callback - Callback function
   */
  once(event, callback) {
    const wrapper = (data) => {
      callback(data);
      this.off(event, wrapper);
    };
    this.on(event, wrapper);
  }

  /**
   * Clear all event listeners
   */
  clear() {
    this.events = {};
  }
}
