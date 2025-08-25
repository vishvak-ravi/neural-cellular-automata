// ControlPanelView.js - Handles control panel UI
import { View } from './View.js';

export class ControlPanelView extends View {
  constructor(element, eventBus) {
    super(element, eventBus);
    
    // Cache DOM elements
    this.modeSelector = null;
    this.speedSlider = null;
    this.fpsLabel = null;
    this.pauseButton = null;
    this.resetButton = null;
    this.warningDiv = null;
  }

  /**
   * Initialize the control panel
   */
  initialize() {
    // Cache DOM elements
    this.modeSelector = this.element.querySelector('#modelSelector');
    this.speedSlider = this.element.querySelector('#speed');
    this.fpsLabel = this.element.querySelector('#fpsVal');
    this.pauseButton = this.element.querySelector('#pauseButton');
    this.resetButton = this.element.querySelector('#resetButton');
    this.warningDiv = this.element.querySelector('.warning');
    
    // Hide warning initially
    if (this.warningDiv) {
      this.warningDiv.style.display = 'none';
    }
    
    // Bind events
    this.bindEvents();
  }

  /**
   * Bind control events
   */
  bindEvents() {
    // Mode selector
    if (this.modeSelector) {
      this.addEventListener('#modelSelector', 'change', (e) => {
        const mode = e.target.value;
        this.emit('controls:mode-selected', { mode });
      });
    }
    
    // Speed slider
    if (this.speedSlider) {
      this.addEventListener('#speed', 'input', (e) => {
        const fps = parseInt(e.target.value);
        this.updateFPSDisplay(fps);
        this.emit('controls:fps-changed', { fps });
      });
    }
    
    // Pause button
    if (this.pauseButton) {
      this.addEventListener('#pauseButton', 'click', () => {
        this.emit('controls:pause-clicked');
      });
    }
    
    // Reset button
    if (this.resetButton) {
      this.addEventListener('#resetButton', 'click', () => {
        this.emit('controls:reset-clicked');
      });
    }
  }

  /**
   * Update FPS display
   */
  updateFPSDisplay(fps) {
    if (this.fpsLabel) {
      this.fpsLabel.textContent = fps;
    }
    if (this.speedSlider) {
      this.speedSlider.value = fps;
    }
  }

  /**
   * Update pause button text
   */
  updatePauseButton(isPaused) {
    if (this.pauseButton) {
      this.pauseButton.textContent = isPaused ? 'Resume' : 'Pause';
    }
  }

  /**
   * Update mode selector
   */
  updateModeSelector(mode, availableModes = null) {
    if (!this.modeSelector) return;
    
    // Update selected value
    this.modeSelector.value = mode;
    
    // Update available options if provided
    if (availableModes) {
      const currentOptions = Array.from(this.modeSelector.options).map(opt => opt.value);
      
      // Disable/enable options based on availability
      this.modeSelector.options.forEach(option => {
        option.disabled = !availableModes.includes(option.value);
      });
    }
  }

  /**
   * Show warning message
   */
  showWarning(message) {
    if (this.warningDiv) {
      const textElement = this.warningDiv.querySelector('b');
      if (textElement) {
        textElement.textContent = message || 'This configuration is not available';
      }
      this.warningDiv.style.display = 'block';
    }
  }

  /**
   * Hide warning message
   */
  hideWarning() {
    if (this.warningDiv) {
      this.warningDiv.style.display = 'none';
    }
  }

  /**
   * Enable all controls
   */
  enableControls() {
    if (this.modeSelector) this.modeSelector.disabled = false;
    if (this.speedSlider) this.speedSlider.disabled = false;
    if (this.pauseButton) this.pauseButton.disabled = false;
    if (this.resetButton) this.resetButton.disabled = false;
  }

  /**
   * Disable all controls (e.g., during loading)
   */
  disableControls() {
    if (this.modeSelector) this.modeSelector.disabled = true;
    if (this.speedSlider) this.speedSlider.disabled = true;
    if (this.pauseButton) this.pauseButton.disabled = true;
    if (this.resetButton) this.resetButton.disabled = true;
  }

  /**
   * Revert mode selector to previous value
   */
  revertModeSelector(previousMode) {
    if (this.modeSelector) {
      this.modeSelector.value = previousMode;
    }
  }
}
