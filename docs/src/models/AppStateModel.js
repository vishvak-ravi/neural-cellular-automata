// AppStateModel.js - Manages application UI state
import { Constants } from '../utils/Constants.js';

export class AppStateModel {
  constructor(eventBus, pokemonConfigModel) {
    this.eventBus = eventBus;
    this.pokemonConfigModel = pokemonConfigModel;
    
    // Initialize state
    this.state = {
      isPaused: false,
      fps: Constants.DEFAULT_FPS,
      selectedPokemon: Constants.DEFAULT_POKEMON,
      selectedMode: Constants.DEFAULT_MODE,
      isModelLoading: false,
      warningMessage: null
    };
  }

  /**
   * Get current state
   */
  getState() {
    return { ...this.state };
  }

  /**
   * Set FPS
   */
  setFPS(value) {
    if (value < 1 || value > 120) return false;
    
    const oldValue = this.state.fps;
    this.state.fps = value;
    this.eventBus.emit('state:fps-changed', { oldValue, newValue: value });
    return true;
  }

  /**
   * Get FPS
   */
  getFPS() {
    return this.state.fps;
  }

  /**
   * Set pause state
   */
  setPaused(value) {
    const oldValue = this.state.isPaused;
    this.state.isPaused = value;
    this.eventBus.emit('state:pause-toggled', { isPaused: value, oldValue });
    return true;
  }

  /**
   * Toggle pause state
   */
  togglePause() {
    return this.setPaused(!this.state.isPaused);
  }

  /**
   * Check if paused
   */
  isPaused() {
    return this.state.isPaused;
  }

  /**
   * Set selected pokemon
   */
  setSelectedPokemon(name) {
    if (!this.pokemonConfigModel.getPokemonConfig(name)) {
      return false;
    }
    
    // Check if combination is valid
    if (!this.canSelectCombination(name, this.state.selectedMode)) {
      this.setWarning('This configuration is not available');
      return false;
    }
    
    const oldValue = this.state.selectedPokemon;
    this.state.selectedPokemon = name;
    this.clearWarning();
    this.eventBus.emit('state:selection-changed', {
      pokemon: name,
      mode: this.state.selectedMode,
      oldPokemon: oldValue
    });
    return true;
  }

  /**
   * Get selected pokemon
   */
  getSelectedPokemon() {
    return this.state.selectedPokemon;
  }

  /**
   * Set selected mode
   */
  setSelectedMode(mode) {
    if (!this.pokemonConfigModel.getModes().includes(mode)) {
      return false;
    }
    
    // Check if combination is valid
    if (!this.canSelectCombination(this.state.selectedPokemon, mode)) {
      this.setWarning('This configuration is not available');
      return false;
    }
    
    const oldValue = this.state.selectedMode;
    this.state.selectedMode = mode;
    this.clearWarning();
    this.eventBus.emit('state:selection-changed', {
      pokemon: this.state.selectedPokemon,
      mode: mode,
      oldMode: oldValue
    });
    return true;
  }

  /**
   * Get selected mode
   */
  getSelectedMode() {
    return this.state.selectedMode;
  }

  /**
   * Check if a pokemon-mode combination can be selected
   */
  canSelectCombination(pokemon, mode) {
    return this.pokemonConfigModel.isValidCombination(pokemon, mode);
  }

  /**
   * Set model loading state
   */
  setModelLoading(value) {
    this.state.isModelLoading = value;
    this.eventBus.emit('state:loading-changed', { isLoading: value });
  }

  /**
   * Check if model is loading
   */
  isModelLoading() {
    return this.state.isModelLoading;
  }

  /**
   * Set warning message
   */
  setWarning(message) {
    this.state.warningMessage = message;
    this.eventBus.emit('state:warning-changed', { message });
  }

  /**
   * Clear warning message
   */
  clearWarning() {
    if (this.state.warningMessage) {
      this.state.warningMessage = null;
      this.eventBus.emit('state:warning-changed', { message: null });
    }
  }

  /**
   * Get warning message
   */
  getWarning() {
    return this.state.warningMessage;
  }
}
