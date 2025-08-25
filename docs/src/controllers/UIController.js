// UIController.js - Handles UI interactions and state management
import { Controller } from './Controller.js';

export class UIController extends Controller {
  constructor(eventBus, appStateModel, simulationModel, pokemonConfigModel, 
              pokemonSelectorView, controlPanelView) {
    super(eventBus);
    
    this.appStateModel = appStateModel;
    this.simulationModel = simulationModel;
    this.pokemonConfigModel = pokemonConfigModel;
    this.pokemonSelectorView = pokemonSelectorView;
    this.controlPanelView = controlPanelView;
  }

  /**
   * Initialize the controller
   */
  initialize() {
    super.initialize();
    
    // Initialize views
    this.controlPanelView.initialize();
    
    // Render pokemon selector
    const pokemonList = this.pokemonConfigModel.getPokemonList();
    this.pokemonSelectorView.render(pokemonList);
    
    // Set initial state
    const currentPokemon = this.appStateModel.getSelectedPokemon();
    const currentMode = this.appStateModel.getSelectedMode();
    
    this.pokemonSelectorView.setSelectedPokemon(currentPokemon);
    this.controlPanelView.updateModeSelector(currentMode);
    this.controlPanelView.updateFPSDisplay(this.appStateModel.getFPS());
    this.controlPanelView.updatePauseButton(this.appStateModel.isPaused());
    
    // Update availability based on current mode
    this.updatePokemonAvailability(currentMode);
  }

  /**
   * Bind event listeners
   */
  bindEvents() {
    // Pokemon selector events
    this.subscribe('selector:pokemon-clicked', async (data) => {
      await this.handlePokemonSelection(data.pokemon);
    });
    
    // Control panel events
    this.subscribe('controls:mode-selected', async (data) => {
      await this.handleModeSelection(data.mode);
    });
    
    this.subscribe('controls:reset-clicked', async () => {
      await this.handleReset();
    });
    
    this.subscribe('controls:pause-clicked', () => {
      this.handlePauseToggle();
    });
    
    this.subscribe('controls:fps-changed', (data) => {
      this.handleFPSChange(data.fps);
    });
    
    // State change events
    this.subscribe('state:warning-changed', (data) => {
      if (data.message) {
        this.controlPanelView.showWarning(data.message);
      } else {
        this.controlPanelView.hideWarning();
      }
    });
    
    this.subscribe('state:loading-changed', (data) => {
      if (data.isLoading) {
        this.controlPanelView.disableControls();
      } else {
        this.controlPanelView.enableControls();
      }
    });
    
    this.subscribe('state:pause-toggled', (data) => {
      this.controlPanelView.updatePauseButton(data.isPaused);
    });
  }

  /**
   * Handle pokemon selection
   */
  async handlePokemonSelection(pokemonName) {
    const currentMode = this.appStateModel.getSelectedMode();
    
    // Check if combination is valid
    if (!this.validateAndUpdateSelection(pokemonName, currentMode)) {
      return;
    }
    
    // Update state
    const success = this.appStateModel.setSelectedPokemon(pokemonName);
    if (!success) {
      return;
    }
    
    // Update view
    this.pokemonSelectorView.setSelectedPokemon(pokemonName);
    
    // Load new model
    this.appStateModel.setModelLoading(true);
    try {
      await this.simulationModel.updateModel(pokemonName, currentMode);
    } catch (error) {
      console.error('Failed to load model:', error);
      this.appStateModel.setWarning('Failed to load model');
    } finally {
      this.appStateModel.setModelLoading(false);
    }
  }

  /**
   * Handle mode selection
   */
  async handleModeSelection(mode) {
    const currentPokemon = this.appStateModel.getSelectedPokemon();
    const previousMode = this.appStateModel.getSelectedMode();
    
    // Check if combination is valid
    if (!this.validateAndUpdateSelection(currentPokemon, mode)) {
      // Revert selector to previous value
      this.controlPanelView.revertModeSelector(previousMode);
      return;
    }
    
    // Update state
    const success = this.appStateModel.setSelectedMode(mode);
    if (!success) {
      this.controlPanelView.revertModeSelector(previousMode);
      return;
    }
    
    // Update pokemon availability
    this.updatePokemonAvailability(mode);
    
    // Load new model
    this.appStateModel.setModelLoading(true);
    try {
      await this.simulationModel.updateModel(currentPokemon, mode);
    } catch (error) {
      console.error('Failed to load model:', error);
      this.appStateModel.setWarning('Failed to load model');
      // Revert on error
      this.appStateModel.setSelectedMode(previousMode);
      this.controlPanelView.revertModeSelector(previousMode);
    } finally {
      this.appStateModel.setModelLoading(false);
    }
  }

  /**
   * Handle reset button
   */
  async handleReset() {
    this.appStateModel.setModelLoading(true);
    try {
      await this.simulationModel.reset();
    } catch (error) {
      console.error('Failed to reset:', error);
      this.appStateModel.setWarning('Failed to reset simulation');
    } finally {
      this.appStateModel.setModelLoading(false);
    }
  }

  /**
   * Handle pause toggle
   */
  handlePauseToggle() {
    this.appStateModel.togglePause();
  }

  /**
   * Handle FPS change
   */
  handleFPSChange(fps) {
    this.appStateModel.setFPS(fps);
  }

  /**
   * Validate and update selection
   */
  validateAndUpdateSelection(pokemon, mode) {
    if (!this.pokemonConfigModel.isValidCombination(pokemon, mode)) {
      this.appStateModel.setWarning('This configuration is not available');
      return false;
    }
    
    this.appStateModel.clearWarning();
    return true;
  }

  /**
   * Update pokemon availability based on mode
   */
  updatePokemonAvailability(mode) {
    const pokemonList = this.pokemonConfigModel.getPokemonList();
    const availabilityMap = {};
    
    pokemonList.forEach(pokemon => {
      availabilityMap[pokemon] = this.pokemonConfigModel.isValidCombination(pokemon, mode);
    });
    
    this.pokemonSelectorView.updateAvailability(availabilityMap);
  }
}
