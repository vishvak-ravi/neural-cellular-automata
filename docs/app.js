// app.js - Application bootstrap
import { EventBus } from './src/utils/EventBus.js';
import { Constants } from './src/utils/Constants.js';

// Models
import { PokemonConfigModel } from './src/models/PokemonConfigModel.js';
import { AppStateModel } from './src/models/AppStateModel.js';
import { SimulationModel } from './src/models/SimulationModel.js';

// Views
import { CanvasView } from './src/views/CanvasView.js';
import { PokemonSelectorView } from './src/views/PokemonSelectorView.js';
import { ControlPanelView } from './src/views/ControlPanelView.js';

// Controllers
import { SimulationController } from './src/controllers/SimulationController.js';
import { UIController } from './src/controllers/UIController.js';

class NeuralCellularAutomataApp {
  constructor() {
    this.eventBus = new EventBus();
    this.models = {};
    this.views = {};
    this.controllers = {};
  }

  /**
   * Initialize the application
   */
  async initialize() {
    try {
      // Create models
      this.models.pokemonConfig = new PokemonConfigModel();
      this.models.appState = new AppStateModel(this.eventBus, this.models.pokemonConfig);
      this.models.simulation = new SimulationModel(this.eventBus, this.models.pokemonConfig);
      
      // Create views
      const canvas = document.querySelector('#c');
      const pokemonSelector = document.querySelector('.subject-selection');
      const controlPanel = document.querySelector('.settings-parent');
      
      this.views.canvas = new CanvasView(canvas, this.eventBus);
      this.views.pokemonSelector = new PokemonSelectorView(pokemonSelector, this.eventBus);
      this.views.controlPanel = new ControlPanelView(controlPanel, this.eventBus);
      
      // Initialize simulation model with defaults
      const defaultPokemon = Constants.DEFAULT_POKEMON;
      const defaultMode = Constants.DEFAULT_MODE;
      await this.models.simulation.updateModel(defaultPokemon, defaultMode);
      
      // Perform initial step to populate board
      await this.models.simulation.step();
      
      // Create controllers
      this.controllers.simulation = new SimulationController(
        this.eventBus,
        this.models.simulation,
        this.views.canvas,
        this.models.appState
      );
      
      this.controllers.ui = new UIController(
        this.eventBus,
        this.models.appState,
        this.models.simulation,
        this.models.pokemonConfig,
        this.views.pokemonSelector,
        this.views.controlPanel
      );
      
      // Initialize controllers
      this.controllers.simulation.initialize();
      this.controllers.ui.initialize();
      
      console.log('Neural Cellular Automata App initialized successfully');
    } catch (error) {
      console.error('Failed to initialize app:', error);
      throw error;
    }
  }

  /**
   * Clean up the application
   */
  destroy() {
    // Destroy controllers
    Object.values(this.controllers).forEach(controller => {
      if (controller.destroy) controller.destroy();
    });
    
    // Destroy views
    Object.values(this.views).forEach(view => {
      if (view.destroy) view.destroy();
    });
    
    // Clear event bus
    this.eventBus.clear();
    
    console.log('Neural Cellular Automata App destroyed');
  }
}

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', async () => {
    window.ncaApp = new NeuralCellularAutomataApp();
    await window.ncaApp.initialize();
  });
} else {
  // DOM is already loaded
  window.ncaApp = new NeuralCellularAutomataApp();
  window.ncaApp.initialize();
}
