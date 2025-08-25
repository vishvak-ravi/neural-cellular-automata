// SimulationController.js - Controls simulation loop and interactions
import { Controller } from './Controller.js';

export class SimulationController extends Controller {
  constructor(eventBus, simulationModel, canvasView, appStateModel) {
    super(eventBus);
    
    this.simulationModel = simulationModel;
    this.canvasView = canvasView;
    this.appStateModel = appStateModel;
    
    // Animation state
    this.animationFrameId = null;
    this.lastFrameTime = 0;
    
    // Mouse state for destruction
    this.isDestroying = false;
    this.destructionCoords = null;
  }

  /**
   * Initialize the controller
   */
  initialize() {
    super.initialize();
    
    // Initialize canvas view
    this.canvasView.initialize();
    
    // Start with initial board state
    const boardData = this.simulationModel.getBoardData();
    const size = this.simulationModel.getSize();
    if (boardData && size) {
      this.canvasView.initializeBoard(boardData, size);
    }
    
    // Start simulation loop
    this.startSimulation();
  }

  /**
   * Bind event listeners
   */
  bindEvents() {
    // Canvas mouse events
    this.subscribe('canvas:mouse-down', (data) => {
      this.handleMouseDown(data);
    });
    
    this.subscribe('canvas:mouse-move', (data) => {
      this.handleMouseMove(data);
    });
    
    this.subscribe('canvas:mouse-up', () => {
      this.handleMouseUp();
    });
    
    // State changes
    this.subscribe('state:fps-changed', (data) => {
      // FPS change is handled in the render loop
    });
    
    this.subscribe('state:pause-toggled', (data) => {
      // Pause state is checked in the render loop
    });
    
    // Model updates
    this.subscribe('model:updated', (data) => {
      // Re-initialize board when model changes
      const boardData = this.simulationModel.getBoardData();
      const size = this.simulationModel.getSize();
      if (boardData && size) {
        this.canvasView.initializeBoard(boardData, size);
      }
    });
    
    this.subscribe('model:reset', () => {
      // Re-initialize board after reset
      const boardData = this.simulationModel.getBoardData();
      const size = this.simulationModel.getSize();
      if (boardData && size) {
        this.canvasView.initializeBoard(boardData, size);
      }
    });
  }

  /**
   * Start the simulation loop
   */
  startSimulation() {
    const render = (timestamp) => {
      this.handleFrame(timestamp);
      this.animationFrameId = requestAnimationFrame(render);
    };
    this.animationFrameId = requestAnimationFrame(render);
  }

  /**
   * Stop the simulation loop
   */
  stopSimulation() {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  /**
   * Handle animation frame
   */
  async handleFrame(timestamp) {
    // Apply destruction if mouse is down
    if (this.isDestroying && this.destructionCoords) {
      this.applyDestruction(this.destructionCoords);
    }
    
    // Check if we should step the simulation
    const fps = this.appStateModel.getFPS();
    const isPaused = this.appStateModel.isPaused();
    const frameInterval = 1000 / fps;
    
    if (!isPaused && timestamp - this.lastFrameTime >= frameInterval) {
      // Perform simulation step
      if (this.simulationModel.isReady()) {
        await this.simulationModel.step();
      }
      this.lastFrameTime = timestamp;
    }
    
    // Always render the current board state
    const boardData = this.simulationModel.getBoardData();
    const size = this.simulationModel.getSize();
    if (boardData && size) {
      this.canvasView.renderBoard(boardData, size);
    }
  }

  /**
   * Handle mouse down event
   */
  handleMouseDown(coords) {
    this.isDestroying = true;
    this.destructionCoords = coords;
  }

  /**
   * Handle mouse move event
   */
  handleMouseMove(coords) {
    if (this.isDestroying) {
      this.destructionCoords = coords;
    }
  }

  /**
   * Handle mouse up event
   */
  handleMouseUp() {
    this.isDestroying = false;
    this.destructionCoords = null;
  }

  /**
   * Apply destruction at coordinates
   */
  applyDestruction(coords) {
    if (coords && coords.x !== undefined && coords.y !== undefined) {
      this.simulationModel.destroyAt(coords.x, coords.y);
    }
  }

  /**
   * Clean up the controller
   */
  destroy() {
    this.stopSimulation();
    super.destroy();
  }
}
