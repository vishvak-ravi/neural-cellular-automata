// SimulationModel.js - Manages NCA simulation state and logic
import { Constants } from '../utils/Constants.js';

export class SimulationModel {
  constructor(eventBus, pokemonConfigModel) {
    this.eventBus = eventBus;
    this.pokemonConfigModel = pokemonConfigModel;
    
    // Simulation state
    this.name = null;
    this.mode = null;
    this.SIZE = 0;
    this.CHAN_COUNT = 0;
    this.session = null;
    this.tensor = null;
    this.boardData = null; // RGB data for rendering
    
    // Destruction offsets
    this.destroyOffsets = this.getOffsetsInRadius(Constants.DESTROY_RADIUS);
  }

  /**
   * Calculate offsets for destruction radius
   */
  getOffsetsInRadius(radius) {
    let result = [];
    for (let y = -radius; y <= radius; y++) {
      for (let x = -radius; x <= radius; x++) {
        if (x * x + y * y <= radius * radius) {
          result.push([x, y]);
        }
      }
    }
    return result;
  }

  /**
   * Initialize or update the model
   */
  async updateModel(name, mode) {
    // Get pokemon configuration
    const config = this.pokemonConfigModel.getPokemonConfig(name);
    if (!config) {
      throw new Error(`Unknown pokemon: ${name}`);
    }
    
    // Check if combination is valid
    if (!this.pokemonConfigModel.isValidCombination(name, mode)) {
      throw new Error(`Unsupported combination: ${name}_${mode}`);
    }
    
    // Update configuration
    this.name = name;
    this.mode = mode;
    this.SIZE = config.size;
    this.CHAN_COUNT = config.channelCount;
    
    // Initialize tensor with seed
    const inputData = new Float32Array(1 * this.CHAN_COUNT * this.SIZE * this.SIZE);
    const center = this.SIZE >> 1;
    // Seed 4th channel at center to 1.0
    inputData[3 * this.SIZE * this.SIZE + center * this.SIZE + center] = 1.0;
    
    this.tensor = new ort.Tensor("float32", inputData, [
      1,
      this.CHAN_COUNT,
      this.SIZE,
      this.SIZE,
    ]);
    
    // Initialize board data (RGB)
    this.boardData = new Float32Array(this.SIZE * this.SIZE * 3);
    this._extractRGBFromTensor(inputData);
    
    // Load ONNX model
    await this.initSession();
    
    // Emit update event
    this.eventBus.emit('model:updated', {
      pokemon: name,
      mode: mode,
      size: this.SIZE
    });
  }

  /**
   * Initialize ONNX session
   */
  async initSession() {
    try {
      this.session = await ort.InferenceSession.create(
        `models/${this.name}_${this.mode}.onnx`
      );
    } catch (err) {
      if (/404|not.*found/i.test(String(err))) {
        this.session = null; // File missing
        console.warn(`Model file not found: ${this.name}_${this.mode}.onnx`);
      } else {
        throw err;
      }
    }
  }

  /**
   * Extract RGB channels from tensor data
   */
  _extractRGBFromTensor(data) {
    // Copy channels 0-2 into board data (H×W×3)
    for (let y = 0; y < this.SIZE; y++) {
      for (let x = 0; x < this.SIZE; x++) {
        const idx = y * this.SIZE + x;
        const base = y * this.SIZE + x;
        const i0 = base; // channel 0
        const i1 = 1 * this.SIZE * this.SIZE + base; // channel 1
        const i2 = 2 * this.SIZE * this.SIZE + base; // channel 2
        const t = idx * 3;
        this.boardData[t] = data[i0];
        this.boardData[t + 1] = data[i1];
        this.boardData[t + 2] = data[i2];
      }
    }
  }

  /**
   * Perform one simulation step
   */
  async step() {
    if (!this.session) {
      return false;
    }
    
    const C = this.CHAN_COUNT;
    const rawIn = this.tensor.data;
    
    // Flip Y on input (WebGL coordinate system)
    const flippedIn = new Float32Array(rawIn.length);
    for (let c = 0; c < C; ++c) {
      const off = c * this.SIZE * this.SIZE;
      for (let y = 0; y < this.SIZE; ++y) {
        const fy = this.SIZE - 1 - y;
        for (let x = 0; x < this.SIZE; ++x) {
          flippedIn[off + y * this.SIZE + x] = rawIn[off + fy * this.SIZE + x];
        }
      }
    }

    // Run model on flipped input
    const flippedTensor = new ort.Tensor("float32", flippedIn, [
      1,
      C,
      this.SIZE,
      this.SIZE,
    ]);
    
    const { slice_scatter_1: newState } = await this.session.run({
      x: flippedTensor,
    });
    const rawOut = newState.cpuData;

    // Flip Y back on output
    const unflipped = new Float32Array(rawOut.length);
    for (let c = 0; c < C; ++c) {
      const off = c * this.SIZE * this.SIZE;
      for (let y = 0; y < this.SIZE; ++y) {
        const fy = this.SIZE - 1 - y;
        for (let x = 0; x < this.SIZE; ++x) {
          unflipped[off + y * this.SIZE + x] = rawOut[off + fy * this.SIZE + x];
        }
      }
    }

    // Update tensor
    this.tensor = new ort.Tensor("float32", unflipped, [
      1,
      C,
      this.SIZE,
      this.SIZE,
    ]);
    
    // Update board data
    this._extractRGBFromTensor(unflipped);
    
    // Emit step complete event
    this.eventBus.emit('model:step-complete', {
      boardData: this.boardData
    });
    
    return true;
  }

  /**
   * Apply destruction at specified coordinates
   */
  destroyAt(x, y) {
    const data = this.tensor.data;
    
    for (const [dx, dy] of this.destroyOffsets) {
      const ix = x + dx;
      const iy = y + dy;
      if (ix >= 0 && ix < this.SIZE && iy >= 0 && iy < this.SIZE) {
        const base = iy * this.SIZE + ix;
        for (let c = 0; c < this.CHAN_COUNT; ++c) {
          data[c * this.SIZE * this.SIZE + base] = 0;
        }
      }
    }
    
    // Update tensor
    this.tensor = new ort.Tensor("float32", data, [
      1,
      this.CHAN_COUNT,
      this.SIZE,
      this.SIZE,
    ]);
    
    // Update board data
    this._extractRGBFromTensor(data);
    
    // Emit destruction event
    this.eventBus.emit('model:destroyed', { x, y });
  }

  /**
   * Reset the simulation
   */
  async reset() {
    if (this.name && this.mode) {
      await this.updateModel(this.name, this.mode);
      this.eventBus.emit('model:reset', {
        pokemon: this.name,
        mode: this.mode
      });
    }
  }

  /**
   * Get current board data (RGB)
   */
  getBoardData() {
    return this.boardData;
  }

  /**
   * Get current size
   */
  getSize() {
    return this.SIZE;
  }

  /**
   * Get current pokemon name
   */
  getCurrentPokemon() {
    return this.name;
  }

  /**
   * Get current mode
   */
  getCurrentMode() {
    return this.mode;
  }

  /**
   * Check if model is ready
   */
  isReady() {
    return this.session !== null;
  }
}
