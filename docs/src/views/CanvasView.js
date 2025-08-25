// CanvasView.js - Handles WebGL rendering and canvas interactions
import { View } from './View.js';
import { WebGLManager } from '../utils/WebGLManager.js';
import { Constants } from '../utils/Constants.js';

export class CanvasView extends View {
  constructor(canvas, eventBus) {
    super(canvas, eventBus);
    
    this.canvas = canvas;
    this.webGLManager = new WebGLManager(canvas);
    this.gl = this.webGLManager.gl;
    
    // WebGL resources
    this.program = null;
    this.texture = null;
    this.quadBuffer = null;
    
    // Mouse state for interaction
    this.mouseDown = false;
    this.mouseClient = { x: 0, y: 0 };
    
    // Current board size
    this.boardSize = 0;
  }

  /**
   * Initialize WebGL rendering
   */
  initialize() {
    // Create shader program
    this.program = this.webGLManager.createProgram(
      Constants.VERTEX_SHADER,
      Constants.FRAGMENT_SHADER
    );
    this.gl.useProgram(this.program);
    
    // Create quad buffer
    this.quadBuffer = this.webGLManager.createQuadBuffer();
    
    // Set up vertex attributes
    this.webGLManager.setupVertexAttribute(
      this.program,
      'a_position',
      this.quadBuffer,
      2,        // size
      this.gl.FLOAT,
      false,    // normalized
      0,        // stride
      0         // offset
    );
    
    // Create texture
    this.texture = this.webGLManager.createTexture({
      minFilter: this.gl.NEAREST,
      magFilter: this.gl.NEAREST
    });
    
    // Set texture uniform
    const textureLocation = this.gl.getUniformLocation(this.program, 'u_board');
    this.gl.uniform1i(textureLocation, 0);
    
    // Bind events
    this.bindEvents();
  }

  /**
   * Bind canvas events
   */
  bindEvents() {
    // Mouse down
    this.addEventListener('this', 'mousedown', (e) => {
      this.mouseDown = true;
      this.mouseClient.x = e.clientX;
      this.mouseClient.y = e.clientY;
      
      const coords = this.getTextureCoordinates();
      if (coords) {
        this.emit('canvas:mouse-down', coords);
      }
    });
    
    // Mouse move
    this.addEventListener('this', 'mousemove', (e) => {
      if (this.mouseDown) {
        this.mouseClient.x = e.clientX;
        this.mouseClient.y = e.clientY;
        
        const coords = this.getTextureCoordinates();
        if (coords) {
          this.emit('canvas:mouse-move', coords);
        }
      }
    });
    
    // Mouse up (on window to catch mouse up outside canvas)
    const handleMouseUp = () => {
      if (this.mouseDown) {
        this.mouseDown = false;
        this.emit('canvas:mouse-up');
      }
    };
    window.addEventListener('mouseup', handleMouseUp);
    
    // Store for cleanup
    this.windowMouseUpHandler = handleMouseUp;
  }

  /**
   * Get texture coordinates from mouse position
   */
  getTextureCoordinates() {
    if (!this.mouseDown || !this.boardSize) return null;
    
    const rect = this.canvas.getBoundingClientRect();
    const localX = this.mouseClient.x - rect.left;
    const localY = this.mouseClient.y - rect.top;
    
    // Check if cursor is outside canvas
    if (localX < 0 || localY < 0 || localX > rect.width || localY > rect.height) {
      return null;
    }
    
    // Convert to texture coordinates
    const texX = Math.floor((localX / rect.width) * this.boardSize);
    // WebGL (0,0) is bottom-left; DOM (0,0) is top-left → flip Y
    const texY = Math.floor(((rect.height - localY) / rect.height) * this.boardSize);
    
    return { x: texX, y: texY };
  }

  /**
   * Render the board
   */
  renderBoard(boardData, size) {
    if (!boardData || !size) return;
    
    this.boardSize = size;
    
    // Update texture with board data
    this.webGLManager.updateTexture(
      this.texture,
      boardData,
      size,
      size,
      true // Use texSubImage2D for updates
    );
    
    // Draw quad
    this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
  }

  /**
   * Initialize board texture
   */
  initializeBoard(boardData, size) {
    if (!boardData || !size) return;
    
    this.boardSize = size;
    
    // Initialize texture with board data
    this.webGLManager.updateTexture(
      this.texture,
      boardData,
      size,
      size,
      false // Use texImage2D for initialization
    );
    
    // Draw quad
    this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
  }

  /**
   * Clean up resources
   */
  destroy() {
    // Remove window event listener
    if (this.windowMouseUpHandler) {
      window.removeEventListener('mouseup', this.windowMouseUpHandler);
    }
    
    // Clean up WebGL resources
    this.webGLManager.cleanup();
    
    // Call parent destroy
    super.destroy();
  }
}
