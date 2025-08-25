// WebGLManager.js - Encapsulates WebGL operations
export class WebGLManager {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext("webgl2");
    if (!this.gl) {
      throw new Error("WebGL2 required");
    }
    this.resources = {
      shaders: [],
      programs: [],
      buffers: [],
      textures: []
    };
  }

  /**
   * Create a shader
   */
  createShader(type, source) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`Shader compilation failed: ${info}`);
    }
    
    this.resources.shaders.push(shader);
    return shader;
  }

  /**
   * Create a program from vertex and fragment shaders
   */
  createProgram(vertexSource, fragmentSource) {
    const gl = this.gl;
    const vertexShader = this.createShader(gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = this.createShader(gl.FRAGMENT_SHADER, fragmentSource);
    
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(program);
      gl.deleteProgram(program);
      throw new Error(`Program linking failed: ${info}`);
    }
    
    this.resources.programs.push(program);
    return program;
  }

  /**
   * Create a texture
   */
  createTexture(options = {}) {
    const gl = this.gl;
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    
    // Set texture parameters
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, options.minFilter || gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, options.magFilter || gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, options.wrapS || gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, options.wrapT || gl.CLAMP_TO_EDGE);
    
    this.resources.textures.push(texture);
    return texture;
  }

  /**
   * Update texture data
   */
  updateTexture(texture, data, width, height, useSubImage = false) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    
    if (useSubImage) {
      gl.texSubImage2D(
        gl.TEXTURE_2D,
        0,
        0,
        0,
        width,
        height,
        gl.RGB,
        gl.FLOAT,
        data
      );
    } else {
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGB32F,
        width,
        height,
        0,
        gl.RGB,
        gl.FLOAT,
        data
      );
    }
  }

  /**
   * Create a buffer for a full-screen quad
   */
  createQuadBuffer() {
    const gl = this.gl;
    const positions = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
      -1,  1,
       1, -1,
       1,  1
    ]);
    
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    
    this.resources.buffers.push(buffer);
    return buffer;
  }

  /**
   * Set up vertex attributes
   */
  setupVertexAttribute(program, attributeName, buffer, size, type, normalized, stride, offset) {
    const gl = this.gl;
    const location = gl.getAttribLocation(program, attributeName);
    if (location === -1) return;
    
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, size, type, normalized, stride, offset);
  }

  /**
   * Clean up all WebGL resources
   */
  cleanup() {
    const gl = this.gl;
    
    this.resources.textures.forEach(texture => gl.deleteTexture(texture));
    this.resources.buffers.forEach(buffer => gl.deleteBuffer(buffer));
    this.resources.programs.forEach(program => gl.deleteProgram(program));
    this.resources.shaders.forEach(shader => gl.deleteShader(shader));
    
    this.resources = {
      shaders: [],
      programs: [],
      buffers: [],
      textures: []
    };
  }
}
