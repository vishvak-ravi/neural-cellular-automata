// Constants.js - Centralized configuration
export const Constants = {
  // Pokemon configuration
  POKEMON: [
    "bulbasaur",
    "pikachu",
    "cyndaquil",
    "mudkip",
    "mewtwo",
    "arceus",
    "darkrai",
  ],
  
  POKEMON_SIZES: {
    mudkip: 67,
    cyndaquil: 68,
    bulbasaur: 61,
    pikachu: 73,
    mewtwo: 94,
    darkrai: 103,
    arceus: 108,
  },
  
  POKEMON_CHANNELS: {
    mudkip: 16,
    cyndaquil: 16,
    bulbasaur: 16,
    pikachu: 16,
    mewtwo: 32,
    darkrai: 32,
    arceus: 32,
  },
  
  UNSUPPORTED_COMBOS: [
    "mewtwo_grow",
    "mewtwo_persist",
    "darkrai_grow",
    "darkrai_persist",
    "arceus_grow",
    "arceus_persist",
  ],
  
  // Simulation modes
  MODES: ["grow", "persist", "regenerate"],
  
  // Default settings
  DEFAULT_POKEMON: "pikachu",
  DEFAULT_MODE: "grow",
  DEFAULT_FPS: 60,
  
  // Simulation parameters
  DESTROY_RADIUS: 5,
  
  // WebGL settings
  CANVAS_WIDTH: 512,
  CANVAS_HEIGHT: 512,
  
  // Shader sources
  VERTEX_SHADER: `#version 300 es
in vec2 a_position;
out vec2 v_uv;
void main() {
  v_uv = a_position * 0.5 + 0.5;              // clip → uv
  gl_Position = vec4(a_position, 0.0, 1.0);
}`,
  
  FRAGMENT_SHADER: `#version 300 es
precision highp float;
uniform sampler2D u_board;
in vec2 v_uv;
out vec4 outColor;
void main() {
  outColor = vec4(texture(u_board, v_uv).rgb, 1.0);
}`,
};
