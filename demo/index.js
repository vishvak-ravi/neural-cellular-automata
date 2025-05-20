// index.js
import { update, SIZE } from "./nca.js";

("use strict");
let board = new Float32Array(SIZE * SIZE * 3); // RGB per cell
board[((SIZE >> 1) * SIZE + (SIZE >> 1)) * 3] = 1.0; // center-pixel = white
board[((SIZE >> 1) * SIZE + (SIZE >> 1)) * 3 + 1] = 1.0; // center-pixel = white
board[((SIZE >> 1) * SIZE + (SIZE >> 1)) * 3 + 2] = 1.0; // center-pixel = white

/*========================  SHADERS  ========================*/
const vs = `#version 300 es
in vec2 a_position;
out vec2 v_uv;
void main() {
  v_uv = a_position * 0.5 + 0.5;              // clip → uv
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

const fs = `#version 300 es
precision highp float;
uniform sampler2D u_board;
in vec2 v_uv;
out vec4 outColor;
void main() {
  outColor = vec4(texture(u_board, v_uv).rgb, 1.0);
}`;

/*========================  GL SETUP  ========================*/
const canvas = document.querySelector("#c");
const gl = canvas.getContext("webgl2");
if (!gl) throw new Error("WebGL2 required");

function sh(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw gl.getShaderInfoLog(s);
  }
  return s;
}
const prog = gl.createProgram();
gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs));
gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs));
gl.linkProgram(prog);
if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
  throw gl.getProgramInfoLog(prog);
gl.useProgram(prog);

/* full-screen quad */
const quad = new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]);
const vbo = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
const loc = gl.getAttribLocation(prog, "a_position");
gl.enableVertexAttribArray(loc);
gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

/*========================  TEXTURE  ========================*/
const tex = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, tex);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
gl.texImage2D(
  gl.TEXTURE_2D,
  0,
  gl.RGB32F,
  SIZE,
  SIZE,
  0,
  gl.RGB,
  gl.FLOAT,
  board
);
gl.uniform1i(gl.getUniformLocation(prog, "u_board"), 0);

/*========================  GAME LOOP  ========================*/
update(board); // initial update

const slider = document.getElementById("speed");
const fpsLbl = document.getElementById("fpsVal");
let fps = +slider.value;

slider.oninput = () => {
  fps = +slider.value;
  fpsLbl.textContent = fps;
};

let last = 0;
function render(now) {
  if (now - last >= 1000 / fps) {
    // throttle by chosen FPS
    update(board);
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      0,
      SIZE,
      SIZE,
      gl.RGB,
      gl.FLOAT,
      board
    );
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    last = now;
  }
  requestAnimationFrame(render);
}
requestAnimationFrame(render);

render();
