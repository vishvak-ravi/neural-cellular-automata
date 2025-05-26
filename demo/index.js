// index.js
import { NCA } from "./nca.js";

("use strict");
const pokemon = [
  "bulbasaur",
  "pikachu",
  "cyndaquil",
  "mudkip",
  "mewtwo",
  "arceus",
  "darkrai",
];
const unsupportedCombos = [
  "mewtwo_grow",
  "mewtwo_persist",
  "darkrai_grow",
  "darkrai_persist",
  "arceus_grow",
  "arceus_persist",
];
let initModel = "pikachu"; // default model
let initMode = "grow"; // default mode
const warningDiv = document.querySelector(".warning"); // CSS: .warning { display:none; }
warningDiv.style.display = "none";
let nca = new NCA(initModel, initMode);
document.getElementById(initModel).classList.add("subject-selected");
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
  nca.SIZE,
  nca.SIZE,
  0,
  gl.RGB,
  gl.FLOAT,
  nca.get_board()
);
gl.uniform1i(gl.getUniformLocation(prog, "u_board"), 0);

/*========================  MOUSE STATE  ========================*/
let mouseDown = false;
let mouseClient = { x: 0, y: 0 };

canvas.addEventListener("mousedown", (e) => {
  mouseDown = true;
  mouseClient.x = e.clientX;
  mouseClient.y = e.clientY;
});

canvas.addEventListener("mousemove", (e) => {
  if (mouseDown) {
    mouseClient.x = e.clientX;
    mouseClient.y = e.clientY;
  }
});

window.addEventListener("mouseup", () => (mouseDown = false));

/*========================  DESTRUCTION  ========================*/
function getDestructionCenterTexCoords() {
  if (!mouseDown) return null;

  const rect = canvas.getBoundingClientRect();
  const localX = mouseClient.x - rect.left;
  const localY = mouseClient.y - rect.top;

  if (localX < 0 || localY < 0 || localX > rect.width || localY > rect.height)
    return null; // cursor outside canvas

  const texX = Math.floor((localX / rect.width) * nca.SIZE);
  // WebGL (0,0) is bottom-left; DOM (0,0) is top-left → flip Y
  const texY = Math.floor(((rect.height - localY) / rect.height) * nca.SIZE);

  return { x: texX, y: texY };
}

/*========================  SETTINGS  ========================*/
var pause = false;

const resetHandler = {
  handleClick: function () {
    nca.reset();
  },
};

document
  .getElementById("resetButton")
  .addEventListener("click", () => resetHandler.handleClick());

const pauseHandler = {
  handleClick: function () {
    pause = !pause;
  },
};

document
  .getElementById("pauseButton")
  .addEventListener("click", () => pauseHandler.handleClick());

document
  .getElementById("modelSelector")
  .addEventListener("change", async (e) => {
    const next = e.target.value;
    const combo = `${nca.name}_${next}`;

    if (unsupportedCombos.includes(combo)) {
      warningDiv.style.display = "block";
      e.target.value = nca.mode; // revert to previous choice
      return;
    }

    warningDiv.style.display = "none";
    await nca.updateModel(nca.name, next); // updateModel sets nca.mode internally
  });

for (let pokemonName of pokemon) {
  document
    .getElementById(pokemonName)
    .addEventListener("click", async function () {
      const combo = pokemonName + "_" + nca.mode; // e.g. "charizard_xgrow"
      if (unsupportedCombos.includes(combo)) {
        warningDiv.style.display = "block";
        return; // do not change model
      } else {
        warningDiv.style.display = "none";
        await nca.updateModel(pokemonName, nca.mode);
        gl.texImage2D(
          gl.TEXTURE_2D,
          0,
          gl.RGB32F,
          nca.SIZE,
          nca.SIZE,
          0,
          gl.RGB,
          gl.FLOAT,
          nca.get_board()
        );
      }
      // Update CSS classes
      for (let other of pokemon) {
        document.getElementById(other).classList.remove("subject-selected");
      }
      this.classList.add("subject-selected");
      // toggle warning
    });
}

/*========================  GAME LOOP  ========================*/
nca.step(); // initial update

const slider = document.getElementById("speed");
const fpsLbl = document.getElementById("fpsVal");
let fps = +slider.value;

slider.oninput = () => {
  fps = +slider.value;
  fpsLbl.textContent = fps;
};

let last = 0;
function render(now) {
  // handle destruction
  let destructionCenterCoords = getDestructionCenterTexCoords();
  if (destructionCenterCoords != null) {
    nca.destroyAt(destructionCenterCoords);
    // nca.destroyAt(destructionCenterCoords);
  }

  if (now - last >= 1000 / fps && !pause) {
    // throttle by chosen FPS
    nca.step();
    last = now;
  }
  let board = nca.get_board();
  gl.texSubImage2D(
    gl.TEXTURE_2D,
    0,
    0,
    0,
    nca.SIZE,
    nca.SIZE,
    gl.RGB,
    gl.FLOAT,
    board
  );
  gl.drawArrays(gl.TRIANGLES, 0, 6);

  requestAnimationFrame(render);
}
requestAnimationFrame(render);

render();
