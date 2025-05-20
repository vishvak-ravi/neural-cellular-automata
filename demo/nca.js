export const SIZE = 48; // grid side length
const DESTROY_RADIUS = 3;

function getOffsetsInRadius(radius) {
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
const destroyOffsets = getOffsetsInRadius(DESTROY_RADIUS);

const session = await ort.InferenceSession.create("../data/params/mudkip.onnx");
const inputData = new Float32Array(1 * 16 * SIZE * SIZE);
const inputShape = [1, 16, SIZE, SIZE];
const tensor = new ort.Tensor("float32", inputData, inputShape);
const output = await session.run({ x: tensor });
console.log(output);

export class NCA {
  constructor() {
    this.session = session; // assume `session` is already created
    var inputData = new Float32Array(1 * 16 * SIZE * SIZE);
    const center = SIZE >> 1;
    // seed 4th channel at center to 1.0
    inputData[3 * SIZE * SIZE + center * SIZE + center] = 1.0;
    this.tensor = new ort.Tensor("float32", inputData, [1, 16, SIZE, SIZE]);
    this.texture = new Float32Array(SIZE * SIZE * 3);
    this._updateTexture(inputData);
  }

  _updateTexture(data) {
    // copy channels 0–2 into texture (H×W×3)
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const idx = y * SIZE + x;
        const base = y * SIZE + x;
        const i0 = base; // channel 0
        const i1 = 1 * SIZE * SIZE + base;
        const i2 = 2 * SIZE * SIZE + base;
        const t = idx * 3;
        this.texture[t] = data[i0];
        this.texture[t + 1] = data[i1];
        this.texture[t + 2] = data[i2];
      }
    }
  }

  async step() {
    const C = 16;
    const rawIn = this.tensor.data;
    // flip Y on input
    const flippedIn = new Float32Array(rawIn.length);
    for (let c = 0; c < C; ++c) {
      const off = c * SIZE * SIZE;
      for (let y = 0; y < SIZE; ++y) {
        const fy = SIZE - 1 - y;
        for (let x = 0; x < SIZE; ++x) {
          flippedIn[off + y * SIZE + x] = rawIn[off + fy * SIZE + x];
        }
      }
    }

    // run model on flipped input
    const flippedTensor = new ort.Tensor("float32", flippedIn, [
      1,
      C,
      SIZE,
      SIZE,
    ]);
    const { slice_scatter_1: newState } = await this.session.run({
      x: flippedTensor,
    });
    const rawOut = newState.data;

    // flip Y back on output
    const unflipped = new Float32Array(rawOut.length);
    for (let c = 0; c < C; ++c) {
      const off = c * SIZE * SIZE;
      for (let y = 0; y < SIZE; ++y) {
        const fy = SIZE - 1 - y;
        for (let x = 0; x < SIZE; ++x) {
          unflipped[off + y * SIZE + x] = rawOut[off + fy * SIZE + x];
        }
      }
    }

    this._updateTexture(unflipped);
    this.tensor = new ort.Tensor("float32", unflipped, [1, C, SIZE, SIZE]);
  }

  get_board() {
    return this.texture;
  }

  destroyAt({ x, y }) {
    const data = this.tensor.data; // Float32Array, length 16·SIZE²
    for (const [dx, dy] of destroyOffsets) {
      const ix = x + dx;
      const iy = y + dy;
      if (ix >= 0 && ix < SIZE && iy >= 0 && iy < SIZE) {
        const base = iy * SIZE + ix; // index inside one channel
        for (let c = 0; c < 16; ++c) {
          data[c * SIZE * SIZE + base] = 0; // zero every channel
        }
      }
    }
    this.tensor = new ort.Tensor("float32", data, [1, 16, SIZE, SIZE]);
    this._updateTexture(data);
  }
}
