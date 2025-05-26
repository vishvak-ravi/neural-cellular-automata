const DESTROY_RADIUS = 5;
const NAME_TO_SIZE = {
  mudkip: 67,
  cyndaquil: 68,
  bulbasaur: 61,
  pikachu: 73,
  mewtwo: 94,
  darkrai: 103,
  arceus: 108,
};

const NAME_TO_CHAN_COUNT = {
  mudkip: 16,
  cyndaquil: 16,
  bulbasaur: 16,
  pikachu: 16,
  mewtwo: 32,
  darkrai: 32,
  arceus: 32,
};

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

export class NCA {
  constructor(name, mode) {
    this.updateModel(name, mode);
  }

  async init_session() {
    try {
      this.session = await ort.InferenceSession.create(
        `models/${this.name}_${this.mode}.onnx`
      );
    } catch (err) {
      if (/404|not.*found/i.test(String(err))) {
        this.session = null; // file missing → null
      } else {
        throw err; // propagate other issues
      }
    }
  }
  _updateTexture(data) {
    // copy channels 0–2 into texture (H×W×3)
    for (let y = 0; y < this.SIZE; y++) {
      for (let x = 0; x < this.SIZE; x++) {
        const idx = y * this.SIZE + x;
        const base = y * this.SIZE + x;
        const i0 = base; // channel 0
        const i1 = 1 * this.SIZE * this.SIZE + base;
        const i2 = 2 * this.SIZE * this.SIZE + base;
        const t = idx * 3;
        this.texture[t] = data[i0];
        this.texture[t + 1] = data[i1];
        this.texture[t + 2] = data[i2];
      }
    }
  }

  async step() {
    if (this.session == null) {
      return;
    }
    const C = this.CHAN_COUNT;
    const rawIn = this.tensor.data;
    // flip Y on input
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

    // run model on flipped input
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

    // flip Y back on output
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

    this._updateTexture(unflipped);
    this.tensor = new ort.Tensor("float32", unflipped, [
      1,
      C,
      this.SIZE,
      this.SIZE,
    ]);
  }

  get_board() {
    return this.texture;
  }

  destroyAt({ x, y }) {
    const data = this.tensor.data; // Float32Array, length 16·SIZE²
    for (const [dx, dy] of destroyOffsets) {
      const ix = x + dx;
      const iy = y + dy;
      if (ix >= 0 && ix < this.SIZE && iy >= 0 && iy < this.SIZE) {
        const base = iy * this.SIZE + ix; // index inside one channel
        for (let c = 0; c < this.CHAN_COUNT; ++c) {
          data[c * this.SIZE * this.SIZE + base] = 0; // zero every channel
        }
      }
    }
    this.tensor = new ort.Tensor("float32", data, [
      1,
      this.CHAN_COUNT,
      this.SIZE,
      this.SIZE,
    ]);
    this._updateTexture(data);
  }

  updateModel(name = null, mode = null) {
    if (name != null) {
      this.name = name;
      this.SIZE = NAME_TO_SIZE[this.name];
      this.CHAN_COUNT = NAME_TO_CHAN_COUNT[this.name];
    }
    if (mode != null) {
      this.mode = mode;
    }
    this.session = null;
    this.tensor = new ort.Tensor(
      "float32",
      new Float32Array(1 * this.CHAN_COUNT * this.SIZE * this.SIZE),
      [1, this.CHAN_COUNT, this.SIZE, this.SIZE]
    );
    const inputData = new Float32Array(
      1 * this.CHAN_COUNT * this.SIZE * this.SIZE
    );
    const center = this.SIZE >> 1;
    // seed 4th channel at center to 1.0
    inputData[3 * this.SIZE * this.SIZE + center * this.SIZE + center] = 1.0;
    this.tensor = new ort.Tensor("float32", inputData, [
      1,
      this.CHAN_COUNT,
      this.SIZE,
      this.SIZE,
    ]);
    this.texture = new Float32Array(this.SIZE * this.SIZE * 3);
    this._updateTexture(inputData);
    this.init_session();
  }
  reset() {
    this.updateModel(this.name, this.mode);
  }
}
