export const SIZE = 48; // grid side length

const session = await ort.InferenceSession.create("models/mudkip.onnx");
const inputData = new Float32Array(1 * 16 * SIZE * SIZE);
const inputShape = [1, 16, SIZE, SIZE];
const tensor = new ort.Tensor("float32", inputData, inputShape);
const output = await session.run({ x: tensor });
console.log(output);

export class NCA {
  constructor() {
    this.session = session; // assume `session` is already created
    this.inputData = new Float32Array(1 * 16 * SIZE * SIZE);
    const center = SIZE >> 1;
    // seed 4th channel at center to 1.0
    this.inputData[3 * SIZE * SIZE + center * SIZE + center] = 1.0;
    this.tensor = new ort.Tensor("float32", this.inputData, [
      1,
      16,
      SIZE,
      SIZE,
    ]);
    this.texture = new Float32Array(SIZE * SIZE * 3);
    this._updateTexture(this.inputData);
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
    // forward pass – model returns Δstate
    const { tanh: deltaT } = await this.session.run({ x: this.tensor });
    const delta = deltaT.data; // Float32Array of deltas
    const state = this.tensor.data; // current state buffer

    this._updateTexture(state); // refresh RGB preview
    this.tensor = new ort.Tensor("float32", state, [1, 16, SIZE, SIZE]);
  }

  get_board() {
    return this.texture;
  }
}
