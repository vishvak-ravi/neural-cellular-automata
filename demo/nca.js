const SIZE = 64; // grid side length
const sobelX = new Float32Array([-1, 0, 1, -2, 0, 2, -1, 0, 1]); // sobel filter for x direction
const sobelY = new Float32Array([-1, -2, -1, 0, 0, 0, 1, 2, 1]); // sobel filter for y direction

const session = await ort.InferenceSession.create("models/mudkip.onnx");
const inputData = new Float32Array(1 * 16 * 32 * 32);
const inputShape = [1, 3, 32, 32];
const tensor = new ort.Tensor("float32", inputData, inputShape);
const output = await session.run({ inputName: tensor });
console.log(output);

function preprocessFloatArray(input) {
  // input is a Float32Array with values between 0 and 1
  // shape is SIZE * SIZE *
  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    output[i] = input[i] / 255.0;
  }
  return output;
}

export function update(b) {
  for (let i = 0; i < b.length; ++i) b[i] = 1.0 - b[i];
}
