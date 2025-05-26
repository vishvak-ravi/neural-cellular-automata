# Neural Cellular Automata

Cellular automata meeting CNNs allow for automata to follow an "attracting" state through a reconstruction loss. The different training schemes give the cells different behaviors, but the "regenerate" task is the most powerful—allowing for the cells to rebuild upon damage.

Original inspiration from [this Distill article](https://distill.pub/2020/growing-ca/) by Mordvintesen et al. which explains the implementation better than I could.

The main differences include modifying the learning rate schedulers and variable state size to allow for training ~75x75 size images (original only allows ~43x43). I still couldn't crack the 90x90+ size as evidenced by the lack of model files for images under `data/src/large`.

Note that a few of these ONNX models don't use the same architecture as the most recent commits. You're better off re-training if you plan to use this repo.

## Demo

You can find a demo (here)[https://vishvak-ravi.github.io/neural-cellular-automata/] implemented with WebGL using the ONNX runtime and vanilla HTML/JS/CSS.
