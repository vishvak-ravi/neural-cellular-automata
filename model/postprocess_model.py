import torch
import onnx
from utils import CAGetBoard, to_onnx, init_board

if __name__ == "__main__":
    # load and convert model
    names = ["darkrai"]
    tasks = ["regenerate"]
    for task in tasks:
        for name in names:
            model_name = f"{name}_{task}"
            src_image = f"data/src/medium/{name}.png"
            features, _ = init_board(src_image, 32)
            C, H, W = features.shape

            torch_model = CAGetBoard(state_size=32)
            torch_model.load_state_dict(
                torch.load(
                    f"demo/models/{model_name}.pt", map_location=torch.device("cpu")
                )
            )
            to_onnx(torch_model, model_name, (1, C, max(H, W), max(H, W)))

            # validate onxx
            onnx_model = onnx.load(f"demo/models/{model_name}.onnx")
            onnx.checker.check_model(onnx_model)
