import torch
import onnx
from utils import CAGetBoard, to_onnx, init_board

if __name__ == "__main__":
    # load and convert model
    names = ["bulbasaur", "cyndaquil", "mudkip", "pikachu"]
    tasks = ["grow", "persist", "regenerate"]
    for task in tasks:
        for name in names:
            model_name = f"{name}_{task}"
            src_image = f"data/src/small/{name}.png"
            features, _ = init_board(src_image, 16)
            C, H, W = features.shape
            features = features.view(1, C, H, W)

            torch_model = CAGetBoard()
            torch_model.load_state_dict(
                torch.load(
                    f"data/params/{model_name}.pt", map_location=torch.device("cpu")
                )
            )
            to_onnx(torch_model, model_name, features.shape)

            # validate onxx
            onnx_model = onnx.load(f"data/params/{model_name}.onnx")
            onnx.checker.check_model(onnx_model)
