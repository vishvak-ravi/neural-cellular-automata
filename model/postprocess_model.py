import torch
import onnx
from utils import CAGetBoard, to_onnx

if __name__ == "__main__":
    # load and convert model
    model_name = "mudkip"
    torch_model = CAGetBoard()
    torch_model.load_state_dict(
        torch.load(f"data/params/{model_name}.pt", map_location=torch.device("cpu"))
    )
    to_onnx(torch_model)

    # validate onxx
    onnx_model = onnx.load(f"data/params/{model_name}.onnx")
    onnx.checker.check_model(onnx_model)
