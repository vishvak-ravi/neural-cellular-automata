import torch
import onnx
from utils import CARule, to_onnx

if __name__ == "__main__":
    # load and convert model
    model_name = "mudkip"
    torch_model = CARule(dense=False)
    torch_model.load_state_dict(
        torch.load(f"data/params/{model_name}.pt", map_location=torch.device("cpu"))
    )
    to_onnx(torch_model)

    # validate onxx
    onnx_model = onnx.load(f"data/params/{model_name}.onxx")
    onnx.checker.check_model(onnx_model)
