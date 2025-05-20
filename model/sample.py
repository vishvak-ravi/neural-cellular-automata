import onnxruntime, torch, torchvision
import torch.nn.functional as F
from utils import CARule

SIZE = 48

def sample(iters: int):
    set_rule = CARule(dense=False)
    set_rule.load_state_dict(torch.load('mudkip.pt', map_location=torch.device('cpu')))
    
    seed = torch.zeros(1, 16, SIZE, SIZE)
    seed[:, 3:, SIZE//2, SIZE//2] = 1.0
    board = seed.numpy()
    board = torch.tensor(board)
    
    ort_session = onnxruntime.InferenceSession('demo/models/mudkip.onnx', providers=['CPUExecutionProvider'])
    
    for iter in range(iters):
        with torch.no_grad():
            # onnxruntime_input = {input_arg.name: onnx_inputs for input_arg in ort_session.get_inputs()}
            # onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]
            torch_outputs = set_rule(torch.tensor(board))
            
            #print(torch.allclose(torch_outputs, torch.tensor(onnxruntime_outputs), rtol=1e-1))
            
            rand_mask = torch.rand((1, 1, SIZE, SIZE)) < 0.5
            
            pre_alive = (
                (F.max_pool2d(board[:, 3:4, :, :], 3, stride=1, padding=1) > 0.1)
                .int()[:, 0]
                .unsqueeze(1)
            )
            torch_outputs = torch_outputs * rand_mask
            
            board = board + torch_outputs
            post_alive = (
                (F.max_pool2d(board[:, 3:4, :, :], 3, stride=1, padding=1) > 0.1)
                .int()[:, 0]
                .unsqueeze(1)
            )
            board = board * (pre_alive & post_alive)
            board[:, :3].clamp_(0.0, 1.0)
            img = board[0, :3, :, :]
            torchvision.utils.save_image(torch.tensor(img), f"data/demo_test/{iter}.png")

if __name__ == '__main__':
    sample(1000)