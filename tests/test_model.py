import torch
from game_model.AI_model.model import SplendidSplendorModel

def test_model_setup():
    model = SplendidSplendorModel(11,6,100,5)

    in_tensor = torch.Tensor([0.5,
                              0.0,1.0,0.0,
                              0.0,1.0,2.0,3.0,4.0,5.0,6.0])
    
    model.forward(in_tensor)
