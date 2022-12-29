import torch
from game_model.AI_model.model import SplendidSplendorModel
from game_model.AI_model.action_output import ActionOutput

def test_model_setup():

    input_dict = {'in1':[1],'in2':[3],'in3':[7]}
    output_dict = {'out1':[1,0,0],'out2':[5,0,0]}

    model = SplendidSplendorModel(input_dict,output_dict,100,5)

    in1_tensor = torch.Tensor([0.5])
    in2_tensor = torch.Tensor([0.0,1.0,0.0])
    in3_tensor = torch.Tensor([0.0,1.0,2.0,3.0,4.0,5.0,6.0])
    input_tensor_dict = {'in1':in1_tensor,'in2':in2_tensor,'in3':in3_tensor}
    

    model.forward(input_tensor_dict)
