import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import json
import copy
import dis
import triton
import triton.language as tl
import marshal


@torch.compile
def complex_1(pixel_grid_tensor):
    local_grid = pixel_grid_tensor.clone()
    local_grid = (local_grid**2)/5
    local_grid = torch.clamp(local_grid, min=0, max=9)
    return local_grid
    
def generate_challenge(height,width):
        input =  torch.randint(low=0,high=9,size=(height,width))
        
        function_list = [complex_1]


        random_function = random.choice(function_list)

        return input.detach().cpu().numpy().tolist() , random_function(input).detach().cpu().numpy().tolist()
    
def generate_symbolic_dataset(name,num_examples=100):
        
        num_train_examples = 2
        num_test_examples = 1
        training_dataset = {}
        training_solutions_dataset = {}

        for n in range(num_examples):
            example_name = ""
            for i in range(8):
                example_name += str(random.randint(0,9))
            train = []
            test = []
            solutions = []
            
            for i in range(num_train_examples):
                input ,output =  generate_challenge(random.randint(5,8),random.randint(5,8))
                train.append({"input":input, "output":output})
            
            for i in range(num_test_examples):
                input ,output =  generate_challenge(random.randint(5,8),random.randint(5,8))
                test.append({"input":input})
                solutions.append(output)
            training_dataset[example_name] = {"train":train,"test":test}
            training_solutions_dataset[example_name] = solutions
        with open(name+'_challenges.json', 'w') as f:
           json.dump(training_dataset, f)
        with open(name+'_solutions.json', 'w') as f:
           json.dump(training_solutions_dataset, f)

def generate_dataset():
    generate_symbolic_dataset("arc-agi_trainingSym")
    generate_symbolic_dataset("arc-agi_evaluationSym")
    generate_symbolic_dataset("arc-agi_conceptSym")
if __name__ == "__main__":
    generate_dataset()
