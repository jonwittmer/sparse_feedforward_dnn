import os
import torch
from torch import nn
import numpy as np
import json


# There are a lot of assumptions that go into being able to serialize a given module.
# In particular, assume that the input is an instance of nn.Sequential
def saveSparseModel(sequentialModel, savePath):
    os.makedirs(savePath, exist_ok=True)
    
    moduleList = [m for m in sequentialModel]
    layers = []
    ind = 0
    for module in moduleList:
        # process activation functions - only relu or elu for now
        if isinstance(module, nn.ReLU):
            layers[-1]["activation"] = "relu"
            continue
        elif isinstance(module, nn.ELU):
            layers[-1]["activation"] = "elu"
            continue

        # convert layer to useful format for saving
        weights, biases, dim = serializeSparseLayer(module)

        # add info to layers list
        layers.append({
            "weights": f"weights_{ind}.csv",
            "biases": f"biases_{ind}.csv",
            "activation": "none",
            "dimension" : dim
        })

        # save weights and biases to file
        np.savetxt(os.path.join(savePath, f"weights_{ind}"), weights, delimiter=",")
        np.savetxt(os.path.join(savePath, f"biases_{ind}"), biases, delimiter=",")
        
        ind += 1

    with open(os.path.join(savePath, "config.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=4)
    

def serializeSparseLayer(layer):
    weightsTensor = layer.weight
    biasesTensor = layer.bias
    dim = weightsTensor.shape

    # convert to COO sparse format 
    weights = weightsTensor.to_sparse()
    weightsNp = weights.cpu().indices().detach().numpy().T
    weightsNp = np.hstack((weightsNp, weights.cpu().values().detach().numpy().reshape((-1, 1))))
    print(weightsNp.shape)

    # turn into row vector for writing to CSV in format 1,2,3,4...
    biasesNp = biasesTensor.cpu().detach().numpy().reshape((1, -1))
        
    return weightsNp, biasesNp, dim
