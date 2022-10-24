# PyNNTraining.jl

[![Build Status](https://github.com/username/PyNNTraining.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/username/PyNNTraining.jl/actions/workflows/CI.yml?query=branch%3Amain)

PyNNTraining.jl is an extension to FluxTraining.jl that allows you to train PyTorch models, without boilerplate, and compatible with the existing Julia ecosystem.

## How to use

- Use PyCall.jl to load a PyTorch model
- Create the `ToPyTorch` callback
- Pass both to FluxTraining.jl's `Learner`
- Train as usual

## Full training example

Here we use a pretrained model from [torchvision]() and finetune it. We leave the data loading and preprocessing to [FastAI.jl]():

```julia
# Load a PyTorch model

using FastAI, FastVision, PyNNTraining, FluxTraining, PyCall, Optimisers
torch, torchvision, funcexp = pyimport("torch"), pyimport("torchvision"), pyimport("functorch.experimental")

function loadresnet(c::Int)
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=false)
    funcexp.replace_all_batch_norm_modules_(model)
    model.fc = torch.nn.Linear(model.fc.in_features, c) # Set no. of classes
    return model
end

model = loadresnet(10)
callback = ToPyTorch()  # uses "cuda" device if available

data, blocks = load(datarecipes()["imagenette2-320"])
task = ImageClassificationSingle(blocks)
learner = tasklearner(task, data; callbacks=[callback, Metrics(accuracy)], model=model,
                      optimizer = Optimisers.Adam(), batchsize = 16)

fitonecycle!(learner, 10, 0.001)
```

## Acknowledgements

PyNNTraining.jl is a wrapper around [PyCallChainRules.jl]() which handles all the heavy lifting of integrating with Python ADs and sharing array memories.

## Limitations

- PyCall.jl doesn't free memory frequently enough when running many allocating operations from Julia. To circumvent this, the `ToPyTorch` manually frees the memory during training. The downside of this is that the maximum memory allocation is a bit lower than what your GPU supports
- Only PyTorch models that are compatible with [functorch]() can be used. In many cases, this will mean that you have to adapt a model's batch normalization layers as described [on this page in the functorch docs]().
- Only explicit optimisers from Optimisers.jl are supported, implicit parameters do not work.