module PyNNTraining

using CUDA: CUDA
using Flux: Flux
using FluxTraining: FluxTraining, Callback, ToDevice, Read, Write
using FluxTraining.Events: Event, EpochBegin, StepBegin
using Functors: Functors, fmap
using InlineTest
using Optimisers
using PyCall: PyCall, PyObject
using PyCallChainRules.Torch: torch, TorchModuleWrapper


include("topytorch.jl")


export ToPyTorch

end
