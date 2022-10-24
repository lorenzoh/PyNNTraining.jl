

struct ToPyTorch <: FluxTraining.Callback
    device::PyObject
    todevice::ToDevice
end


function ToPyTorch(device = default_torch_device())
    return ToPyTorch(torch.device(device), ToDevice(totorch, totorch))
end

Base.show(io::IO, cb::ToPyTorch) = print(io, "ToPyTorch(\"", cb.device.type, "\")")


FluxTraining.resolveconflict(::FluxTraining.Scheduler, to::ToPyTorch) = FluxTraining.RunFirst(to)

#=
function FluxTraining.init!()
    TODO: check that Optimisers.jl is used
    =#

function FluxTraining.on(e::EpochBegin, p, cb::ToPyTorch, learner)
    FluxTraining.on(e, p, cb.todevice, learner)
end


function FluxTraining.on(e::StepBegin, p, cb::ToPyTorch, learner)
    FluxTraining.on(e, p, cb.todevice, learner)

    if cb.device.type == "cuda"
        memusage = memoryusage()
        if memusage > 0.9
            gc_pytorch(memusage > 0.95)
        end
    end
end


FluxTraining.stateaccess(::ToPyTorch) = (
    model = Write(),
    params = Write(),
    step = Write(),
    optimizer = Read(),
)


function default_torch_device()
    if torch.cuda.is_available()
        return torch.device("cuda")
    else
        return torch.device("cpu")
    end
end

_ismodule(m::TorchModuleWrapper) = true
_ismodule(m::PyObject) = PyCall.builtin.isinstance(m, torch.nn.Module)
_ismodule(_) = false

function totorch(x; device = default_torch_device())
    fmap(x; exclude = y -> Flux._isleaf(y) || _ismodule(y)) do leaf
        if leaf isa PyObject
            leaf = TorchModuleWrapper(leaf)
        end
        if device.type == "cuda"
            Flux.gpu(leaf)
        else
            Flux.cpu(leaf)
        end
    end
end


function memoryusage()
    info = CUDA.MemoryInfo()
    return 1 - (info.free_bytes / info.total_bytes)
end

function gc_pytorch(full = false)
    GC.gc(full)
    torch.cuda.empty_cache()
end
