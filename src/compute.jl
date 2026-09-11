using Preferences

abstract type ComputeBackend end

#Fallback: get_backend returns the backend itself
get_backend(b::ComputeBackend) = b

# Default Backend, simple nonthreaded, non-distributed computations
struct DefaultBackend <: ComputeBackend end

# Wrapper type that lets users switch backends dynamically as the program runs
# TODO add user interface for dynamic backend switching
mutable struct DynamicBackend <: ComputeBackend
    current_backend::ComputeBackend
end
get_backend(b::DynamicBackend) = b.current_backend

const backend = @load_preference("backend", "default")

function set_backend(new_backend::String)
    if !(new_backend in ("default", "dynamic", "DiskArrayEngine"))
        throw(ArgumentError("Invalid backend: \"$(new_backend)\""))
    end

    # Set it in our runtime values, as well as saving it to disk
    @set_preferences!("backend" => new_backend)
    @info("New backend set; restart your Julia session for this change to take effect!")
end

function load_backend()
    @static if backend == "default"
        DefaultBackend()
    elseif backend == "dynamic"
        return DynamicBackend(DefaultBackend())
    elseif backend == "DiskArrayEngine"
        DAE = Base.get_extension(@__MODULE__, :DiskArrayEngineExt)
        isnothing(DAE) && error("Please add DiskArrayEngine to your environment to set it as a backend")
        return DAE.DiskArrayEngineBackend()
    else
        return nothing
    end
end
const compute_backend = load_backend()

