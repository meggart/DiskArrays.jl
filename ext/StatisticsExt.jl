module StatisticsExt
import Statistics
import DiskArrays: AbstractDiskArray, DefaultBackend, get_backend, compute_backend, diskarrays_mean_impl

function Statistics.mean(f::Function, a::AbstractDiskArray; kwargs...)
    diskarrays_mean_impl(f, a, get_backend(compute_backend); kwargs...)
end
Statistics.mean(a::AbstractDiskArray; kwargs...) = Statistics.mean(identity, a; kwargs...)

diskarrays_mean_impl(f, a::AbstractDiskArray, ::DefaultBackend;kwargs...) =
    invoke(Statistics.mean,Tuple{typeof(f),AbstractArray{eltype(a),ndims(a)}},f,a;kwargs...)





end