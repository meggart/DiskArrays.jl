macro implement_iteration(t)
quote
function Base.iterate(a::$t)
    cc = eachchunk(a)
    ii = iterate(cc)
    ii === nothing && return nothing
    cifirst, cistate = ii
    datacur = (cistate,a[toRanges(cifirst)...])
    innerinds = eachindex(datacur[2])
    ii = iterate(innerinds)
    ii === nothing && return nothing
    innerfirst, innerstate = ii
    return datacur[2][innerfirst],(datacur,cc,cistate,innerinds,innerstate)
end
function Base.iterate(a::$t,i)
    datacur,cc,cistate,innerinds,innerstate = i
    ii = iterate(innerinds, innerstate)
    if ii===nothing
        cii = iterate(cc,cistate)
        cii === nothing && return nothing
        cinow, cistate = cii
        datacur = (cistate,a[toRanges(cinow)...])
        innerinds = eachindex(datacur[2])
        ii = iterate(innerinds)
        ii === nothing && return nothing
    end
    innerfirst, innerstate = ii
    datacur[1] == cistate || error("Something has messed up this iterator")

    return datacur[2][innerfirst],(datacur,cc,cistate,innerinds,innerstate)
end
end
end
