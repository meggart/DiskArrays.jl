
macro implement_show(t)
    t = esc(t)
    quote
        Base.show(io::IO, mime::MIME"text/plain", A::$t) = _show(io, mime, A)
        Base.show(io::IO, A::$t) = _show(io, A) 
    end
end

_show(io, A) = summary(io, A)
function _show(io, mime, A)
    summary(io, A)
    println(io)
    println(io)
    print(io, nameof(typeof(haschunks(A))))
    if haschunks(A) isa Chunked
        print(io, ": ")
        show(io, mime, eachchunk(A))
    end
end
