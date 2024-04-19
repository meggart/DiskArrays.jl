
macro implement_show(t)
    t = esc(t)
    quote
        Base.show(io::IO, mime::MIME"text/plain", A::$t) = _show(io, mime, A)
        Base.show(io::IO, A::$t) = _show(io, A) 
    end
end

_show(io, A) = summary(io, A)
function _show(io, _, A)
    summary(io, A)
    println(io)
    println(io)
    print(io, nameof(typeof(haschunks(A))))
    if haschunks(A) isa Chunked
        print(io, ": (\n")
        foreach(eachchunk(A).chunks) do c
            print(io,"    ")
            show(IOContext(io,:compact => true), length.(c))
            print(io,"\n")
        end
        println(io,")")
    end
end
