function apply_filter(frame)
    blur = imfilter(frame .* (Gray.(frame) .> 0.4), Kernel.gaussian(2))
    clamp01.(frame + 2*blur)
end

function recordframe_withbloom!(io::Makie.VideoStream)
    cb = Makie.colorbuffer(io.screen, Makie.GLNative)
    frame = convert(Matrix{Makie.RGB{Makie.N0f8}}, apply_filter(cb))
    _xdim, _ydim = size(frame)
    if isodd(_xdim) || isodd(_ydim)
        xdim = iseven(_xdim) ? _xdim : _xdim + 1
        ydim = iseven(_ydim) ? _ydim : _ydim + 1
        padded = fill(zero(eltype(frame)), (xdim, ydim))
        padded[1:_xdim, 1:_ydim] = frame
        frame = padded
    end
    write(io.io, frame)
    return
end