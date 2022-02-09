using LinearAlgebra
using Distributions
using CairoMakie

function gaussianProcessTarget(duration, ndim, tau; eps=1e-6)
    dists = [min(abs(i-j), abs(duration+j-i), abs(-duration+j-i)) 
             for i=1:duration, j=1:duration]
    cov = exp.(-(dists.^2)./(tau^2))
    return rand(MvNormal(zeros(duration), cov+eps*I), ndim)
end

points = rand(Point3f, 200)
segs = Point3f[]
for i=1:length(points), j=1:length(points)
    if (rand() < 0.2 && i!=j)
        push!(segs, points[i])
        push!(segs, points[j])
    end
end

colors = gaussianProcessTarget(200, 200, 20.0)

start_colors = 1 ./ (1 .+ exp.(-colors[1, :] .+ 2))
#set_theme!(theme_black())
set_theme!(backgroundcolor=(:black, 1.0))
fig, ax, scatterObj = scatter(points, color=start_colors, markersize=5, colormap = :inferno, colorrange=(0.0, 1.0)) #[(:white, rand()) for i=1:200]
linesegments!(segs, linewidth=0.5, color=(:gray, 0.05))
fig

im = Makie.colorbuffer(fig.scene)# Makie.GLNative)
blur = imfilter(im .* (Gray.(im) .> 0.4), Kernel.gaussian(2))
clamp01.(im + 2*blur)

r = LinRange(0,1,10) .- .5
dop = [exp(-sqrt(x*x+y*y+z*z)) for x=r, y=r, z=r]
#record(fig, "dummy_animation.mp4", 1:200;
#        framerate = 20) do t
#    scatterObj.color = 1 ./ (1 .+ exp.(-colors[t, :] .+ 2))
#end
include("makie_bloom.jl")
record(fig, "dummy_filtered.mp4"; framerate=20) do io
    for t = 1:200
        scatterObj.color = 1 ./ (1 .+ exp.(-colors[t, :] .+ 2))
        recordframe_withbloom!(io)
    end
end