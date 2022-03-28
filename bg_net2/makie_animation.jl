using LinearAlgebra
using Distributions
using GLMakie
using Colors
using Images

include("experiment.jl")

function recordframe_withfilter!(filter::Function, io::Makie.VideoStream)
    cb = Makie.colorbuffer(io.screen, Makie.GLNative)
    frame = convert(Matrix{Makie.RGB{Makie.N0f8}}, filter(cb))
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

function get_positions(net::BgNet)
    positions = Dict{Symbol, Vector{Point3f}}()
    positions[:str_dmsn] = [Point3f(net.str_dmsn_pos[i, :]) for i=1:size(net[:str_dmsn])]
    positions[:str_imsn] = [Point3f(net.str_imsn_pos[i, :]) for i=1:size(net[:str_imsn])]
    positions[:ctx_exc] = [Point3f(rand(), rand(), 1.5 + rand()) for i=1:size(net[:ctx_exc])]
    positions[:ctx_inh] = [Point3f(rand(), rand(), 1.5 + rand()) for i=1:size(net[:ctx_inh])]
    positions[:thal] = [Point3f([-0.5, i/size(net[:thal]), 1.0]) for i=1:size(net[:thal])]
    positions[:snr] = [Point3f([0.5, 1/10+i/5, -1.0]) for i=1:size(net[:snr])]
    positions[:snc] = [Point3f([-0.5, 1/10+i/5, -0.75]) for i=1:size(net[:snr])]
    return positions
end

function plot_labels!()
    rot = Quaternion(0.5, 0.5, 0.5, 0.5)
    text!("GPi/SNr", position=Point3f([0.8, 0.7, -1]), space=:data,
          textsize=0.15, rotation=rot, color=:gray35, align=(:center, :center))
    text!("Striatum", position=Point3f([1.5, 0.7, 0.75]), space=:data,
          textsize=0.15, rotation=rot, color=:gray35, align=(:center, :center))
    text!("Cortex", position=Point3f([1.5, 0.7, 2.25]), space=:data,
          textsize=0.15, rotation=rot, color=:gray35, align=(:center, :center))
    text!("Thalamus", position=Point3f([-0.5, 0.5, 1.25]), space=:data,
          textsize=0.15, rotation=rot, color=:gray35, align=(:center, :center))
    text!("SNc", position=Point3f([-0.2, 0.7, -0.2]), space=:data,
          textsize=0.15, rotation=rot, color=:gray35, align=(:center, :center))
end

function getDopamineConc(r, varico_pos, varico_act; lam=0.015)
    dop = zeros(length(r), length(r), length(r))
    for x_i=1:length(r), y_i=1:length(r), z_i=1:length(r)
        x = r[x_i]
        y = r[y_i]
        z = r[z_i]
        for i=1:size(varico_pos,1), j=1:size(varico_pos,2)
            dx = x .- varico_pos[i,j][1]
            dy = y .- varico_pos[i,j][2]
            dz = z .- varico_pos[i,j][3]
            dop[x_i,y_i,z_i] += varico_act[i]*exp(-sqrt((dx*dx + dy*dy + dz*dz)/lam))
        end
    end
    return dop
end

function animate_activity(filename::String, net::BgNet, log, target; framerate=20)
    positions = get_positions(net)
    connections = Point3f[]
    colors = String[]
    for pre_pop=pop_order(net), post_pop=pop_order(net)
        projs = net[post_pop].projections
        if net[pre_pop] in keys(projs)
            proj = projs[net[pre_pop]]
            for post=1:length(proj)
                for syn in proj[post]
                    push!(connections, positions[pre_pop][syn.pre])
                    push!(connections, positions[post_pop][post])
                    if pre_pop == :ctx_exc || pre_pop == :thal || (pre_pop == :str_imsn && post_pop == :snr)
                        push!(colors, "hsla(0, 20%, 40%, 0.05)")
                    else
                        push!(colors, "hsla(240, 20%, 40%, 0.05)")
                    end
                end
            end
        end
    end
    snc_size = size(net[:snr])
    varico_pos = rand(Point3f, snc_size, 10)
    for pre=1:size(net[:snr]), post=1:10
        push!(connections, positions[:snc][pre])
        push!(connections, varico_pos[pre, post])
        push!(colors, "hsla(120, 20%, 40%, 0.5)")
    end
    for i=1:size(net[:snr])
        push!(connections, positions[:snc][i])
        push!(connections, positions[:snr][i])
        push!(colors, "hsla(120, 20%, 40%, 0.5)")
    end
    
    scatters = Dict{Symbol, Makie.Scatter}()
    set_theme!(theme_black(), backgroundcolor=(:black, 1.0))
    #fig = Figure(resolution=(800, 1000))
    #axs = Axis3(fig[1, 1]; perspectiveness=1.0, aspect=:data, viewmode=:fitzoom,
    #            rotation=Billboard(), azimuth=0.0, elevation=0.0)
    fig = Figure(resolution=(600, 1000))#,camera = cam3d!)
    axs, = scatter(fig[1,1], [Point3f([0,0,0]), Point3f([1,1,1])], markersize=0,
                   transparency=true, axis=(viewmode=:fitzoom,))
    for pop=pops
         scatters[pop] = scatter!(positions[pop], color=log[pop][:, 1], markersize=20,
                                  colormap = :inferno, colorrange=(0.0, 1.0), transparency=true)
    end
    plot_labels!()
    linesegments!(connections, linewidth=.5, color=colors, transparency=true)
    
    dop_colormap = [RGBAf(0,(i>12) ? (i-12)/10 : 0,0,(i>10) ? (i-10)/10 : 0) for i=1:20]
    varico_act = Observable(zeros(snc_size))
    r = LinRange(-0.2,1.2,50)
    dop = lift(a->getDopamineConc(r, varico_pos, a; lam=1.5e-3), varico_act)
    vol = volume!(axs, r, r, r, dop, algorithm = :absorption, transparency=true, colormap=dop_colormap)
    
    Makie.scale!(fig.scene, 1.2, 1.2, 1.2)
    record(fig, filename; framerate=30, compression=2) do io
        @showprogress "Animating $filename" for t = 1:200
            for pop in pops
                scatters[pop].color = log[pop][:, t]
            end
            varico_act[] = 100*(0.15 .+ log[:snc][:, t])
            recordframe_withfilter!(io) do frame
                blur = imfilter(frame .* (Gray.(frame) .> 0.5), Kernel.gaussian(2.0))
                clamp01.(frame + 2*blur)
            end
        end
    end
    fig
end
function add_snc(log, target)
    merge(log, NamedTuple((:snc=>target'-log[:snr],)))
end
pops = (:thal, :ctx_exc, :ctx_inh, :str_dmsn, :str_imsn, :snr, :snc)

net = BgNet(200, 4, 1e-3, 25*1e-3, SynapseType=EligabilitySynapse)
input = create_input(size(net[:thal]), 200)
target = 0.5 .+ 0.15*gaussianProcessTarget(200, 4, 20)
input_fcn(t) = input[t, :]
target_fcn(t) = target[t, :]
log = recordSampleRun(net, 200, clamp=(thal=input_fcn,))
log = recordSampleRun(net, 200, clamp=(thal=input_fcn,))
log = add_snc(log, target)
animate_activity("full_net_anim_untrained_GL_with_dopamine.mp4", net, log, target)

#net, losses = train_network("Training network", synapseType=EligabilitySynapse)
#log = recordSampleRun(net, 200, clamp=(thal=input_fcn,))
losses = Float64[]
@showprogress "Training network" for trial_id=1:1000
    loss = run_trial(net, target, input, :dopamine)
    push!(losses, loss)
end

log = recordSampleRun(net, 200, clamp=(thal=input_fcn,))
animate_activity("full_net_anim_trained_GL.mp4", net, log)
#points = rand(Point3f, 200)
#segs = Point3f[]
#for i=1:length(points), j=1:length(points)
#    if (rand() < 0.2 && i!=j)
#        push!(segs, points[i])
#        push!(segs, points[j])
#    end
#end

#colors = gaussianProcessTarget(200, 200, 20.0)

#start_colors = 1 ./ (1 .+ exp.(-colors[1, :] .+ 2))
#set_theme!(theme_black())
#set_theme!(backgroundcolor=(:black, 1.0))
#f = Figure(resolution=(400, 1000))
#3ax, scatterObj = scatter(f[1,1], points, color=start_colors, markersize=5, #colormap = :inferno, colorrange=(0.0, 1.0)) #[(:white, rand()) for i=1:200]
#linesegments!(segs, linewidth=0.5, color=(:gray, 0.05))
#f

#my_im = Makie.colorbuffer(f.scene, Makie.GLNative)
#blur = imfilter(my_im .* (Gray.(my_im) .> 0.4), Kernel.gaussian(2))
#clamp01.(my_im + 2*blur)

#r = LinRange(0,1,10) .- .5
#dop = [exp(-sqrt(x*x+y*y+z*z)) for x=r, y=r, z=r]
#record(fig, "dummy_animation.mp4", 1:200;
#        framerate = 20) do t
#    scatterObj.color = 1 ./ (1 .+ exp.(-colors[t, :] .+ 2))
#end
#include("makie_bloom.jl")
#record(fig, "dummy_filtered.mp4"; framerate=20) do io
#    for t = 1:200
#        scatterObj.color = 1 ./ (1 .+ exp.(-colors[t, :] .+ 2))
#        recordframe_withbloom!(io)
#    end
#end


colormap = [RGBAf(0,(i>12) ? (i-12)/10 : 0,0,(i>10) ? (i-10)/10 : 0) for i=1:20]
r = LinRange(-0.2,1.2,30)
centers = rand(Point3f, 40)
function getDop(x, y, z)
    dop = 0
    for i=1:size(centers,1)
        dx = x .- centers[i][1]
        dy = y .- centers[i][2]
        dz = z .- centers[i][3]
        dop += randn()*exp(-sqrt((dx*dx + dy*dy + dz*dz)/0.025))
    end
    dop
end
dop = [0*getDop(x,y,z) for x=r, y=r, z=r]
fig = Figure()
axs, = scatter(fig[1,1], centers, color=colorant"green", transparency=true, markersize=1)
vol = volume!(fig[1,1], r, r, r, dop, algorithm = :absorption, transparency=true, colormap=colormap)
fig

points = rand(Point3f, 200)
segs = Point3f[]
for i=1:length(points), j=1:length(points)
    if (rand() < 0.2 && i!=j)
        push!(segs, points[i])
        push!(segs, points[j])
    end
end
linesegments!(segs, linewidth=0.5, color=RGBAf(0.0, 0.0, 0.0, 0.05), transparency=true)
scatter!(axs, points, markersize=20, transparency=true)

fig

