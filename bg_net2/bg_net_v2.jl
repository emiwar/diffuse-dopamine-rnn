include("network.jl")

mutable struct BgNet
    populations::Dict{Symbol, Population}
    eta::Float64
    str_dmsn_pos::Matrix{Float64}
    str_imsn_pos::Matrix{Float64}
    feedback_dmsn::Matrix{Float64}
    feedback_imsn::Matrix{Float64}
end

function BgNet(size::Integer, readout_size::Integer, eta::Float64)
    populations = Dict{Symbol, Population}()
    
    populations[:ctx_exc] = Population(floor(Int, 0.8*size), bias=0.0)
    populations[:ctx_inh] = Population(floor(Int, 0.2*size))
    populations[:str_dmsn] = Population(floor(Int, 0.5*size))
    populations[:str_imsn] = Population(floor(Int, 0.5*size))
    populations[:snr] = Population(readout_size, tau=0.0, bias=1.0)
    populations[:thal] = Population(readout_size*10, tau=0.0, bias=0.0)

    connect!(StaticSynapse, populations[:ctx_exc], populations[:ctx_exc],
             (pre, post)->25*rand()/sqrt(size), (pre, post)->0.1*(pre!=post))
    connect!(StaticSynapse, populations[:ctx_exc], populations[:ctx_inh],
             (pre, post)->75*rand()/sqrt(size), 0.1)
    connect!(BalanceSynapse, populations[:ctx_inh], populations[:ctx_exc],
             (pre, post)->-25*rand()/sqrt(size), 0.4)
    connect!(BalanceSynapse, populations[:ctx_inh], populations[:ctx_inh],
             (pre, post)->-75*rand()/sqrt(size), (pre, post)->0.4*(pre!=post))
    #connect!(StaticSynapse, populations[:ctx_exc], populations[:str_dmsn],
    #         (pre, post)->5*rand()/sqrt(size), 0.2)
    #connect!(StaticSynapse, populations[:ctx_exc], populations[:str_imsn],
    #         (pre, post)->5*rand()/sqrt(size), 0.2)
    connect!(StaticSynapse, populations[:str_dmsn], populations[:str_dmsn],
             (pre, post)->-50*rand()/sqrt(size), (pre, post)->0.1*(pre!=post))
    connect!(StaticSynapse, populations[:str_dmsn], populations[:str_imsn],
             (pre, post)->-50*rand()/sqrt(size), 0.1)
    connect!(StaticSynapse, populations[:str_imsn], populations[:str_dmsn],
             (pre, post)->-50*rand()/sqrt(size), 0.1)
    connect!(StaticSynapse, populations[:str_imsn], populations[:str_imsn],
             (pre, post)->-50*rand()/sqrt(size), (pre, post)->0.1*(pre!=post))
    connect!(EligabilitySynapse, populations[:str_dmsn], populations[:snr],
             (pre, post)->-25*rand()/sqrt(size), 1.0)
    connect!(EligabilitySynapse, populations[:str_imsn], populations[:snr],
             (pre, post)-> 25*rand()/sqrt(size), 1.0)
    #connect!(StaticSynapse, populations[:snr], populations[:thal],
    #         (pre, post)->-5*rand(), 1.0)
    connect!(StaticSynapse, populations[:thal], populations[:ctx_exc],
             (pre, post)->25*rand()/Base.size(populations[:thal]), 0.25)
    connect!(EligabilitySynapse, populations[:thal], populations[:str_dmsn],
             (pre, post)->100*rand()/Base.size(populations[:thal]), 0.25)
    connect!(EligabilitySynapse, populations[:thal], populations[:str_imsn],
             (pre, post)->100*rand()/Base.size(populations[:thal]), 0.25)
    str_dmsn_pos = rand(size, 3)
    str_imsn_pos = rand(size, 3)
    feedback_dmsn = createFeedbackMatrix(str_dmsn_pos, readout_size)
    feedback_imsn = createFeedbackMatrix(str_imsn_pos, readout_size)
    #ctx_exc_avg = 0.5*ones(Base.size(populations[:ctx_exc]))
    return BgNet(populations, eta, str_dmsn_pos, str_imsn_pos, feedback_dmsn, feedback_imsn)   
end

pop_order(::BgNet) = (:ctx_exc, :ctx_inh, :str_dmsn, :str_imsn, :snr, :thal)

function step!(net::BgNet; clamp::NamedTuple=NamedTuple())
    for pop in pop_order(net)
        step!(net[pop])
        if pop in keys(clamp)
            net[pop].r = clamp[pop]
        end
    end
end

function step!(net::BgNet, target; clamp::NamedTuple=NamedTuple(), updateStriatum=:dopamine)
    step!(net, clamp=clamp)
    error = net[:snr].r - target
    updateWeights!(net[:snr], -error, net.eta)
    if updateStriatum==:dopamine
        postFactor = net[:snr].r .* (1 .- net[:snr].r)
    #feedback_dmsn = -100*(getWeightMatrix(net[:str_dmsn], net[:snr]))'*(error .* postFactor)
    #feedback_imsn = -100*(getWeightMatrix(net[:str_imsn], net[:snr]))'*(error .* postFactor)
        feedback_dmsn =  100*(net.feedback_dmsn)*(error .* postFactor)
        feedback_imsn = -100*(net.feedback_imsn)*(error .* postFactor)
        updateWeights!(net[:str_dmsn], feedback_dmsn, net.eta)
        updateWeights!(net[:str_imsn], feedback_imsn, net.eta)
    elseif updateStriatum==:ideal
        postFactor = net[:snr].r .* (1 .- net[:snr].r)
        feedback_dmsn = -100*(getWeightMatrix(net[:str_dmsn], net[:snr]))'*(error .* postFactor)
        feedback_imsn = -100*(getWeightMatrix(net[:str_imsn], net[:snr]))'*(error .* postFactor)
        updateWeights!(net[:str_dmsn], feedback_dmsn, net.eta)
        updateWeights!(net[:str_imsn], feedback_imsn, net.eta)
    end
    #net.ctx_exc_avg .= 0.999 .* net.ctx_exc_avg .+ 0.001 .* net[:ctx_exc].r
    #updateWeights!(net[:ctx_exc], 0.5 .- net.ctx_exc_avg, net.eta)
    #updateWeights!(net[:ctx_inh], 0.5 .- net[:ctx_inh].r, net.eta)
    #updateWeights!(net[:ctx_exc], nothing, net.eta)
    #updateWeights!(net[:ctx_inh], nothing, net.eta)
    #net[:snr].r .= target
    return sum(error.^2)
end

function recordSampleRun(net::BgNet, T::Integer; clamp::NamedTuple=NamedTuple())
    log = NamedTuple(pop=>zeros(size(net[pop]), T) for pop in pop_order(net))
    for t=1:T
        step!(net, clamp=map(x->x(t), clamp))
        for pop in pop_order(net)
            log[pop][:, t] = net[pop].r
        end
    end
    return log
end

function Base.show(io::IO, net::BgNet)
    println(io, "BgNet")
    for pop in pop_order(net)
        println("  $pop: $(size(net[pop])) units")
    end
    println("  Learning rate: $(net.eta)")
end

Base.getindex(net::BgNet, pop::Symbol) = net.populations[pop]

function createFeedbackMatrix(str_pos, output_size; n_varicosities=10, lambda=0.1)
    str_size = size(str_pos, 1)
    DA = zeros(str_size, output_size)
    for i=1:output_size
        varicosities_pos = rand(n_varicosities, 3)
        for j=1:str_size, k=1:n_varicosities
            d = norm(str_pos[j, :] - varicosities_pos[k, :])
            DA[j, i] += exp(-d/lambda)
        end
    end
    return DA
end

## Plotting

function plotHeatmaps(log::NamedTuple)
    y = 1
    yticks = Float64[]
    p = plot()
    for pop in keys(log)
        dy = size(log[pop], 1)
        heatmap!(p, 1:T, y:(y+dy-1), log[pop], clim=(0, 1))
        push!(yticks, y+0.5dy)
        y += dy+5
    end
    plot!(p, yticks=(yticks, keys(log)))
end

function plotTraces(log::NamedTuple; target=nothing)
    y = 0
    yticks = Float64[]
    p = plot(size=(700, 800))
    for pop in keys(log)
        T = size(log[pop], 2)
        plot!(p, 1:T, log[pop]'.+y)
        if pop==:snr && target != nothing
            tar = target isa Function ? hcat(target.(1:T)...)' : target
            plot!(1:T, tar.+y, color=[1 2], linestyle=:dash)
        end
        push!(yticks, y+0.6)
        y += 1.2
    end
    plot!(p, yticks=(yticks, keys(log)), legend=false)
end

plotHeatmaps(net::BgNet, T::Integer) = plotHeatmaps(recordSampleRun(net, T))
plotTraces(net::BgNet, T::Integer; target=nothing) = plotTraces(recordSampleRun(net, T), target=target)