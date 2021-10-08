include("network.jl")

mutable struct BgNet
    populations::Dict{Symbol, Population}
    eta::Float64
    ctx_exc_avg::Vector{Float64}
end

function BgNet(size::Integer, readout_size::Integer, eta::Float64)
    populations = Dict{Symbol, Population}()
    
    populations[:ctx_exc] = Population(floor(Int, 0.8*size), bias=0.0)
    populations[:ctx_inh] = Population(floor(Int, 0.2*size))
    populations[:str_dmsn] = Population(floor(Int, 0.5*size))
    populations[:str_imsn] = Population(floor(Int, 0.5*size))
    populations[:snr] = Population(readout_size, tau=0.0)

    connect!(PlasticSynapse, populations[:ctx_exc], populations[:ctx_exc],
             (pre, post)->5*rand()/sqrt(size), (pre, post)->0.1*(pre!=post))
    connect!(StaticSynapse, populations[:ctx_exc], populations[:ctx_inh],
             (pre, post)->5*rand()/sqrt(size), 0.1)
    connect!(StaticSynapse, populations[:ctx_inh], populations[:ctx_exc],
             (pre, post)->-5*rand()/sqrt(size), 0.4)
    connect!(StaticSynapse, populations[:ctx_inh], populations[:ctx_inh],
             (pre, post)->-5*rand()/sqrt(size), (pre, post)->0.4*(pre!=post))
    connect!(PlasticSynapse, populations[:ctx_exc], populations[:str_dmsn],
             (pre, post)->5*rand()/sqrt(size), 0.2)
    connect!(PlasticSynapse, populations[:ctx_exc], populations[:str_imsn],
             (pre, post)->5*rand()/sqrt(size), 0.2)
    connect!(PlasticSynapse, populations[:str_dmsn], populations[:str_dmsn],
             (pre, post)->-5*rand()/sqrt(size), (pre, post)->0.1*(pre!=post))
    connect!(PlasticSynapse, populations[:str_dmsn], populations[:str_imsn],
             (pre, post)->-5*rand()/sqrt(size), 0.1)
    connect!(PlasticSynapse, populations[:str_imsn], populations[:str_dmsn],
             (pre, post)->-5*rand()/sqrt(size), 0.1)
    connect!(PlasticSynapse, populations[:str_imsn], populations[:str_imsn],
             (pre, post)->-5*rand()/sqrt(size), (pre, post)->0.1*(pre!=post))
    connect!(PlasticSynapse, populations[:str_dmsn], populations[:snr],
             (pre, post)->-rand()/sqrt(size), 1.0)
    connect!(PlasticSynapse, populations[:str_imsn], populations[:snr],
             (pre, post)-> rand()/sqrt(size), 1.0)
    connect!(StaticSynapse, populations[:snr], populations[:ctx_exc],
             (pre, post)->-2*rand(), 0.2)
    
    ctx_exc_avg = 0.5*ones(Base.size(populations[:ctx_exc]))
    return BgNet(populations, eta, ctx_exc_avg)    
end

pop_order(::BgNet) = (:ctx_exc, :ctx_inh, :str_dmsn, :str_imsn, :snr)

function step!(net::BgNet)
    for pop in pop_order(net)
        step!(net[pop])
    end
end

function step!(net::BgNet, target)
    step!(net)
    #error = net[:snr].r - target
    #updateWeights!(net[:snr], -error, net.eta)
    net.ctx_exc_avg .= 0.999 .* net.ctx_exc_avg .+ 0.001 .* net[:ctx_exc].r
    updateWeights!(net[:ctx_exc], 0.5 .- net.ctx_exc_avg, net.eta)
    #updateWeights!(net[:ctx_inh], 0.5 .- net[:ctx_inh].r, net.eta)
    #net[:snr].r .= target
end

function Base.show(io::IO, net::BgNet)
    println(io, "BgNet")
    for pop in pop_order(net)
        println("  $pop: $(size(net[pop])) units")
    end
    println("  Learning rate: $(net.eta)")
end

Base.getindex(net::BgNet, pop::Symbol) = net.populations[pop]