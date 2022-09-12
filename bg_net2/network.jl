using LinearAlgebra

abstract type Synapse end
abstract type PlasticSynapse <: Synapse end

mutable struct Population
    v::Vector{Float64}
    r::Vector{Float64}
    projections::Dict{Population, Vector{Vector{T}} where T <: Synapse}
    alpha::Float64
    bias::Float64
    noise::Float64
end

Base.size(pop::Population) = length(pop.v)
function Population(size::Integer; tau=20.0, bias=0.0, noise=0.0)
    Population(zeros(size), zeros(size),
        Dict{Population, Vector{Vector{T}} where T <: Synapse}(),
        exp(-1/tau), bias, noise)
end
phi(v) = 1/(1+exp(-v+2))

mutable struct StaticSynapse <: Synapse
    weight::Float64
    pre::UInt64
end

mutable struct EligabilitySynapse{sign} <: PlasticSynapse
    weight::Float64
    pre::UInt64
    trace::Float64
end

mutable struct BalanceSynapse <: Synapse
    weight::Float64
    pre::UInt64
end

mutable struct AdamSynapse{sign} <: PlasticSynapse
    weight::Float64
    pre::UInt64
    trace::Float64
    m::Float64
    v::Float64
end

mutable struct DelayedSynapse{sign} <: PlasticSynapse
    weight::Float64
    pre::UInt64
    trace::Float64
    dopTrace::Float64
    traceTrace::Float64
end



EligabilitySynapse{s}(weight, pre) where s = EligabilitySynapse{s}(weight, pre, 0.0)
Base.sign(::EligabilitySynapse{s}) where s = s

AdamSynapse{s}(weight, pre) where s = AdamSynapse{s}(weight, pre, 0.0, 0.0, 0.0)
Base.sign(::AdamSynapse{s}) where s = s

DelayedSynapse{s}(weight, pre) where s = DelayedSynapse{s}(weight, pre, 0.0, 0.0, 0.0)
Base.sign(::DelayedSynapse{s}) where s = s

function connect!(synapseType::Type, prePop::Population, postPop::Population,
                  weight, prob)
    synapses = [synapseType[] for _=1:size(postPop)]
    for post=1:size(postPop), pre=1:size(prePop)
        p = prob isa Function ? prob(pre, post) : prob
        if p >= rand()
            w = weight isa Function ? weight(pre, post) : weight
            push!(synapses[post], synapseType(w, pre))
        end
    end
    postPop.projections[prePop] = synapses
end

function eachSynapse(f::Function, postPop::Population; synapseType::Type=Synapse)
    for (prePop, synapses) in postPop.projections
        if eltype(eltype(synapses)) <: synapseType
            @Threads.threads for post=1:size(postPop)
                for synapse=synapses[post]
                    f(synapse, prePop, post)
                end
            end
        end
    end
end

function step!(pop::Population)
    pop.v .= pop.alpha .* pop.v .+ (1-pop.alpha) .* pop.bias
    
    eachSynapse(pop) do synapse, prePop, post
        pop.v[post] += (1-pop.alpha) * synapse.weight * prePop.r[synapse.pre]
    end
    
    r = phi.(pop.v)
    dr = r .* (1 .- r) #(1 .- r.^2)
    
    #Union{EligabilitySynapse, AdamSynapse}) do synapse, prePop, post
    eachSynapse(pop, synapseType=PlasticSynapse) do synapse, prePop, post
        synapse.trace *= pop.alpha
        synapse.trace += (1-pop.alpha)*prePop.r[synapse.pre]*dr[post]
    end
    
    pop.r = r + pop.noise*randn(size(pop))
end

function updateWeights!(pop::Population, feedback, eta, t)
    eachSynapse(pop, synapseType=EligabilitySynapse{1}) do synapse, prePop, post
        synapse.weight += eta*synapse.trace*feedback[post]
        synapse.weight = max(synapse.weight, 0)
    end
    eachSynapse(pop, synapseType=EligabilitySynapse{-1}) do synapse, prePop, post
        synapse.weight += eta*synapse.trace*feedback[post]
        synapse.weight = min(synapse.weight, 0)
    end
    eachSynapse(pop, synapseType=BalanceSynapse) do synapse, prePop, post
        #sgn = sign(synapse.weight)
        synapse.weight += eta*prePop.r[synapse.pre]*(-pop.v[post])
        synapse.weight = min(synapse.weight, 0.0)
        #synapse.weight *= (sgn*synapse.weight) >= 0
    end
    
    for synSign in [1, -1]
        eachSynapse(pop, synapseType=AdamSynapse{synSign}) do synapse, prePop, post
            g = synapse.trace*feedback[post]
            synapse.m = 0.9*synapse.m + 0.1*g
            synapse.v = 0.99*synapse.v + 0.01*g*g
            mhat = synapse.m / (1-0.9^t)
            vhat = synapse.v / (1-0.99^t)
            synapse.weight += eta*mhat/(sqrt(vhat)+1e-6)
            if synSign == 1
                synapse.weight = max(synapse.weight, 0)
            else
                synapse.weight = min(synapse.weight, 0)
            end
        end
        
        eachSynapse(pop, synapseType=DelayedSynapse{synSign}) do synapse, prePop, post
            synapse.dopTrace = 0.9*synapse.dopTrace + 0.1*feedback[post]
            synapse.traceTrace = 0.9*synapse.traceTrace + 0.1*synapse.trace
            synapse.weight += eta*synapse.traceTrace*synapse.dopTrace
            if synSign == 1
                synapse.weight = max(synapse.weight, 0)
            else
                synapse.weight = min(synapse.weight, 0)
            end
        end
    end
end

function getWeightMatrix(pops::Vector{Population})
    tot_size = sum(size.(pops))
    W = zeros(tot_size, tot_size)
    offset = Dict(zip(pops, [0; cumsum(size.(pops))[1:end-1]]))
    for pop in pops
        eachSynapse(pop) do synapse, prePop, post
            if prePop in keys(offset)
                i = offset[pop] + post
                j = offset[prePop] + synapse.pre
                W[i, j] += synapse.weight
            end
        end
    end
    return W
end

function getWeightMatrix(prePop::Population, postPop::Population)
    W = zeros(size(postPop), size(prePop))
    projection = postPop.projections[prePop]
    for post=1:size(postPop)
        for synapse in projection[post]
            W[post, synapse.pre] += synapse.weight
        end
    end
    return W
end

function balanceWeights!(pop::Population)
    for i=1:size(pop)
        totW = 0.0
        cntExc = 0
        cntInh = 0
        for synapses in values(pop.projections)
            totW += sum(getfield.(synapses[i], :weight))
            cntExc += sum(getfield.(synapses[i], :weight) .>= 0)
            cntInh += sum(getfield.(synapses[i], :weight) .<  0)
        end
        cnt = totW >= 0 ? cntInh : cntExc
        for synapses in values(pop.projections)
            map(synapses[i]) do synapse
                if Base.sign(synapse.weight) != Base.sign(totW)
                    synapse.weight -= totW / cnt
                end
            end
        end
    end
end