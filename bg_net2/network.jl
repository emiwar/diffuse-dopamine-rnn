using LinearAlgebra

abstract type Synapse end
abstract type PlasticSynapse <: Synapse end

mutable struct Population
    v::Vector{Float64}
    r::Vector{Float64}
    projections::Dict{Population, Vector{Vector{T}} where T <: Synapse}
    alpha::Float64
    bias::Float64
end

Base.size(pop::Population) = length(pop.v)
Population(size::Integer; tau=20.0, bias=0.0) = Population(zeros(size), zeros(size), Dict{Population, Vector{Vector{T}} where T <: Synapse}(), exp(-1/tau), bias)
phi(v) = 1/(1+exp(-v))

mutable struct StaticSynapse <: Synapse
    weight::Float64
    pre::UInt64
end

mutable struct EligabilitySynapse <: PlasticSynapse
    weight::Float64
    pre::UInt64
    trace::Float64
end

mutable struct BalanceSynapse <: PlasticSynapse
    weight::Float64
    pre::UInt64
end

EligabilitySynapse(weight, pre) = EligabilitySynapse(weight, pre, 0.0)

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
    
    eachSynapse(pop, synapseType=EligabilitySynapse) do synapse, prePop, post
        synapse.trace *= pop.alpha
        synapse.trace += (1-pop.alpha)*prePop.r[synapse.pre]*dr[post]
    end
    
    pop.r = r
end

function updateWeights!(pop::Population, feedback, eta)
    eachSynapse(pop, synapseType=EligabilitySynapse) do synapse, prePop, post
        sgn = sign(synapse.weight)
        synapse.weight += eta*synapse.trace*feedback[post]
        synapse.weight *= (sgn*synapse.weight) >= 0
    end
    eachSynapse(pop, synapseType=BalanceSynapse) do synapse, prePop, post
        #sgn = sign(synapse.weight)
        synapse.weight += eta*prePop.r[synapse.pre]*(-pop.v[post])
        synapse.weight = min(synapse.weight, 0.0)
        #synapse.weight *= (sgn*synapse.weight) >= 0
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
                if sign(synapse.weight) != sign(totW)
                    synapse.weight -= totW / cnt
                end
            end
        end
    end
end