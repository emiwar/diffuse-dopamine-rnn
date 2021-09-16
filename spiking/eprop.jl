using LinearAlgebra

struct SpikingLayer
    constraints::Symbol
    W_inp::Matrix{Float64}
    W_rec::Matrix{Float64}
    v::Vector{Float64}
    inp_trace::Vector{Float64}
    rec_trace::Vector{Float64}
    last_spike::Vector{Float64}
    alpha::Float64
    refract::Float64
    eta::Float64
    #P::Matrix{Float64}
end

function SpikingLayer(constraints::Symbol, size::Integer, inp_size::Integer;
                      g_inp=5e-2, g_rec=5e-2, eta=1e-5, tau=20.0,
                      dt=1.0, refract=2.0)
    if constraints == :striatum
        W_inp =  rand(size, inp_size) * g_inp / inp_size
        W_rec = -rand(size, size) * g_rec / size
    elseif constraints == :gp
        W_inp = -rand(size, inp_size) * g_inp / inp_size
        W_rec = zeros(size, size)
    elseif constraints == :thalamus
        W_inp = -rand(size, inp_size) * g_inp / inp_size
        W_rec = randn(size, size) * g_rec / sqrt(size)
    end
    v = rand(size)
    inp_trace = zeros(inp_size)
    rec_trace = zeros(size)
    last_spike = 1e255 * ones(size)
    alpha = exp(-dt/tau)
    SpikingLayer(constraints, W_inp, W_rec, v, inp_trace, rec_trace, last_spike, alpha,
                 refract, eta, 0.01*I(size))
end

function step!(net::SpikingLayer, inp)
    if net.constraints == :gp
        cur = net.W_inp*inp + 0.2
    else
        cur = net.W_inp*inp + net.W_rec*(net.last_spike.==0.0)
    end
    not_refract = (net.last_spike .> net.refract)
    net.v .= (net.alpha*net.v .+ cur) .* not_refract
    net.last_spike .+= 1
    net.last_spike .*= net.v .< 1.0
    net.inp_trace .= net.alpha .* net.inp_trace .+ inp
    net.rec_trace .= net.alpha .* net.rec_trace .+ (net.last_spike.==0.0)
    #P = net.P
    #r = net.rec_trace
    #net.P .-= P*r*(r')*P / (1+(r')*P*r)
end

function learn!(net::SpikingLayer, L)
    psi = L .* max.(0, net.v)
    net.W_inp .-= net.eta*psi*net.inp_trace'
    net.W_rec .-= net.eta*psi*net.rec_trace'
    #net.W_inp .-= psi*(net.inp_trace)'
    #net.W_rec .-= psi*(net.rec_trace)'
end

mutable struct StriatumOnly
    striatum::SpikingLayer
    W_readout::Vector{Float64}
    input::Float64
    eta::Float64
end

function step!(net::StriatumOnly, target::Float64)
    step!(net.striatum, [net.input])
    out = net.W_readout'*net.striatum.rec_trace
    err = out - target
    #L = (net.W_readout .* net.striatum.rec_trace) * err
    L = net.W_readout * err
    learn!(net.striatum, L)
    net.W_readout .-= net.eta*err*net.striatum.rec_trace
    net.input = out+0.5
    return out
end

mutable struct BgNet2
    striatum::SpikingLayer
    gpi::SpikingLayer
    gpe::SpikingLayer
    thal::SpikingLayer
end

function BgNet2(strSize, gpSize, thalSize)
    str = SpikingLayer(:striatum, strSize, thalSize)
    gpi = SpikingLayer(:gp, gpSize, strSize+gpSize)
    gpe = SpikingLayer(:gp, gpSize, strSize)
    thal = SpikingLayer(:thalamus, thalSize, gpSize)
    BgNet2(str, gpi, gpe, thal)
end


