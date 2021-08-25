using LinearAlgebra

mutable struct RPE_NET
    W_in::Matrix{Float64}
    W_rec::Matrix{Float64}
    W_out::Matrix{Float64}
    DA::Matrix{Float64}
    tau_m::Float64
    eta_in::Float64
    eta_rec::Float64
    eta_out::Float64
    v::Vector{Float64}
    s::Vector{Float64}
    p::Matrix{Float64}
    q::Matrix{Float64}
end

function RPE_NET(W_in, W_rec, W_out, tau_m, eta_in, eta_rec, eta_out)
    n_hidden, n_in = size(W_in)
    n_out, n_hidden = size(W_out)
    DA = ones(n_hidden, n_out)
    DA[end÷2+1:end, :] .*= -1
    RPE_NET(W_in, W_rec, W_out, DA, tau_m, eta_in, eta_rec, eta_out,
             zeros(n_hidden), zeros(n_hidden), zero(W_rec), zero(W_in))
end

function RPE_NET(W_in, W_rec, W_out, tau_m, eta)
    RPE_NET(W_in, W_rec, W_out, tau_m, eta, eta, eta)
end

function RPE_NET(n_in, n_hidden, n_out; in_scale=0.8, rec_scale=-2.0,
                      out_scale=2.0, tau_m=10, eta_in=5e-3, eta_rec=5e-3,
                      eta_out=5e-3, lambda=0.1)
    W_in = in_scale*rand(n_hidden, n_in)
    W_rec = rec_scale*rand(n_hidden, n_hidden) / n_hidden
    W_out = out_scale*(2*rand(n_out, n_hidden) .- 1) / sqrt(n_hidden)
    RPE_NET(W_in, W_rec, W_out, tau_m, eta_in, eta_rec, eta_out)
end
function reset_state!(net::RPE_NET)
    net.v .= 0.0
    net.p .= 0.0
    net.q .= 0.0
    
end
function step!(net::RPE_NET, inp, target)
    prev_h = @. 1/(1+exp(-net.v))
    u = net.W_rec*prev_h + net.W_in*inp .- 2
    net.v += -net.v/net.tau_m + u
    #net.s += (1 .- net.s .- prev_h.*prev_h)/(0.5*net.tau_m)
    
    h = @. 1/(1+exp(-net.v))
    dh = @. (1-h)*h
    net.p += -net.p/net.tau_m + dh*prev_h'
    net.q += -net.q/net.tau_m + dh*inp'
    
    y = net.W_out*h
    err = target-y
    feedback = net.DA*(norm(err)*ones(size(y)))
    
    net.W_out += net.eta_out*   err   *h'
    net.W_rec += net.eta_rec*feedback.*net.p
    net.W_in  += net.eta_in *feedback.*net.q
    
    net.W_rec .= min.(net.W_rec, 0)
    net.W_in .= max.(net.W_in, 0)
    net.W_out[:, 1:end÷2] .= max.(net.W_out[:, 1:end÷2], 0)
    net.W_out[:, end÷2+1:end] .= min.(net.W_out[:, end÷2+1:end], 0)
    return y
end

function episode!(net::RPE_NET, inp, target)
    reset_state!(net)
    t_max = size(inp)[1]
    y = zero(target)
    h = zeros(t_max, length(net.v))
    for t=2:t_max
        y[t, :] = step!(net, inp[t, :], target[t, :])
        h[t, :] = @. 1/(1+exp(-net.v))
    end
    return y, h
end
