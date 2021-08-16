mutable struct REC_NET
    W_in::Matrix{Float64}
    W_rec::Matrix{Float64}
    W_out::Matrix{Float64}
    tau_m::Float64
    eta_in::Float64
    eta_rec::Float64
    eta_out::Float64
    v::Vector{Float64}
    p::Matrix{Float64}
    q::Matrix{Float64}
end

function REC_NET(W_in, W_rec, W_out, tau_m, eta_in, eta_rec, eta_out)
    n_hidden, n_in = size(W_in)
    REC_NET(W_in, W_rec, W_out, tau_m, eta_in, eta_rec, eta_out,
             zeros(n_hidden), zero(W_rec), zero(W_in))
end

function REC_NET(W_in, W_rec, W_out, tau_m, eta)
    REC_NET(W_in, W_rec, W_out, tau_m, eta, eta, eta)
end

function REC_NET(n_in, n_hidden, n_out; in_scale=.05, rec_scale=0.5, out_scale=.5,
                  tau_m=10, eta_in=5e-3, eta_rec=5e-3, eta_out=5e-3)
    W_in = in_scale*(2*rand(n_hidden, n_in) .- 1)
    W_rec = rec_scale*randn(n_hidden, n_hidden) / sqrt(n_hidden)
    W_out = out_scale*(2*rand(n_out, n_hidden) .- 1) / sqrt(n_hidden)
    REC_NET(W_in, W_rec, W_out, tau_m, eta_in, eta_rec, eta_out)
end
function reset_state!(net::REC_NET)
    net.v .= 0.0
    net.p .= 0.0
    net.q .= 0.0
end
function step!(net::REC_NET, inp, target)
    prev_h = @. 2/(1+exp(-net.v))-1
    u = net.W_rec*prev_h + net.W_in*inp
    net.v += -net.v/net.tau_m + u
    
    h = @. 2/(1+exp(-net.v))-1
    dh = @. (1-h*h)/2
    net.p += -net.p/net.tau_m + dh*prev_h'
    net.q += -net.q/net.tau_m + dh*inp'
    
    y = net.W_out*h
    err = target-y
    feedback = net.W_out'*err
    
    net.W_out += net.eta_out*   err   *h'
    net.W_rec += net.eta_rec*feedback.*net.p
    net.W_in  += net.eta_in *feedback.*net.q
    return y
end

function episode!(net::REC_NET, inp, target)
    reset_state!(net)
    t_max = size(inp)[1]
    y = zero(target)
    h = zeros(t_max, length(net.v))
    for t=2:t_max
        y[t, :] = step!(net, inp[t, :], target[t, :])
        h[t, :] = @. 2/(1+exp(-net.v))-1
    end
    return y, h
end
