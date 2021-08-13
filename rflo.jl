mutable struct RFLO_NET
    W_in::Matrix{Float64}
    W_rec::Matrix{Float64}
    W_out::Matrix{Float64}
    tau_m::Float64
    eta_in::Float64
    eta_rec::Float64
    eta_out::Float64
    h::Vector{Float64}
    p::Matrix{Float64}
    q::Matrix{Float64}
end

function RFLO_NET(W_in, W_rec, W_out, tau_m, eta_in, eta_rec, eta_out)
    n_hidden, n_in = size(W_in)
    RFLO_NET(W_in, W_rec, W_out, tau_m, eta_in, eta_rec, eta_out,
             zeros(n_hidden), zero(W_rec), zero(W_in))
end

function RFLO_NET(W_in, W_rec, W_out, tau_m, eta)
    RFLO_NET(W_in, W_rec, W_out, tau_m, eta, eta, eta)
end

function RFLO_NET(n_in, n_hidden, n_out; in_scale=1, rec_scale=5, out_scale=2,
                  tau_m=10, eta_in=5e-3, eta_rec=5e-3, eta_out=5e-3)
    W_in = (in_scale*rand(n_hidden, n_in) .- 1)
    W_rec = rec_scale*randn(n_hidden, n_hidden) / sqrt(n_hidden)
    W_out = out_scale*(2*rand(n_out, n_hidden) .- 1) / sqrt(n_hidden)
    RFLO_NET(W_in, W_rec, W_out, tau_m, eta_in, eta_rec, eta_out)
end
function reset_state!(net::RFLO_NET)
    net.h .= 0.0
    net.p .= 0.0
    net.q .= 0.0
end
function step!(net::RFLO_NET, inp, target)
    u = net.W_rec*net.h + net.W_in*inp
    f = @. 2/(1+exp(-u))-1
    df = @. (1-f*f)/2
    net.h += (-net.h + f)/net.tau_m   
    net.p += (-net.p + df*net.h')/net.tau_m
    net.q += (-net.q + df*inp')/net.tau_m
    
    y = net.W_out*net.h
    err = target-y
    feedback = net.W_out'*err
    
    net.W_out += net.eta_out*   err   *net.h'
    net.W_rec += net.eta_rec*feedback.*net.p
    net.W_in  += net.eta_in *feedback.*net.q
    return y
end

function episode!(net::RFLO_NET, inp, target)
    reset_state!(net)
    t_max = size(inp)[1]
    y = zero(target)
    h = zeros(t_max, length(net.h))
    for t=2:t_max
        y[t, :] = step!(net, inp[t, :], target[t, :])
        h[t, :] = net.h
    end
    return y, h
end
