using LinearAlgebra

abstract type REC_NET end

r(net::REC_NET) = tanh.(net.v)

function reset_state!(net::REC_NET)
    net.v .= net.v0
end

function step!(net::REC_NET, target=nothing)
    u = net.W_rec*r(net) + net.W_in*net.inp
    net.v .+= -net.v/net.tau_m + u
    y = net.W_out*r(net)
    if target !== nothing
        learn!(net, y, target)
    end
    if target === nothing || !net.clamp_target
        net.inp .= y
    else
       net.inp .= target
    end
    return y
end

function episode!(net::REC_NET, target::Array)
    reset_state!(net)
    t_max = size(target)[1]
    y_out = zero(target)
    r_out = zeros(t_max, length(net.v))
    for t=1:t_max
        y_out[t, :] = step!(net, target[t, :])
        r_out[t, :] = r(net)
    end
    return y_out, r_out
end

function episode!(net::REC_NET, steps::Integer)
    reset_state!(net)
    t_max = steps
    y_out = zeros(t_max, size(net.W_out)[1])
    r_out = zeros(t_max, length(net.v))
    for t=1:t_max
        y_out[t, :] = step!(net)
        r_out[t, :] = r(net)
    end
    return y_out, r_out
end


struct ONLINE_ECHO_NET <: REC_NET
    W_in::Matrix{Float64}
    W_rec::Matrix{Float64}
    W_out::Matrix{Float64}
    v::Vector{Float64}
    v0::Vector{Float64}
    inp::Vector{Float64}
    tau_m::Float64
    eta::Float64
    clamp_target::Bool
end

function ONLINE_ECHO_NET(in_size, hidden_size, out_size; tau=10, eta=0.01,
                 g_in=1, g_rec=1, g_out=1, clamp_target=true)
    W_in = g_in*randn(hidden_size, in_size) / sqrt(in_size)
    W_rec = g_rec*randn(hidden_size, hidden_size) / sqrt(hidden_size)
    W_out = g_out*randn(out_size, hidden_size) / sqrt(hidden_size)
    v0 = randn(hidden_size)
    ONLINE_ECHO_NET(W_in, W_rec, W_out, v0, copy(v0), zeros(out_size),
                    tau, eta, clamp_target)
end

function learn!(net::ONLINE_ECHO_NET, y, target)
    err = target-y
    net.W_out .+= net.eta*err*(r(net)')
end
    


struct FORCE_NET <: REC_NET
    W_in::Matrix{Float64}
    W_rec::Matrix{Float64}
    W_out::Matrix{Float64}
    P::Matrix{Float64}
    v::Vector{Float64}
    v0::Vector{Float64}
    inp::Vector{Float64}
    tau_m::Float64
    clamp_target::Bool
end

function FORCE_NET(in_size, hidden_size, out_size; tau=10, g_in=1,
                   g_rec=1, g_out=1, alpha=10, clamp_target=true)
    W_in = g_in*randn(hidden_size, in_size) / sqrt(in_size)
    W_rec = g_rec*randn(hidden_size, hidden_size) / sqrt(hidden_size)
    W_out = g_out*randn(out_size, hidden_size) / sqrt(hidden_size)
    P = (1/alpha)*Matrix{Float64}(I, hidden_size, hidden_size)
    v0 = randn(hidden_size)
    FORCE_NET(W_in, W_rec, W_out, P, v0, copy(v0), zeros(out_size),
              tau, clamp_target)
end

function learn!(net::FORCE_NET, y, target)
    ra = r(net)
    P = net.P
    P .-= P*ra*(ra')*P / (1+(ra')*P*ra)
    err = y-target
    net.W_out .-= err*((P*ra)')
end


struct SANGER_NET <: REC_NET
    W_in::Matrix{Float64}
    W_rec::Matrix{Float64}
    W_middle::Matrix{Float64}
    W_out::Matrix{Float64}
    v::Vector{Float64}
    v0::Vector{Float64}
    inp::Vector{Float64}
    tau_m::Float64
    clamp_target::Bool
end

function SANGER_NET(in_size, hidden_size, middle_size, out_size; tau=10, g_in=1,
                   g_rec=1, g_out=1, clamp_target=true)
    W_in = g_in*randn(hidden_size, in_size) / sqrt(in_size)
    W_rec = g_rec*randn(hidden_size, hidden_size) / sqrt(hidden_size)
    W_middle = g_out*randn(middle_size, hidden_size) / sqrt(hidden_size)
    W_out = g_out*randn(out_size, middle_size) / sqrt(middle_size)
    v0 = randn(hidden_size)
    SANGER_NET(W_in, W_rec, W_middle, W_out, v0, copy(v0), zeros(out_size),
             tau, clamp_target)
end

function step!(net::SANGER_NET, target=nothing)
    u = net.W_rec*r(net) + net.W_in*net.inp
    net.v .+= -net.v/net.tau_m + u
    middle = net.W_middle*r(net)
    y = net.W_out*middle
    if target !== nothing
        net.W_middle .+= 2e-4*(middle*r(net)' - LowerTriangular(middle*middle')*net.W_middle)
        #learn!(net, y, target)
    end
    if target === nothing || !net.clamp_target
        net.inp .= y
    else
       net.inp .= target
    end
    return middle
end