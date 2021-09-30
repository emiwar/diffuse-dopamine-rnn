using LinearAlgebra

mutable struct BG_NET
    W_ctxstr::Matrix{Float64}
    W_strstr::Matrix{Float64}
    W_strsnr::Matrix{Float64}
    W_snrctx::Matrix{Float64}
    W_ctxctx::Matrix{Float64}
    DA::Matrix{Float64}
    str_pos::Matrix{Float64}
    tau_m::Float64
    eta_str::Float64
    eta_snr::Float64
    v_str::Vector{Float64}
    v_ctx::Vector{Float64}
    p::Matrix{Float64}
    q::Matrix{Float64}
end

function BG_NET(W_ctxstr, W_strstr, W_strsnr, W_snrctx, W_ctxctx, DA, str_pos, tau_m,
                eta_str, eta_snr)
    n_str, n_ctx = size(W_ctxstr)
    BG_NET(W_ctxstr, W_strstr, W_strsnr, W_snrctx, W_ctxctx, DA, str_pos, tau_m,
           eta_str, eta_snr, zeros(n_str), zeros(n_ctx), zero(W_strstr),
           zero(W_ctxstr))
end

function BG_NET(n_ctx, n_str, n_snr; tau_m=10, eta_str=5e-3, eta_snr=5e-3, lambda=0.1)
    W_ctxstr = 10*rand(n_str, n_ctx) / n_ctx
    W_strstr = -15*rand(n_str, n_str) / n_str
    W_strsnr = -3*(2*rand(n_snr, n_str) .- 1) / sqrt(n_str)
    W_snrctx = -1.0*rand(n_ctx, n_snr)
    W_ctxctx = (1.5*randn(n_ctx, n_ctx) .+ 0.0) / sqrt(n_ctx)
    #for i=1:n_ctx
    #    W_ctxctx[i,i] = 0.0
    #end
    str_pos = rand(n_str, 3)
    DA = draw_DA(n_str, n_snr, str_pos; lambda=lambda)
    DA[end÷2+1:end, :] .*= -1
    BG_NET(W_ctxstr, W_strstr, W_strsnr, W_snrctx, W_ctxctx, DA, str_pos, tau_m, eta_str, eta_snr)
end
function reset_state!(net::BG_NET)
    net.v .= 0.0
    net.p .= 0.0
    net.q .= 0.0
end
function step!(net::BG_NET, target; lock_feedback=false)
    prev_r_str = @. 1/(1+exp(-net.v_str))
    prev_r_ctx = @. 1/(1+exp(-net.v_ctx))
    u_str = net.W_strstr*prev_r_str + net.W_ctxstr*prev_r_ctx
    net.v_str += -net.v_str/net.tau_m + u_str
    #net.s += (1 .- net.s .- prev_h.*prev_h)/(0.5*net.tau_m)
    
    r_str = @. 1/(1+exp(-net.v_str))
    dr_str = @. (1-r_str)*r_str
    net.p += -net.p/net.tau_m + dr_str*prev_r_str'
    net.q += -net.q/net.tau_m + dr_str*prev_r_ctx'
    
    u_snr = net.W_strsnr*r_str .+ 0.5
    net.v_ctx += -net.v_ctx/net.tau_m + net.W_ctxctx*prev_r_ctx .+ 0.5
    if lock_feedback
        net.v_ctx += net.W_snrctx*target
    else
        net.v_ctx += net.W_snrctx*u_snr
    end
    err = target-u_snr
    feedback = net.DA*err
    #feedback = net.W_strsnr'*err
    
    net.W_strsnr += net.eta_snr *    err   *r_str'
    net.W_strstr += net.eta_str * feedback.*net.p
    net.W_ctxstr += net.eta_str * feedback.*net.q
    
    net.W_strstr .= min.(net.W_strstr, 0)
    net.W_ctxstr .= max.(net.W_ctxstr, 0)
    net.W_strsnr[:, 1:end÷2] .= max.(net.W_strsnr[:, 1:end÷2], 0)
    net.W_strsnr[:, end÷2+1:end] .= min.(net.W_strsnr[:, end÷2+1:end], 0)
    return u_snr
end

function draw_DA(n_str, n_snr, str_pos; n_varicosities=10, lambda=0.1)
    DA = zeros(n_str, n_snr)
    for i=1:n_snr
        varicosities_pos = rand(n_varicosities, 3)
        for j=1:n_str, k=1:n_varicosities
            d = norm(str_pos[j, :] - varicosities_pos[k, :])
            DA[j, i] += exp(-d/lambda)
        end
    end
    return DA
end
