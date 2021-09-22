using Plots
using ProgressBars
include("bg_net.jl")

target_fcn(t) = [sin(2*pi*t/100)+0.5sin(4*pi*t/100)+0.25sin(8*pi*t/100),
                -sin(2*pi*t/100)+0.5sin(4*pi*t/100)-0.25sin(8*pi*t/100)]
#inp =[sin.(theta) cos.(theta) -sin.(theta) -cos.(theta)] .+ 1
net = BG_NET(100, 100, 2; eta_str=5e-4, eta_snr=5e-4)
for t=ProgressBar(1:100000)
    step!(net, target_fcn(t))
end
net.eta_str = 0.0
net.eta_snr = 0.0
max_t = 500
log_str_r = zeros(max_t, 100)
log_ctx_r = zeros(max_t, 100)
log_snr_u = zeros(max_t, 2)
log_target =zeros(max_t, 2)
for t=1:max_t
    log_snr_u[t, :] = step!(net, target_fcn(t))
    log_str_r[t, :] = @. 1/(1+exp(-net.v_str))
    log_ctx_r[t, :] = @. 1/(1+exp(-net.v_ctx))
    log_target[t, :] = target_fcn(t)
end

plot_ctx_r = plot(log_ctx_r, ylim=(-0.1, 1.1), legend=false)
plot_str_r = plot(log_str_r, ylim=(-0.1, 1.1), legend=false)
plot_snr_u = plot(log_snr_u, ylim=(-1.5, 1.5), color=[1 2], label="Readout")
plot!(plot_snr_u, log_target, label="Target", color=[1 2], linestyle=:dash)
l = @layout [a;b;c]
plot(plot_ctx_r, plot_str_r, plot_snr_u, layout=l)
