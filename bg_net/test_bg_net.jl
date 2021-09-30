using Plots
using ProgressBars
include("bg_net.jl")
base_period = 200
target_fcn(t) = 0.25*[sin(2*pi*t/base_period)+0.5sin(4*pi*t/base_period)+0.25sin(8*pi*t/base_period),
                     0.6*cos(2*pi*t/base_period)+1.0sin(4*pi*t/base_period)-0.5sin(8*pi*t/base_period)].+0.5
#inp =[sin.(theta) cos.(theta) -sin.(theta) -cos.(theta)] .+ 1
net = BG_NET(100, 100, 2; eta_str=0e-4, eta_snr=2e-4)
for t=ProgressBar(1:150000)
    step!(net, target_fcn(t), lock_feedback=true)
end
net.eta_str = 0.0
net.eta_snr = 0.0
max_t = 500
log_str_r = zeros(max_t, 100)
log_ctx_r = zeros(max_t, 100)
log_snr_u = zeros(max_t, 2)
log_target =zeros(max_t, 2)
for t=1:max_t
    log_snr_u[t, :] = step!(net, target_fcn(t), lock_feedback=true)
    log_str_r[t, :] = @. 1/(1+exp(-net.v_str))
    log_ctx_r[t, :] = @. 1/(1+exp(-net.v_ctx))
    log_target[t, :] = target_fcn(t)
end

plot_ctx_r = plot(log_ctx_r, ylim=(-0.1, 1.1), legend=false)
plot_str_r = plot(log_str_r, ylim=(-0.1, 1.1), legend=false)
plot_snr_u = plot(log_snr_u, ylim=(-0.1, 1.1), color=[1 2], label="Readout")
plot!(plot_snr_u, log_target, label="Target", color=[1 2], linestyle=:dash)
l = @layout [a;b;c]
plot(plot_ctx_r, plot_str_r, plot_snr_u, layout=l)
