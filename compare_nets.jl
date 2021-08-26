using Plots
include("dale_net.jl")
include("rpe_net.jl")
include("dopamine_net.jl")

t_max = 100
theta = range(0,2pi;length=t_max)
y_target = @. sin(theta)+0.5sin(2theta)+0.25sin(4theta)
inp =[sin.(theta) cos.(theta) -sin.(theta) -cos.(theta)] .+ 1

nets = [RPE_NET(4, 100, 1; eta_in=1e-3, eta_rec=1e-3, eta_out=1e-3),
        DOPAMINE_NET(4, 100, 1; eta_in=0.0, eta_rec=0.0, eta_out=1e-3),
        DOPAMINE_NET(4, 100, 1; eta_in=1e-3, eta_rec=1e-3, eta_out=1e-3),
        DALE_NET(4, 100, 1; eta_in=1e-3, eta_rec=1e-3, eta_out=1e-3)]

losses = [Float64[] for _=1:4]
n_episodes = 1200
for i=1:4
    for ep=ProgressBar(1:n_episodes)
        y, h = episode!(nets[i], inp, y_target)
        push!(losses[i], sum((y.-y_target).^2)/t_max)
    end
    
end
labels = ["Scalar dopamine" "No dopamine" "Vector dopamine" "Ideal (non-local)"]
l = @layout [a{0.6w} [b;c;d;e]]
p = plot(layout=l, fontfamily="arial")
plot!(p[1], losses, xlim=(1, n_episodes), ylim=(1e-4, 1e0),
     xticks=[1,div(n_episodes, 2),n_episodes], yaxis=:log,
     labels=labels, xlabel="Episode", ylabel="Squared loss")
for i=1:4
    plot!(p[i+1], y_target, legend=false, linestyle=:dash, color=:black)
    y, h = episode!(nets[i], inp, y_target)
    plot!(p[i+1], y, color=i)
end
plot!(p[2], title="Final output")
plot!(p[5], xlabel="Time")