using Plots
#include("../dendritic_rls/reference_force.jl")
include("rflo.jl")


t_max = 100
theta = range(0,2pi;length=t_max)
y_target = @. sin(theta)+0.5sin(2theta)+0.25sin(4theta)

actor_net  = RFLO_NET(1, 50, 1; eta_in=1e-2, eta_rec=1e-2, eta_out=1e-2)
critic_net = RFLO_NET(1, 50, 3; eta_in=1e-4, eta_rec=1e-4, eta_out=1e-4)

function ep!(actor_net, critic_net, y_target)
    actions = zero(y_target)
    coeffs = zeros(size(y_target, 1), 3)
    a = 1 .- 2*rand(1)
    reset_state!(critic_net)
    for t=1:t_max
        a = step!(actor_net, a)
        actions[t, :] = a
        dist = sum((a - y_target[t, :]).^2)
        #learn!(actor_net, a - y_target[t, :])
        
        c = step!(critic_net, [1.0])
        coeffs[t, :] = c
        L = (c[1]*a[1])^2 + c[2]*a[1] + c[3] - dist
        err = L*[2*c[1]*a[1].^2, a[1], 1]
        learn!(critic_net, err)
    end
    return actions, coeffs
end

function fake_ep!(actor_net, coeffs, y_target, P)
    actions = zero(y_target)
    a = 1 .- 2*rand(1)
    reset_state!(critic_net)
    for t=1:t_max
        a = [4*rand()-2]#step!(actor_net, a)
        actions[t, :] = a
        dist = sum((a - y_target[t, :]).^2)
        #learn!(actor_net, a - y_target[t, :])
        
        c = coeffs[t, :]
        #bases = [a[1]^2, a[1], 1]
        #P = Ps[t, :, :]
        #Ps[t,:,:] .-= P*bases*(bases')*P / (1+bases'*P*bases)
        #L = c'*bases - dist
        #coeffs[t, :] .-= L*(Ps[t,:,:]*bases)
        
        #P .-= P*bases*(bases')*P / (1+bases'*P*bases)
        #L = c'*bases - dist
        #coeffs[t, :] .-= L*(P*bases)
        L = (c[1]*a[1])^2 + c[2]*a[1] + c[3] - dist
        #err = L*[a[1].^2, a[1], 1]
        coeffs[t, :] .-= 0.01*L*[2*c[1]*a[1]^2, a[1], 1]
        #learn!(critic_net, err)
    end
    return actions
end

actions, coeffs = ep!(actor_net, critic_net, y_target)

@gif for t=1:100
    P = [10.0*(a==b) for a=1:3, b=1:3]
    actions = fake_ep!(actor_net, coeffs, y_target, P)
    val_map(t, y) = (coeffs[t, 1]*y)^2+coeffs[t, 2]*y + coeffs[t, 3]
    p = heatmap(1:t_max, -1.5:0.01:1.5, val_map, clim=(-2, 2))
    #plot!(p, actions, ylim=(-1.5, 1.5), color=1, lw=2)
    plot!(p, y_target, ylim=(-1.5, 1.5), color=3, lw=2)
    plot!(p, -coeffs[:, 2] ./ (2 .* coeffs[:, 1]), color=1, lw=2, linestyle=:dash)
end