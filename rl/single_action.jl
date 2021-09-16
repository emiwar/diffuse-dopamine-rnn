using Plots
using LinearAlgebra

function circ(x, y, r)
    plot!(x.+r.*cos.(0:0.1:2pi), y.+r.*sin.(0:0.1:2pi))
end

plot(xlim=(0, 2))
plot(ylim=(0, 2))
true_pos = [0.3, 0.4]
guess = [1.0, 1.0]
dist = sum((true_pos-guess).^2)
circ(guess[1], guess[2], sqrt(dist))
guess = [0.7, 0.2]
dist = sum((true_pos-guess).^2)
circ(guess[1], guess[2], sqrt(dist))

#plot!()