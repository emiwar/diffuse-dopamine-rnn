using Plots
using LinearAlgebra

c = rand(3)
points = zeros(0, 2)
@gif for t=1:100
    x = 2*rand()-1
    dist = (x-0.73)^2
    L = (c[1]*x)^2 + c[2]*x + c[3] - dist
    err = L*[2*c[1]*x.^2, x, 1]
    c .-= 0.1*err
    global points
    points = [points; [x (x-0.73)^2]]
    plot(-1:0.01:1, map(x->c[1]*x^2+c[2]*x+c[3], -1:0.01:1))
    plot!(-1:0.01:1, map(x->(x-0.73)^2, -1:0.01:1), label="target")
    scatter!(points[:,1], points[:,2])
end

c = rand(3)
points = zeros(0, 2)
P = collect(Float64, (1/0.01)*I(3))
x_last = 2*rand()-1
@gif for t=1:100
    global points
    x = 2*(t/100)-1 + 0.01*rand()
    #x = 2*rand()-1
    #x = 0.9*x_last-0.1*c[2]/(2*c[1])
    global x_last
    x_last = x
    #x = -c[2]/(2*c[1])
    dist = (x-0.23)^2
    bases = [x^2, x, 1]
    #bases = [(x+1)^2, x^2, (x-1)^2]
    P .-= P*bases*(bases')*P / (1+bases'*P*bases)
    L = c'*bases - dist
    c .-= L*(P*bases)
    #c .-= 0.05*L*bases
    points = [points; [x (x-0.23)^2]]
    plot(-1:0.01:1, map(x->c'*[x^2, x, 1], -1:0.01:1))
    plot!(-1:0.01:1, map(x->(x-0.23)^2, -1:0.01:1), label="target")
    scatter!(points[:,1], points[:,2])
end
