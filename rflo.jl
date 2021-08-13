using Plots

n_in = 1
n_hidden = 50
n_out = 1

tau_m = 10

W_in = (2*rand(n_hidden, n_in) .- 1)
W_rec = 5*randn(n_hidden, n_hidden) / sqrt(n_hidden)
W_out = 2*(2*rand(n_out, n_hidden) .- 1) / sqrt(n_hidden)
eta_out = 0.5
eta_rec = 0.5
eta_in = 0.05


phi(u) = 1/(1+exp(-u))
t_max = 100
theta = range(0,2pi;length=t_max)
y_target = @. sin(theta)+0.5sin(2theta)+0.25sin(4theta)
#plt = plot(0:0, ylim=(-1.5,1.5))
@gif for ep=1:300
    global W_out, W_rec, W_in
    y = zeros(t_max)
    h = zeros(t_max, n_hidden)
    x = ones(t_max)
    p = zeros(n_hidden, n_hidden)
    q = zeros(n_hidden, n_in)
    for t=2:t_max
        f = 2*phi.(W_rec*h[t-1,:] + W_in*x[t, :]).-1
        df = (1 .- f.*f)./2
        
        h[t,:] = h[t-1,:] + (-h[t-1,:] + f)/tau_m
        y[t,:] = W_out*h[t, :]
        err = y_target[t, :]-y[t,:]
        
        p += (-p + df*h[t, :]')/tau_m 
        q += (-q + df*x[t, :]')/tau_m
        
        feedback = W_out'*err
        
        dW_out = err*h[t,:]'
        dW_rec = feedback.*p
        dW_in = feedback.*q
        
        W_out += eta_out*dW_out/t_max
        W_rec += eta_rec.*dW_rec/t_max
        W_in += eta_in.*dW_in/t_max
    end
    plot(y, label="y", ylim=(-1.5, 1.5), title="Episode $ep")
    plot!(y_target, label="y_target")
end