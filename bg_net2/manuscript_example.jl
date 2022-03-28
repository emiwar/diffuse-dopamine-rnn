include("experiment.jl")
git_commit = readchomp(`git rev-parse --short HEAD`)

function save_log(filename, hdfKey, log; attrs...)
    h5open(filename, "cw") do fid
        for pop in keys(log)
            fid[hdfKey * "/" * string(pop)] = log[pop]
        end
        for k in keys(attrs)
            if v isa Number
                attributes(fid[hdfKey])[string(k)] = v
            else
                attributes(fid[hdfKey])[string(k)] = string(v)
            end
        end
    end
end

net = BgNet(200, 4, 1e-3, 2.5e-2)
input = create_input(size(net[:thal]), 200)
target = 0.5 .+ 0.15*gaussianProcessTarget(200, 4, 20)
input_fcn(t) = input[t, :]
target_fcn(t) = target[t, :]
log = recordSampleRun(net, 200, clamp=(thal=input_fcn,))
#save_log("data/exampleRun.h5", "warmup", log)
log = recordSampleRun(net, 200, clamp=(thal=input_fcn,))
#save_log("data/exampleRun.h5", "first_trial", log)
losses = Float64[]
@showprogress for trial_id=1:5000
    loss = run_trial(net, target, input, :dopamine)
    push!(losses, loss)
end
log = recordSampleRun(net, 200, clamp=(thal=input_fcn,))
#save_log("data/exampleRun.h5", "last_trial", log; n_trials=5000, 
#         learning_rate=1e-3, net_size=200, ntrial_length=200, git=git_commit,
#         striatumUpdate=:dopamine)

h5open("data/exampleRun.h5", "cw") do fid
    fid["losses"] = losses
    fid["input"] = collect(input)
    fid["target"] = target
end
