include("experiment.jl")



function run_experiment_with_lin_ind(fn; params...)
    param_ranges = NamedTuple(params)
    git_commit = readchomp(`git rev-parse --short HEAD`)
    for (i, values) in enumerate(product(param_ranges...))
        named_params = NamedTuple{keys(param_ranges)}(values)
        if named_params.striatumUpdate == :ideal
            named_params = merge(named_params, (;feedback_factor = named_params.feedback_factor*10.0))
        end
        all_params = merge(DEFAULT_PARAMS, named_params)
        net, losses, angles = train_network("$named_params "; all_params...)
        fb_mat = vcat(-net.feedback_dmsn, net.feedback_imsn)
        feedback_eig1 = eigvals(fb_mat' * fb_mat)[1]
        h5open(fn, "cw") do fid
            fid["run$i"] = losses
            for k in keys(all_params)
                v = all_params[k]
                if v isa Number
                    attributes(fid["run$i"])[string(k)] = v
                else
                    attributes(fid["run$i"])[string(k)] = string(v)
                end
            end
            attributes(fid["run$i"])["git"] = git_commit
            attributes(fid["run$i"])["feedback_eig1"] = feedback_eig1
        end
    end
end
