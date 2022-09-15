using HDF5
using DataFrames
using Glob

function mergeDataFrames(pattern, onlyFinal=true)
    data = DataFrame[]
    for fn in glob(pattern)
       h5open(fn, "r") do fid
           for key in keys(fid)
       	        losses = fid[key]
           	 attrs = attributes(fid[key])
            	 d = Dict(k=>read(attrs, k) for k in keys(attrs))
            	 if onlyFinal
                    d["loss"] = losses[end]
            	else
		    d["loss"] = read(losses)
                    d["trial"] = 1:length(d["loss"])
            	end
            	push!(data, DataFrame(d))
            end
	end
    end
    return vcat(data...)
end

function mergeH5s(pattern, merged_name)
    h5open(merged_name, "cw") do m_fid
        for fn in glob(pattern)
	    h5open(fn, "r") do fid
	       for key in keys(fid)
	           println("$(fn) -> $(key)")
	           m_fid[key] = read(fid[key])
		   attrs = attributes(fid[key])
		   for att in keys(attrs)
		       attributes(m_fid[key])[att] = read(attrs, att)
		   end
                end
            end
	end
    end
end
