using JSON

n_item = 3
n_trial = 10000

trials = map(1:n_trial) do i
    values = rand(1:9, n_item)
    fixations = mapreduce(vcat, enumerate(values)) do (i, v)
        (i * ones(Int, v)) .- 1
    end
    choice = argmax(values) - 1
    (;values, fixations, choice)
end

# mkpath("simulated_data")
write("simulated_data/train_data_v1.0.json",  json(trials))