include("misc.jl") # Includes "mode" function

using DelimitedFiles
dataTable = readdlm("fluTrends.csv", ',')

X = real(dataTable[2:end,:])

@show minimum(X)
@show maximum(X)

using Statistics
@show mean(X)
@show median(X)

@show mode(X)