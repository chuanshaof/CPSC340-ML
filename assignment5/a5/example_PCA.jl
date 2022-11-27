using DelimitedFiles

# Load data
dataTable = readdlm("animals.csv",',')
X = float(real(dataTable[2:end,2:end]))
(n,d) = size(X)

include("PCA.jl")
model = PCA(X, 2)
Z = model.compress(X)

# Plot matrix as image
using Plots
graph = scatter(Z[:,1], Z[:,2], label="PCA", legend=:topleft)

for i in 1:n
    annotate!(Z[i,1],Z[i,2],dataTable[i+1,1],annotationfontsize=8)
end

plot(graph)
    
