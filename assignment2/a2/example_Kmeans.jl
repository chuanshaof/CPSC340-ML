# Load data
using JLD
X = load("clusterData.jld","X")

# K-means clustering
k = 4
include("kMeans.jl")
# model = kMeans(X,k,doPlot=false)
# y = model.predict(X)

global bestModel = kMeans(X,k,doPlot=false)
global besty = bestModel.predict(X)
global bestScore = kMeansError(X, besty, bestModel.W)

@show(bestScore)

# for i in 1:49
#     model = kMeans(X,k,doPlot=false)
#     y = model.predict(X)
#     score = kMeansError(X, y, model.W)
#     if score < bestScore
#         global bestScore = score
#         global bestModel = model
#     end
# end

# include("clustering2Dplot.jl")
# clustering2Dplot(X,y,model.W)
