using Printf
using Statistics

# Load X and y variable
using JLD
data = load("basisData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Fit a least squares model
include("leastSquares.jl")
# model = leastSquares(X,y)
# model = leastSquaresBias(X,y)

# # Evaluate training error
# yhat = model.predict(X)
# trainError = mean((yhat - y).^2)
# @printf("Squared train Error with least squares: %.3f\n",trainError)

# # Evaluate test error
# yhat = model.predict(Xtest)
# testError = mean((yhat - ytest).^2)
# @printf("Squared test Error with least squares: %.3f\n",testError)

# # Plot model
# using Plots
# scatter(X,y,legend=false,linestyle=:dot)
# Xhat = minimum(X):.1:maximum(X)
# yhat = model.predict(Xhat)
# plot!(Xhat,yhat,legend=false)
# gui()

trainList = []
testList = []

for p in 1:10
    model = leastSquaresBasis(X,y,p)

    # Evaluate training error
    yhat = model.predict(X)
    trainError = mean((yhat - y).^2)
    @printf("Squared train Error of p=%d with least squares: %.3f\n", p, trainError)
    push!(trainList, trainError)

    # Evaluate test error
    yhat = model.predict(Xtest)
    testError = mean((yhat - ytest).^2)
    @printf("Squared test Error of p=%d with least squares: %.3f\n", p, testError)
    push!(testList, testError)
end

using Plots
plot(1:10, trainList, label="train")
plot!(1:10, testList, label="test")

