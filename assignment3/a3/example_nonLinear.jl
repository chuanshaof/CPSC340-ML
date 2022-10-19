using Printf
using Statistics

# Load X and y variable
using JLD
data = load("basisData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Fit a least squares model
include("leastSquares.jl")

# For single iteration regressions
# --------------------------------------------
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


# For multiple iterations regressions
# --------------------------------------------
# trainList = []
# testList = []

# for p in 1:10
#     model = leastSquaresBasis(X,y,p)

#     # Evaluate training error
#     yhat = model.predict(X)
#     trainError = mean((yhat - y).^2)
#     @printf("Squared train Error of p=%d with least squares: %.3f\n", p, trainError)
#     push!(trainList, trainError)

#     # Evaluate test error
#     yhat = model.predict(Xtest)
#     testError = mean((yhat - ytest).^2)
#     @printf("Squared test Error of p=%d with least squares: %.3f\n", p, testError)
#     push!(testList, testError)
# end

# using Plots
# plot(1:10, trainList, label="train")
# plot!(1:10, testList, label="test")

function guassianTransform(X, Xtilde, sigma)
    # Create Z
    Z = zeros(size(Xtilde)[1], size(X)[1])
    
    # Calculate Z
    for i = 1:size(Xtilde)[1]
        for j = 1:size(X)[1]
            Z[i,j] = exp(-((Xtilde[i] - X[j])^2)/(sigma^2))
        end
    end

    return Z
end

# For GuassianRBF
# ---------------------------------------------
sigmas = [0.1, 1, 10]

for sigma in sigmas
    local Z = guassianTransform(X, X, sigma)
    local Ztest = guassianTransform(X, Xtest, sigma)
    local model = leastSquares(Z,y)

    # Evaluate training error
    yhat = model.predict(Z)
    trainError = mean((yhat - y).^2)
    @printf("Squared train Error with gaussianRBF of sigma %.1f: %.3f\n", sigma, trainError)

    # Evaluate test error
    yhat = model.predict(Ztest)
    testError = mean((yhat - ytest).^2)
    @printf("Squared test Error with gaussianRBF of sigma %.1f: %.3f\n", sigma, testError)

    # # Plot model
    # using Plots
    # scatter(X,y,legend=false,linestyle=:dot)
    # Xhat = minimum(X):.1:maximum(X)
    # yhat = model.predict(Xhat)
    # plot!(Xhat,yhat,legend=false)
end
