using Printf
include("misc.jl")
include("findMin.jl")

function robustRegression(X,y)

	(n,d) = size(X)

	# Initial guess
	w = zeros(d,1)

	# Function we're going to minimize (and that computes gradient)
	funObj(w) = robustRegressionObj(w,X,y)

	# This is how you compute the function and gradient:
	(f,g) = funObj(w)

	# Derivative check that the gradient code is correct:
	g2 = numGrad(funObj,w)

	if maximum(abs.(g-g2)) > 1e-4
		@printf("User and numerical derivatives differ:\n")
		@show([g g2])
	else
		@printf("User and numerical derivatives agree\n")
	end

	# Solve least squares problem
	w = findMin(funObj,w)

	# Make linear prediction function
	predict(Xhat) = Xhat*w

	# Return model
	return GenericModel(predict)
end

function robustRegressionObj(w,X,y)
	epsilon = 1
	
	# Unsure what the objective function should be here
	f = 0
	
	g = zeros(size(w))

	for j in 1:size(w)[1]
		for i in 1:size(X)[1]
			r_i = w'[1] * X[i] - y[i]

			if abs(r_i) < epsilon
				g[j] += r_i * X[i][j]
			elseif r_i < 0
				g[j] += -epsilon * X[i][j]
			else
				g[j] += epsilon * X[i][j]
			end

			if abs(r_i) < epsilon
				f += 1/2 * r_i^2
			else
				f += epsilon * (abs(r_i) - epsilon/2)
			end
		end
	end
	
	return (f,g)
end

