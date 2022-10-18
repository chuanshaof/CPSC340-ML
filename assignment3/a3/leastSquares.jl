include("misc.jl")

function leastSquares(X,y)
	# Find regression weights minimizing squared error
	w = (X'X)\(X'y)

	# Make linear prediction function
	predict(Xhat) = Xhat*w

	# Return model
	return GenericModel(predict)
end

function leastSquaresBias(X,y)
	X = [ones(size(X,1)) X]

	# Find regression weights minimizing squared error
	w = (X'X)\(X'y)

	# Make linear prediction function
	function predict(Xhat)
		Xhat = [ones(size(Xhat,1)) Xhat]
		return Xhat*w
	end

	# Return model
	return GenericModel(predict)
end

# x is a vector
# p is the polynomial order
function leastSquaresBasis(x,y,p)
	Z = zeros(length(x),p+1)

	# Creating Z, first row should be ones
	for i = 1:p+1
		Z[:,i] = x.^(i-1)
	end

	# Find regression weights minimizing squared error
	V = (Z'Z)\(Z'y)

	function predict(xhat)
		z = zeros(length(xhat),p+1)
		for i = 1:p+1
			z[:,i] = xhat.^(i-1)
		end
		return z*V
	end	
	
	return GenericModel(predict)
end

function gaussianRBF(X, y, variance)

	function predict(Xhat)
	end
end

function weightedLeastSquares(X, y, v)
	# f'(w) = X'VXw - X'Vy
	
	# Find regression weights minimizing squared error
	V = Diagonal(v)
	w = (X'V*X)\(X'V*y)

	# Make linear prediction function
	predict(Xhat) = Xhat*w

	# Return model
	return GenericModel(predict)
end
