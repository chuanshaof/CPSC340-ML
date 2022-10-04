include("decisionStump.jl")

function decisionTree(X,y,depth)
	# Fits a decision tree using greedy recursive splitting
	# (uses recursion to make the code simpler)

	(n,d) = size(X)

	# Learn a decision stump
	splitModel = decisionStump(X,y)

	if depth <= 1 || splitModel.baseSplit
		# Base cases where we stop splitting:
		# - this stump gets us to the max depth
		# - this stump doesn't split the data
		return splitModel
	else
		# Use the decision stump to split the data
		yes = splitModel.split(X)

		# Recusively fit a decision tree to each split
		yesModel = decisionTree(X[yes,:],y[yes],depth-1)
		noModel = decisionTree(X[.!yes,:],y[.!yes],depth-1)
		
		# Make a predict function
		function predict(Xhat)
			(t,d) = size(Xhat)
			yhat = zeros(t)

			yes = splitModel.split(Xhat)

			yhat[yes] = yesModel.predict(Xhat[yes,:])
			yhat[.!yes] = noModel.predict(Xhat[.!yes,:])
			return yhat
		end

		return GenericModel(predict)
	end
end

function depth2DecisionTree(X, y)
	# Split from the top
	(n,d) = size(X)

	# Top level splitting
	splitModel = decisionStump(X,y)
	
	# yes represents where the model will be split at the top level
	yes = splitModel.split(X)

	# Bottom level splitting
	yesModel = decisionStump(X[yes,:],y[yes])
	noModel = decisionStump(X[.!yes,:],y[.!yes])

	# Combining both top and bottom level to return a GenericModel
	function predict(Xhat)
		(t,d) = size(Xhat)
		yhat = zeros(t)

		yes = splitModel.split(Xhat)

		yhat[yes] = yesModel.predict(Xhat[yes,:])
		yhat[.!yes] = noModel.predict(Xhat[.!yes,:])
		return yhat
	end

	return GenericModel(predict)
end