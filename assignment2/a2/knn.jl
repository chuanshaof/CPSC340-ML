include("misc.jl") # Includes GenericModel typedef

function knn_predict(Xhat,X,y,k)
  (n,d) = size(X)
  (t,d) = size(Xhat)

  # X is the training features data
  # y is the training target data
  # k = number of neighbours
  # Xhat is the test features

  # Note that distance square rooted or not does not make a difference
  distances = distancesSquared(Xhat,X)

  # @show(distances)

  # Outcome of this distance squared for a 2 feature matrix would be
  # a (t x n) matrix that contains
    # [d11 d12 ... d1n]
    # [d21 d22 ... d2n]
    # [... ... ... ...]
    # [dt1 dt2 ... dtn]
  # dtn represents the distance from Xhat_t to X_n

  # sortrows(distances)
  
  k = min(n,k) # To save you some debuggin

  # Find the k smallest distances for each row
  nearest_neighbours = fill(0, t, k)

  for i in 1:t
    nearest_distances = fill(Inf, k)
    for j in 1:n
      furthest_nearest_distance = maximum(nearest_distances)

      if distances[i, j] < furthest_nearest_distance
        # @show(findmax(nearest_distances)[2])
        nearest_neighbours[i, findmax(nearest_distances)[2]] = j
        nearest_distances[findmax(nearest_distances)[2]] = distances[i, j]
      end
    end

    # Convert distances to target values
    for j in 1:k
      nearest_neighbours[i, j] = y[nearest_neighbours[i, j]]
    end
  end

  # Find the most common class in the k nearest neighbours
  yhat = fill(0, t)
  for i in 1:t
    yhat[i] = mode(nearest_neighbours[i, :]) 
  end

  return yhat
end

function knn(X,y,k)
	# Implementation of k-nearest neighbour classifier
  predict(Xhat) = knn_predict(Xhat,X,y,k)
  return GenericModel(predict)
end
