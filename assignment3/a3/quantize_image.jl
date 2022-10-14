using Images, Plots

# file: file name
# b: Number of clusters (2^b)

# Output: y, W, nRows, nCols
function quantizeImage(file, b)
    # Load image
    I = load(file)
    (nRows,nCols) = size(I)

    # Convert to features
    R = permutedims(channelview(I),[2,3,1])
    X = reshape(float64.(R),(nRows*nCols,3))

    include("kMeans.jl")
    # X represents the data
    # k represents number of clusters
    model = kMeans(X, (2^b))

    y = model.predict(X)
    W = model.W

    return (y, W, nRows, nCols)
end

# y: Cluster assignments
# W: Cluster means
# nRows: Number of rows in the image
# nCols: Number of columns in the image

# Output: Version of image replaced with colours replaced with nearest prototype colour
function deQuantizeImage(y, W, nRows, nCols)
    # Replace each pixel with its nearest prototype colour
    R2 = reshape(W[y,:],(nRows,nCols,3))
    I2 = colorview(RGB,permutedims(R2,[3,1,2]))
    return I2
end

# Do 1, 2, 4, 6
(y, W, nRows, nCols) = quantizeImage("dog.png", 6)
I = deQuantizeImage(y, W, nRows, nCols)

# Show image
plot(I)
