using JLD
using SparseArrays

data = load("newsgroups.jld")
X = data["X"]
y = data["y"]
Xtest = data["Xtest"]
ytest = data["ytest"]
wordlist = data["wordlist"]
groupnames = data["groupnames"]

# @show(wordlist[50])
# @show(wordlist)
# @show(groupnames)

# @show(X[500, :])
# @show(wordlist[6], wordlist[24], wordlist[25], wordlist[70], wordlist[88])

# @show(y[500, :])

# rows = rowvals(X)

# for j in 1:size(X)[1]
#     @show(j)
#     if rows[j] == 500
#         @show(X[j, :])
#     end
# end