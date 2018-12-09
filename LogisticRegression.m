function [w,b] = LogisticRegression(X, Y, iterations, learning_rate)
    [dim,NumSets] = size(X);
    [w,b] = HelperFunc.initializeweight(dim);
    [w,b,dw,db,J] = HelperFunc.optimize(w, b, X, Y, iterations, learning_rate);
end