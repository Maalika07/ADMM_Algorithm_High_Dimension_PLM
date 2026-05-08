function d = degreeNL(x,y)

    x = (x - mean(x)) / std(x);          
    y = y - mean(y);                     % centre response
    
    % straight-line fit
    L    = [ones(length(x),1)  x];
    mseL = mean((y - L*(pinv(L)*y)).^2);
    
    % cubic-spline fit 
    k     = 3;
    knots=[repmat(min(x),1,k),quantile(x,[.05 .20 .35 .50 .65 .80 .95]),repmat(max(x),1,k)];
    S=[ones(length(x),1) bspline_basismatrix(k,knots,x)];
    mseS =mean((y - S*(pinv(S)*y)).^2);
    
    d = max(0,(mseL - mseS)/mseL);
end
