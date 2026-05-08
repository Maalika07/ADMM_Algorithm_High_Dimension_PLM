%  ADMM for a LINEAR model: minimize ||Y - B*gamma||^2
% R=[]

function [betastar] = ADMM_Algorithm_Linear_Uncons(Y, X, tol, maxiter)

    % ADMM loop
    [~, p]=size(X);
    beta=zeros(p, 1);
    
    XtX=X'*X;

    % ADMM loop
    for iter=1:maxiter
        betaprev=beta;

        % Beta update
        beta=pinv(XtX)*(X' * Y);

        % Convergence checking
        omegachange=norm(beta - betaprev);
        
        if omegachange<=tol
            break
        end
    end

    % Output
    betastar=beta;
end
