%  ADMM for a LINEAR model: minimize ||Y - X*beta||^2
%  subject to R*beta = d

function [betastar] = ADMM_Algorithm_Linear(Y, X, R, d, rho, tol, maxiter)

    % Initialization
    [~, p] = size(X);
    beta = zeros(p,1);
    lambda = zeros(size(R,1),1);
    
    XtX_rho_RtR=X'*X+rho*(R'*R);
   
    % ADMM loop
    for k=1:maxiter
        beta_prev=beta;
        lambda_prev=lambda;

        % Beta update
        beta=pinv(XtX_rho_RtR)*(X'*Y + rho*R'*(d + lambda/rho));

        % Dual update
        lambda=lambda+rho*(R*beta-d);

        % Convergence checking
        omega_curr=[beta;lambda];
        omega_prev=[beta_prev;lambda_prev];
        diff=norm(omega_curr - omega_prev);

       
        if diff <= tol
            break
        end
    end

    % Output
    betastar = beta;
end
