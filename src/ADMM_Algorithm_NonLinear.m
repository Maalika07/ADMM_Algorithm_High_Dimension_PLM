function [betastar, gammastar] = ADMM_Algorithm_NonLinear(Y, B, R, d, rho, tol, maxiter)


    % Initialization
    [~, q]=size(B);
    gamma=zeros(q, 1);
    lambda=zeros(size(R,1), 1);
    
   %ADMM Loop 
    for iter=1:maxiter
        gammaprev=gamma;
        lambdaprev=lambda;
        
        % Gamma update 
        gamma=pinv(B'*B+rho*(R'*R))*(B'*Y+R'*(rho*d-lambda));
        
        % Dual update
        lambda=lambda+rho*(R*gamma-d);
        
        % Convergence checking
        omegacurr=[gamma;lambda];
        omegaprev=[gammaprev; lambdaprev];
        omegachange=norm(omegacurr - omegaprev);
        
        if omegachange <= tol
            break
        end


    end

    % Output
    betastar = [];
    gammastar = gamma;
end
