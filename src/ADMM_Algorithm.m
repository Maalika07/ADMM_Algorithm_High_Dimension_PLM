function [beta_star,gamma_star] = ADMM_Algorithm(Y,X,B,R,d,rho,tol,max_iter)
    
    p=size(X,2);    
    q=size(B,2);
    
    % initial values
    beta=zeros(p,1);
    gamma=zeros(q,1);
    lambda=zeros(size(d));
    
    BtB=B'*B;                       
    XtX_rho_RtR=X'*X+rho*(R'*R);        
    
    %ADMM Loop 
    for k=1:max_iter
        beta_prev=beta;
        gamma_prev=gamma;
        lambda_prev=lambda;
    
        % gamma–update
        gamma=pinv(BtB)*(B'*(Y-X*beta));
    
        % beta–update
        beta=pinv(XtX_rho_RtR)*(X'*(Y-B*gamma)+R'*(rho*d-lambda));
    
        % lambda–update (dual variable)
        lambda=lambda+rho*(R*beta-d);
        
        %Convergence
        omega_curr=[beta;gamma;lambda];
        omega_prev=[beta_prev;gamma_prev;lambda_prev];
        diff=norm(omega_curr-omega_prev);

        
        if diff<=tol
            break
        end
    end
    
    beta_star=beta;
    gamma_star=gamma;
end
