% Inputs:
% indx  = starting index for each task
% X     = training data matrix, size d by m
% y     = training output, size m by 1
% lam   = regularisation parameter
% opts  = opts.maxits (maximum number of iterations),
%         opts.tol (stop when relative residue drops below tol)
%         opts.eps (regularisation, barrier to keep D from boundary of S^+)
%
% Outputs:
% W     = regression coefficients
% fval  = objective values
function [W,fval] = func_MTL_fista_XtX(XtX,Xty,lam,opts)
L = normest(XtX);
d = size(XtX,1);
T = size(Xty,2);
W = zeros(d,T);
maxits = opts.maxits;
fval = zeros(maxits,1);
gamma = 1/L;
U = W;
theta = 1;
sumv = @(x) sum(x(:));
for i = 1:maxits
    Wm = W;
    
    if opts.recordobj
        fval(i) =  0.5*sumv(U.*(XtX*U)) - sumv(U.*Xty) +lam*sum(svd(U));
        
    end
    
    W  = U - gamma*(XtX*U - Xty);
    
    [L,S,V] = svd(W);
    sig = wthresh(diag(S), 's', gamma*lam);
    S2 = spdiags(sig,0,d,T);
    W = L*S2*V';  
    
    theta = (1+sqrt(1+4*theta^2))/2;
    Um = U;
    U = W + (theta-1)/theta*(W-Wm);
    
    res = norm(W(:)- Wm(:))/norm(W(:));
    
    if res <opts.tol
        break
    end
    
    if (Um(:)-W(:))'*(W(:)-Wm(:)) > 0
        U = W;
        theta = 1;
    end
    
    
    
end

fval = fval(1:i);

end
