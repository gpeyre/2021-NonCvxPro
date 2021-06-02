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
function [W,fval] = func_MTL_fista(X,y,indx,lam,opts)
d = size(X,1);
T = length(indx);
[XtX, Xty,L] = split(X,y,indx);
W = zeros(d,T);
maxits = opts.maxits;
fi = 0;
fval = zeros(maxits,1);
gamma = 1/L;
U = W;
theta = 1;
for i = 1:maxits
    Wm = W;
    
    if opts.recordobj
        fi = lam*sum(svd(U));
    end
    
    for t=1:T
        ut = U(:,t);
        Q = XtX{t};
        z = Xty{t};
        fi = fi  +ut'*Q*ut/2 - ut'*z;
        
        ut  = ut - gamma*(Q*ut - z);
        W(:,t) = ut;
    end
    fval(i) = fi;
    
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

function [XtX, Xty,L] = split(X,y,indx)
T = length(indx);
m = size(X,2);
indx(T+1) = m+1;
L = 0;
for t=1:T
    i1 = indx(t);
    i2 = indx(t+1)-1;
    Xt = X(:, i1:i2);
    yt = y(i1:i2);
    Q = Xt*Xt';
    XtX{t} = Q;
    Xty{t} = Xt*yt;
    L = max(normest(Q),L);
end
end



