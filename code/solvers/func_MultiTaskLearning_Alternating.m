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
function [W,fval_all] = func_MultiTaskLearning_Alternating(X,y,indx,lam,opts)
d = size(X,1);
T = length(indx);
[XtX, Xty] = split(X,y,indx);

%initialise with symmetric pos-def trace 1 matrix
D = rand(d,d);
D = D'*D; 
D = D/trace(D);
Dinv = inv(D);
% Dinv = d*eye(d);
W = zeros(d,T);
eps = opts.eps;
% eps = opts.eps;

multieps = 1;
fval_all = [];
while eps>1e-16 && multieps


    if ~opts.decrease_eps %only run for the give epsilon value
        multieps = 0;
    end
fval = zeros(opts.maxits,1);

for i=1:opts.maxits
    Wold = W;
    fi = 0;
    %min over W
    for t=1:T
        Q = XtX{t};
        z = Xty{t};
        wt  = (Q +  2*lam*Dinv)\z;
        W(:,t) = wt;
        fi = fi  +wt'*Q*wt/2 - wt'*z;  
    end
    
    %min over D
    D1 = W*W' + eps*eye(d);
    tr2 = sqrt(trace(D1));
    [U,S,~] = svd(D1);
    sig = diag(S);
    D2 = U*diag(sqrt(sig))*U'; 
    D = D2/tr2;    
    Dinv = inv(D);
%     tmp = 1./sqrt(sig);
%     Dinv = tr2*U*diag(tmp)* U';
%     fval(i) = fi + lam*trace(W'*Dinv*W) + lam*eps*trace(Dinv);

%record function value 1/2|X W - Y|^2 + lam*|W|_*^2
%     fval(i) = fi + lam*sum(sqrt(max(sig-eps,0)))^2; 
    fval(i) = fi + lam*norm(svd(W),1)^2; 
    res = norm(W(:)- Wold(:))/norm(W(:));

    if res <opts.tol
        break
    end
end

fval = fval(1:i);
eps = eps/10;
fval_all = [fval_all;fval];
end
eps*trace(Dinv)
end


function [XtX, Xty] = split(X,y,indx)
T = length(indx);
m = size(X,2);
indx(T+1) = m+1;

for t=1:T
    i1 = indx(t);
    i2 = indx(t+1)-1; 
    Xt = X(:, i1:i2);
    yt = y(i1:i2);
    XtX{t} = Xt*Xt';
    Xty{t} = Xt*yt;
end
end
