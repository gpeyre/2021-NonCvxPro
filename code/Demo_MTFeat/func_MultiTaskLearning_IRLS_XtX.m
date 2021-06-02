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
function [W,fval_all] = func_MultiTaskLearning_IRLS_XtX(XtX,Xty,lam,opts)
d = size(XtX,1);
T = size(Xty,2);

%initialise with symmetric pos-def trace 1 matrix
D = rand(d,d);
D = D'*D;
% D = D/trace(D);
Dinv = inv(D);
W = zeros(d,T);
eps = opts.eps;
sumv = @(x) sum(x(:));
multieps = 1;
fval_all = [];
while eps>1e-16 && multieps
    
    
    fval = zeros(opts.maxits,1);
    
    for i=1:opts.maxits
        Wold = W;
        fi = 0;
        %min over W
        
        W = (XtX+  lam*Dinv)\Xty;
        
        %min over D
        D1 = W*W' + eps*eye(d);
        [U,S,~] = svd(D1);
        sig = diag(S);
        Dinv = U*diag(1./sqrt(sig))* U';
        
        
        %record function value 1/2|X W - Y|^2 + lam*|W|_*^2
        fval(i) =  0.5*sumv(W.*(XtX*W)) - sumv(W.*Xty) + lam*sum(sqrt(max(sig-eps,0)));
        res = norm(W(:)- Wold(:))/norm(W(:));
        
        if res <opts.tol
            break
        end
    end
    
    fval = fval(1:i);
    eps = eps/10;
    fval_all = [fval_all;fval];
    if ~opts.decrease_eps %only run for the give epsilon value
        multieps = 0;
    end
end
end

