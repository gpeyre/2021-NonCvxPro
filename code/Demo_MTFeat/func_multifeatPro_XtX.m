% Inputs:
% indx  = starting index for each task
% X     = training data matrix, size d by m
% y     = training output, size m by 1
% lam   = regularisation parameter
%
% Outputs:
% W     = regression coefficients
% fval  = objective values

function [W,f] = func_multifeatPro_XtX(XtX,Xty,lam,opts)
n = size(XtX,1);

Efun = @(V) EvalFn(V,XtX,lam,Xty,n);

t0 = randn(n^2,1);
[t,f,~] =  func_lbfgs(Efun,t0(:),opts);

V = reshape(t,n,n);
U = (lam*eye(n)+ V'*XtX*V)\ (V'*Xty);

W = V*U;


end



%returns [f;g] where f is the function value, g is the gradient
function fg = EvalFn(V,XtX,lam,Xty,n)
sumv = @(x) sum(x(:));
V = reshape(V,n,n);
U = (lam*eye(n)+ V'*XtX*V)\ (V'*Xty);
VU = V*U;
grad = lam*V + (XtX*VU - Xty)*U';

fval = lam/2*norm(V,'fro')^2 + lam/2*norm(U,'fro')^2 ...
    + 0.5*sumv(VU.*(XtX*VU)) - sumv(VU.*Xty);


fg = [fval;grad(:)];
end
