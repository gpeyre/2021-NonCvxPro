% Implements varpro + nonconvex splitting for
%    (*)       min_A  1/2 |X A - Y|^2 + lam*sum_i |Ai|_{2}
%  where Ai is ith row of A
% Inputs:
%   X       = n by p matrix
%   Y       = n by q matrix
%   lam     = regularisation parameter
%   opts    = LBFGS options
% Outputs:
%   A       = p by q matrix
%   f       = function values
%

function [W,f] = func_MultiTaskLassoPro(X,Y,lam,opts)
[n,p] = size(X);

Efun = @(t) EvalFn(t,X,lam,Y);
t0 = rand(p,1);
[t,f,~] =  func_lbfgs(Efun,t0,opts);
M = eye(n)+1/lam*X*(t.^2.*X');
alpha = -M\Y;
W = -1/lam*t.^2.*(X'*alpha);
end

%returns [f;g] where f is the function value, g is the gradient
function fg = EvalFn(t,X,lam,Y)
[n,~] = size(Y);
M = eye(n)+1/lam*X*(t.^2.*X');
alpha = -M\Y;
Xta = X'*alpha;
v = t.* sum(Xta.^2,2);
grad = lam*t - 1/lam*v;

fval = -1/2*norm(alpha,'fro')^2 ...
    - 1/2/lam*norm(t.*Xta, 'fro')^2+lam/2*norm(t)^2 - sum(alpha(:).*Y(:));
fg = [fval;grad];
end
