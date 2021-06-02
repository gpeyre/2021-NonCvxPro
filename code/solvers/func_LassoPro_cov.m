% Implements varpro + nonconvex splitting for
%    (*)       min_a  1/2 |X a - y|^2 + lam*|a|_1
%
% Inputs:
%   X and X' are either function handles or matrices
%   lam > 0 is reg parameter
%   y is a vector of length n, where n = size(X,1);
%   n is problem dimension, i.e dim of a in (*)
%   opts are options for lbfgs (see comments inside lbfgsb function),
%            set opts.pos = 1 to enforce positivity
%
% Outputs:
%   x in argmin (*), f and g are the function and gradient values
%
function [x,f,g]  = func_LassoPro_cov(Q,c,lam,opts)



n = size(Q,1);
Efun = @(t) EvalFn(t,Q,lam,c);

t0 = ones(n,1);
[t,f,g] =  func_lbfgs(Efun,t0,opts);

n   = length(t);
a   = (lam*eye(n) + Q.*(t.*t') )\(t.*c);
x   = t.*a;

end

%returns [f;g] where f is the function value, g is the gradient
function fg = EvalFn(t,Q,lam,c)
n   = length(t);
a   = (lam*eye(n) + Q.*(t.*t') )\(t.*c);
x   = t.*a;
v1  = lam*t +a.* (Q*x-c);
f   = lam* norm(a)^2/2 +lam* norm(t)^2/2 + x'*(Q*x)/2 - x'*c;
fg  = [f ;v1];
end





