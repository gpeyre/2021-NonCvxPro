% Solves min_a 1/2 |X a - y|^2 + lam*|a|_1
% where a in R^n,
% implements varpro + eta trick
function [x_lbfgs,f_lbfgs,g_lbfgs]  = func_QuadVar_Cov(Q,X,y,lam,opts)
if nargin<6
    opts = struct();
end
opts.pos = 1;

n = size(Q,1);
z = (X'*y);
Efun = @(t) EvalFn(t,Q,X,lam,y,z);

t0 = abs(randn(n,1));

[t,f_lbfgs,g_lbfgs] =  func_lbfgs(Efun,t0,opts);

st = sqrt(t);
Iz = (lam*eye(n)+Q.*sqrt(t*t'))\(st.*z);
alpha =y - X*(st.*Iz);
x_lbfgs = t.*(X'*alpha)/lam;


end

%returns [f;g] where f is the function value, g is the gradient
function fg = EvalFn(t,Q,X,lam,y,z)
st = sqrt(t);
n = size(Q,1);
Iz = (lam*eye(n)+Q.*(st*st'))\(st.*z);
alpha =y - X*(st.*Iz);

u = X'*alpha;
g = -1/2*abs(u).^2/lam + lam/2;
f = -norm(alpha)^2/2-1/2* sum(t.*(u).^2)/lam +sum(y.*alpha) + sum(t)*lam/2;
fg = [f;g ];



end



