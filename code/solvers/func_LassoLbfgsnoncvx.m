% l-bfgs to solve min_{t,a} 0.5*|X(t.*a) - y|^2 +lam/2*( |t|^2 + |a|^2 )
% simultaneously optimise over t and a
function [x,f,g]  = func_LassoLbfgsnoncvx(X,y,lam,opts)
if nargin<4
    opts = struct();
end
n = size(X,2);

Xt = X';
% A = X'*X;
z = X'*y;


Efun = @(t) EvalFn(t,X,Xt,z,lam);
t0 = randn(2*n,1);
[v,f,g] =  func_lbfgs(Efun,t0,opts);

x = v(1:end/2).*v(end/2+1:end);

end



function fg = EvalFn(x,X,Xt,z,lam)


t =  x(1:end/2);
a =  x(end/2+1:end);

v = a.*t;
% Av = A*v;
Av = Xt*(X*v);
f = 0.5*v'*Av- z'*v ...
    + lam/2*norm(t)^2+lam/2*norm(a)^2;

gt = a.*(Av- z) + lam*t;
ga =  t.*(Av - z) + lam*a;

fg = [f;gt;ga];

end