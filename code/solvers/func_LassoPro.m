% Implements varpro + nonconvex splitting for
%    (*)       min_a  1/(2*lam) |X a - y|^2 + |a|_1
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
%   x in argmin (*), 
%   f and g are the function and gradient values
%
function [x,f,g]  = func_LassoPro(X,Xt,y,lam,n,opts)
if nargin<6
    opts = struct();
    
end
if ~isfield( opts, 'pcgtol' )
    opts.pcgtol = 1e-8;
end
if ~isfield( opts, 'pcgits' )
    opts.pcgits = 1000;
end

m = length(y);
if isa(X,'function_handle')
    setGlobal(randn(m,1));
end


Efun = @(t) EvalFn(t,X,Xt,lam,m,y,opts);

t0 = rand(n,1)*0.1;
[t,f,g] =  func_lbfgs(Efun,t0,opts);

if isa(X,'function_handle')
    afun = @(x) lam*x + X((t.^2).*Xt(x));
    a = pcg(afun,-y,opts.pcgtol,opts.pcgits);
    Xta = Xt(a);
%     setGlobal(randn(m,1));
else
    M = lam*speye(m)+X*spdiags(t.^2,0,length(t),length(t))*X';
    a = -M\y;
    Xta = Xt*a;
end

x = -(t.^2).*Xta;

end

%returns [f;g] where f is the function value, g is the gradient
function fg = EvalFn(t,X,Xt,lam,m,y,opts)

if isa(X,'function_handle')
    a0 = getGlobal;
    afun = @(x) lam*x + X((t.^2).*Xt(x));
    alph = pcg(afun,-y,opts.pcgtol,opts.pcgits,[],[],a0);
    Xta = Xt(alph);
    setGlobal(alph);
    
else
    M = lam*speye(m)+X*spdiags(t.^2,0,length(t),length(t))*X';
    alph = -M\y;
    Xta = Xt*alph;
end

v1 = t - t.*(Xta.^2);
f = - lam*norm(alph)^2/2 + norm(t)^2/2 ...
        - sum(alph.*y)- sum(t.^2.*Xta.^2)/2;
fg = [f ;v1];





end




function setGlobal(a_val)
global alpha_last
alpha_last = a_val;
end

function r = getGlobal
global alpha_last
r = alpha_last;
end

