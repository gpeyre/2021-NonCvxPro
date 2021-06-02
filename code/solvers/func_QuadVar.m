% Solves min_a 1/2 |X a - y|^2 + lam*|a|_1
% where a in R^n,
% implements varpro + eta trick
function [x_lbfgs,f_lbfgs,g_lbfgs]  = func_QuadVar(X,Xt,y,lam,n,opts)
if nargin<6
    opts = struct();
end
opts.pos = 1;

if ~isfield( opts, 'pcgtol' )
    opts.pcgtol = 1e-8;
end
if ~isfield( opts, 'pcgits' )
    opts.pcgits = 1000;
end
m = length(y);
setGlobal(randn(m,1));

Efun = @(t) EvalFn(t,X,Xt,lam,y,opts);

t0 = ones(n,1);

[t,f_lbfgs,g_lbfgs] =  func_lbfgs(Efun,t0,opts);


if isa(X,'function_handle')
    ainit = getGlobal;
    afun = @(x) x + 1/lam*X(t.*Xt(x));
    tol = opts.pcgtol;  
    maxit = opts.pcgits;
    alpha = pcg(afun,y,tol,maxit,[],[],ainit);
    x_lbfgs = t.*(Xt(alpha))/lam;
    setGlobal(alpha);

else
    M =  X.*t';
%     m = length(y);
    [m,n] = size(X);
    if m<n
    alpha = (eye(m) + 1/lam*X*M' )\y;
    else
        alpha =y - X*((lam*diag(1./t)+X'*X)\(X'*y));
    end
    x_lbfgs = t.*(Xt*alpha)/lam;

end


end

%returns [f;g] where f is the function value, g is the gradient
function fg = EvalFn(t,X,Xt,lam,y,opts)

if isa(X,'function_handle')
    ainit = getGlobal;
    afun = @(x) x + 1/lam*X(t.*Xt(x));
    tol = opts.pcgtol;  
    maxit = opts.pcgits;
    alpha = pcg(afun,y,tol,maxit,[],[],ainit);
    u = Xt(alpha);
    setGlobal(alpha)
else
    M =  X.*t';
%     m = length(y);
%     alpha = (eye(m) + 1/lam*X*M' )\y;
    [m,n] = size(X);
    if m<n
        alpha = (eye(m) + 1/lam*X*M' )\y;
    else
        alpha =y - X*((lam*diag(1./t)+X'*X)\(X'*y));
%         z = X'*y;
%         Iz = (lam*eye(n)+(X'*X).*sqrt(t*t'))\(sqrt(t).*z);
%         alpha =y - X*(sqrt(t).*Iz);
    end
    
    u = Xt*alpha;
    
    
end

g = -1/2*abs(u).^2/lam + lam/2;
f = -norm(alpha)^2/2-1/2* sum(t.*(u).^2)/lam +sum(y.*alpha) + sum(t)*lam/2;
fg = [f;g ];
       


end




function setGlobal(val)
global alpha_last
alpha_last = val;
end

function r = getGlobal
global alpha_last
r = alpha_last;
end



