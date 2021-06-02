
% Solve min_{t,a} 0.5*|X(t.*a) - y|^2 +lam/2*( |t|^2 + |a|^2 )
% by alternating minimisation  over t and a
function [x,res_alt,f_alt] = func_LassoAlternating(X,Xt,y,lam,maxits,n,x0)

% n = size(X,2);
m = length(y);

if isa(X,'function_handle')
    Ax = @(t,a) Xt(X(t.*a));
    pcgtol = 1e-4;
    pcgits = 100;
    z = Xt(y);
else
    A = X'*X;
    Ax = @(t,a) A*(t.*a);
    z = X'*y;
end





t = x0;%randn(n,1);
a2 = randn(m,1);
t2 = randn(m,1);
a = t;
res_alt = zeros(maxits,1);
f_alt = zeros(maxits,1);
for it = 1:maxits
    tm = t;    
    if m<n
        if isa(X,'function_handle')
            afun = @(x) x + 1/lam*X((t.^2).*Xt(x));
            a2 = pcg(afun,-y,pcgtol,pcgits,[],[],a2);
            a = -t.*(Xt(a2))/lam;
            
            afun = @(x) x + 1/lam*X((a.^2).*Xt(x));
            t2 = pcg(afun,-y,pcgtol,pcgits,[],[],t2);
            t = -a.*(Xt(t2))/lam;

        
        else
            M =  X.*t';
            a2 = -(eye(m) +1/lam* (M*M') )\y;
            a = -t.*(X'*a2)/lam;
            
            M =  X.*a';
            t2 = -(eye(m) +1/lam* (M*M') )\y;
            t = -a.*(X'*t2)/lam;
        end
        
    else
        if isa(X,'function_handle')
            afun = @(x) lam*x + t.*Xt(X(t.*x));
            a = pcg(afun,t.*z,pcgtol,pcgits,[],[],a);
            
            afun = @(x) lam*x + a.*Xt(X(a.*x));
            t = pcg(afun,a.*z,pcgtol,pcgits,[],[],t);
            
        else
            a = (lam*eye(n)+ A.*(t*t'))\(t.*z);
            t = (lam*eye(n)+ A.*(a*a'))\(a.*z);
        end

    end
    
    res = norm(t-tm);
    res_alt(it) = res;
    
    
    x = t.*a;
    f_alt(it) = 0.5*x'*Ax(t,a) - z'*x +lam/2*norm(t)^2+lam/2*norm(a)^2;
    if res<1e-12
        break
    end
end
x = t.*a;
f_alt = f_alt(1:it-1);
res_alt = res_alt(1:it-1);
end