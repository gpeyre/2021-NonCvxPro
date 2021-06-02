% Inputs:
% indx  = starting index for each task
% X     = training data matrix, size d by m
% y     = training output, size m by 1
% lam   = regularisation parameter
%
% Outputs:
% W     = regression coefficients
% fval  = objective values

function [W,f,o] = func_MultifeatPro(X,y,indx,lam,opts)
d = size(X,1);
T = length(indx);
[XtX, Xty] = split(X,y,indx);

Efun = @(U) EvalFn(U,XtX,lam,Xty,T,d);


t0 = randn(d^2,1);
[t,f,~] =  func_lbfgs(Efun,t0,opts);

U = reshape(t,d,d);
V = zeros(d,T);
for t=1:T
    Q = XtX{t};
    z = Xty{t};
    M = lam*eye(d) + U'*Q*U;
    vt = M\(U'*z);
    V(:,t) = vt;
end
W = U*V;

o = checkOptimality(W,XtX,Xty,lam);

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

%returns [f;g] where f is the function value, g is the gradient
function fg = EvalFn(U,XtX,lam,Xty,T,d)

fval = lam/2*norm(U,'fro')^2;
U = reshape(U,d,d);
grad = lam*U;

for t=1:T
    Q = XtX{t};
    z = Xty{t};
    M = lam*eye(d) + U'*Q*U;
    vt = M\(U'*z);
    Uv = U*vt;
    XXUv = (Q*Uv);
    grad = grad + (XXUv - z)*vt';
    fval = fval + lam/2*norm(vt)^2 - z'* Uv + Uv'*XXUv/2;
end

fg = [fval;grad(:)];
end


%should return a value at most 1
function o = checkOptimality(W,XtX,Xty,lam)
[d,T] = size(W);
G = zeros(size(W));
for t=1:T
    wt = W(:,t);
    Q = XtX{t};
    z = Xty{t};    
    ut  = (Q*wt - z);
    G(:,t) = ut;
end

[U,~,V] = svd(W);

o = norm(G/lam + U*spdiags(ones(d,1),0,d,T)*V' );

end