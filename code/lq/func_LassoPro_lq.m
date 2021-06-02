function beta = func_LassoPro_lq(X,y,q)
[~,n] = size(X);
dotp = @(u,v)sum(u(:).*v(:));
A = @(v)-( X*diag(v.^2)*X' )\y;
Beta = @(v)-v.*v.*(X'*A(v));

p = 2*q/(2-q);
C = (2-q)*q^(q/(2-q))/2;

f = @(v,a) C*sum(abs(v).^p)- norm( v .* (X'*a) )^2/2 - dotp(y,a);
warning off
GradF = @(v,a) deal( f(v,a), C*p*abs(v).^(p-1).*sign(v) - (X'*a).^2 .* v );
Gradf = @(v) GradF(v, -( X*diag(v.^2)*X' ) \y );
options.verb = 0;
options.niter = 1500;
options.bfgs_memory = 20;
v0 =  randn(n,1)*.10;
[v, R, info] = perform_bfgs(Gradf, v0, options);
beta = Beta(v);
end