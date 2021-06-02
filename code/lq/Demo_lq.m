addpath('./toolbox/');
rng(1124,'twister')

q =0.8; %lq regularisation

%create matrix
type = 'CS';
switch type
    case 'CS'
        M=90;
        n = 256;
        K =40;
        X = randn(M,n);
        beta0 = zeros(n,1);
        beta0(randperm(n,K)) = randn(K,1);
        y = X*beta0;
    case 'deconv'        % deconvolution matrix
        M=80;
        n = 256;
        K =17;
        del = .1;
        % kernel
        t = [0:n/2,-n/2+1:-1]'/n;
        % non smooth box kernel
        h = double(abs(t)<del);
        % gaussian kernel
        h = exp(-t.^2/(2*del^2));
        h = h/sum(h);
        X =  convmtx(h,n);
        X = X(1:4:end,:);
        beta0 = zeros(n,1);
        beta0(randperm(n,K)) = randn(K,1);
        y = X*beta0;
end


dotp = @(u,v)sum(u(:).*v(:));
A = @(v)-( X*diag(v.^2)*X' )\y;
Beta = @(v)-v.*v.*(X'*A(v));
%% compute l1 solution
options.niter = 1000;
options.bfgs_memory = 20;
options.report = @(v,r)norm( Beta(v) - beta0, 1 );
f = @(v,a) 0.5*norm(v)^2- norm( v .* (X'*a) )^2/2 - dotp(y,a);
GradF = @(v,a)deal( f(v,a), v - (X'*a).^2 .* v );
Gradf = @(v)GradF(v, -pinv( X*diag(v.^2)*X' ) * y );
v0 =  randn(n,1)*0.1;
[v, R, info] = perform_bfgs(Gradf, v0, options);
v1 = v;
beta1 = Beta(v1);
%% lq Pro

p = 2*q/(2-q);
C = (2-q)*q^(q/(2-q))/2;

f = @(v,a) C*sum(abs(v).^p)- norm( v .* (X'*a) )^2/2 - dotp(y,a);
warning off
GradF = @(v,a) deal( f(v,a), C*p*abs(v).^(p-1).*sign(v) - (X'*a).^2 .* v );
Gradf = @(v) GradF(v, -( X*diag(v.^2)*X' ) \y );

lqnorm = @(b) sum(abs(b).^q);
options.niter = 2000;
options.bfgs_memory = 20;
options.report = @(v,r)norm( Beta(v) - beta0, 1 );
bpro = X\y;

%note that since the function is nonconvex
%one can get stuck at `bad' local minimums
%so I run here 10 times with different starting points
%
%investigating graduated nonconvex or other optimisation approaches
%might be interesting future work.
for i=1:10
    v0 =  randn(n,1)*.10;
    [v, R, info] = perform_bfgs(Gradf, v0, options);
    b = Beta(v);
    lqnorm(b)
    norm(beta0-b)
    if lqnorm(bpro)>lqnorm(b)
        bpro=b;
    end
end
errpro = norm(bpro-beta0)/norm(beta0);

%%
if 0
niter = 100000;
v0 =  randn(n,1)*.10;
v = v0;
tau = .5;
for it=1:niter 
    progressbar(it,niter);
    a = ( X*diag(v.^2)*X' )\y;
    grd = C*p*abs(v).^(p-1).*sign(v) - (X'*a).^2 .* v ;
    v = v-tau*grd;
end
bpro2 = Beta(v);
lqnorm(bpro2)

warning on
end
%% IRLS
warning off
eps= 1;
x = pinv(X)*y;
% x = randn(n,1);
kmax = 16;
for k=1:kmax
    it = 1;
    res = 1;
    progressbar(k,kmax);
    while it<1000 && res >1e-10
        xm = x;
        w = (abs(x).^2+eps).^(q/2-1);
        x = diag(1./w)*X'*((X*diag(1./w)*X')\y);
        res =  norm(xm-x);
        it = it+1;
    end
    eps = eps/10;
end
br  = x;
errrw = norm(br-beta0)/norm(beta0);
warning on
%%
fprintf('\nIRLS lq norm: %s, Error:%s\n', num2str(lqnorm(br)),num2str(errrw))
fprintf('Pro lq norm: %s, Error:%s\n', num2str(lqnorm(bpro)),num2str(errpro))


clf
stem(br, 'bd', 'markersize',10)
hold on
stem(bpro,'m*', 'markersize',10)
stem(beta0, 'gx', 'markersize',10)
stem(beta1, 'k+')
legend('IRLS','lqPro','true','l1')
