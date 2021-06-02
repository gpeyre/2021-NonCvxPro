function  [x,fvals] = func_FISTA(opt)
lam = opt.lam;
maxits = opt.maxits;
% x = opt.x0;
grad = opt.grad;
Obj = opt.obj;
L = opt.L;
z = opt.z;
x = z;
prox = opt.prox;
tol = opt.tol;

y = x;
theta =1;
gamma = 1/L;
fvals = zeros(maxits,1);
verbose = opt.verbose;
for it = 1:maxits
    xm = x;
    x = prox(y - gamma*grad(y),  gamma*lam);
    theta = (1+sqrt(1+4*theta^2))/2;    
    ym = y;
    aa = (theta-1)/theta ;
    y = x + aa*(x - xm);    
    r = norm(x-xm)/norm(x);
    fvals(it) = Obj(x);
    
    %stop?
    if r< tol 
        break
    end
    
    %restart?
    if (ym(:)-x(:))'*(x(:)-xm(:)) > 0
        y = x;
        theta = 1;
    end
    
    if mod(it,verbose) == 0
        fprintf('Iteration: %d, fval: %.3e\n',it,fvals(it));
    end
    
      
end

fvals = fvals(1:it);

end