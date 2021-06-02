function br = func_IRLQ_lq(X,y,q)

eps= 1;
x = pinv(X)*y;
kmax = 16;
for k=1:kmax
    it = 1;
    res = 1;
%     progressbar(k,kmax);
    while it<1000 && res >1e-8
        xm = x;
        w = (abs(x).^2+eps).^(q/2-1);
        x = diag(1./w)*X'*((X*diag(1./w)*X')\y);
        res =  norm(xm-x);
        it = it+1;
    end
    eps = eps/10;
end
br  = x;
end