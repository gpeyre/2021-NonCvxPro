function s = checkOptimality(X,y,indx,W,lam)

T = length(indx);
m = size(X,2);
indx(T+1) = m+1;
C = W;
for t=1:T
    i1 = indx(t);
    i2 = indx(t+1)-1;
    Xt = X(:, i1:i2);
    yt = y(i1:i2);
    C(:,t) = Xt*(Xt'*W(:,t))- Xt*yt;
end
C = C/lam;
s = svd(C);

end