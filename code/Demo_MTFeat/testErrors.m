% W = regression coefficients, size d times T
% X = data matrix size d times m
function [Errors, RMSE] = testErrors(W, X,y,y_tot,indx,indx_tot)
[~,T] = size(W);
m = size(X,2);
Errors = zeros(T,1);

Vari = zeros(T,1);
indx(T+1) = m+1;
M = length(y_tot);
indx_tot(T+1)= M+1;
residual = zeros(m,1);
for t=1:T
    i1 = indx(t);
    i2 = indx(t+1)-1;
    haty = X(:,i1:i2)'*W(:,t);
    r =  haty - y(i1:i2);
    Errors(t) = r'*r/(i2+1-i1);
    Vari(t) = var( y_tot(indx_tot(t):indx_tot(t+1)-1) , 1);
    residual(i1:i2) = r;
end

RMSE = norm(residual)/sqrt(m);
