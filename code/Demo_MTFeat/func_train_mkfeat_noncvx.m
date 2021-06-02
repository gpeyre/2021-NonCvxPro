%k fold cross validation
function [W1,optlambda,optval,errorcurve] = func_train_mkfeat_noncvx(X,y,indx0,k,lamvals,trainer)
m = length(y);

group = ones(m,1);
L =length(indx0);
for i=2:L-1
    group(indx0(i):indx0(i+1)-1) = i;
end
group(indx0(L):end) = L;

hpartition = cvpartition(group,'KFold',k); % Nonstratified partition


Llam = length(lamvals);
error = zeros(Llam,1);

for i=1:k
    [Xtst,ytst,indtst,Xtr,ytr,indtr] = extract(i,hpartition,X,y,group);
    
    erri = zeros(Llam,1); %RMSE across different reg. params
    for j=1:Llam
        lambda = lamvals(j);
        W1 = trainer(Xtr,ytr,indtr,lambda);
        [~, RMSE_j] = testErrors(W1, Xtst,ytst,y,indtst,indx0);
        erri(j) = RMSE_j;
    end
    
    error = error+erri;
end
errorcurve = error/k;
[optval,i] = min(error);

optlambda = lamvals(i);
W1 = trainer(X,y,indx0,optlambda);

end


%given data matrix X, vector y and indx indication which group each entry
%belongs to, extract elements corresponding to idx_tst.
%
function [Xtst,ytst,indtst,Xtr,ytr,indtr] = extract(i,hpartition,X,y,group)

idxTrain = training(hpartition,i);
idxNew = test(hpartition,i);

ytrn = y(idxTrain);
Xtrn = X(:,idxTrain);
indtrn = group(idxTrain);
[indtrn,j] = sort(indtrn);
ytr = ytrn(j);
Xtr = Xtrn(:,j);
L = indtrn(end);
indtr = ones(L,1);
for j=2:L
    indtr(j) = find(indtrn==j,1);
end


ytst = y(idxNew);
Xtst = X(:,idxNew);
indts = group(idxNew);
[indts,j] = sort(indts);
ytst = ytst(j);
Xtst = Xtst(:,j);
L = indts(end);
indtst = ones(L,1);
for j=2:L
    indtst(j) = find(indts==j,1);
end

end
