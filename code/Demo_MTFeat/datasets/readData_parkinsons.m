fileID = fopen('parkinsons_updrs.data','r');

C = textscan(fileID,'%s');

fclose(fileID);


Z = C{1,1};
A = zeros(length(Z)-1,22);
for i=2:length(Z)
    str = Z{i,1};
    A(i-1,:) = str2num(str);
end

L = max(A(:,1));
indx = ones(L,1);
for i=2:L
    k = find(A(:,1)==i,1);
    indx(i) = k;
end

%column 5 and 6 are the values to predict
y = A(:,5:6);
Data = A(:,[2:4,7:end]);
x = Data';
task_indexes = indx;
save('fullData.mat', 'x','y','task_indexes');


%%
indx(L+1) = size(Data,1)+1;
for s=1:10
    
    tr = [];
    tst = [];    
    id_train(1) = 1;
    id_test(1) = 1;
    for i=1:L
        n = indx(i+1)-indx(i);
        rn = randperm(n);
        idx = (indx(i):indx(i+1)-1)';
        m = ceil(0.75*n);
        tr = [tr; idx(rn(1:m)) ];
        tst = [tst; idx(rn(m+1:end) ) ];
            
        id_train(i+1) = length(tr)+1;
        id_test(i+1) = length(tst)+1;
        
        
    end

id_train = id_train(1:end-1)';
id_test = id_test(1:end-1)';


tr = tr(:);
tr_indexes = id_train;
tst = tst(:);
tst_indexes = id_test;

save(sprintf('split%d.mat',s), 'tr','tst','tr_indexes','tst_indexes');

end
