Data = load('sarcos_inv.mat').sarcos_inv;
Data2 = load('sarcos_inv_test.mat').sarcos_inv_test;
Data = [Data;Data2];
x = Data(:,1:21);
m  = size(x,1);
y = Data(:,22:28);
y = y(:);
task_indexes = (1:m:7*m)';
x = repmat(x,7,1);
x = x';

save('fullData.mat', 'x','y','task_indexes');


%%
task_indexes(end+1) = size(x,2)+1;
for s=1:10
    
    tr = [];
    tst = [];    
    id_train(1) = 1;
    id_test(1) = 1;
    for i=1:7
        n = task_indexes(i+1)-task_indexes(i)
        rn = randperm(n);
        idx = (task_indexes(i):task_indexes(i+1)-1)';
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
