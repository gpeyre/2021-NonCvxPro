clf
M = 140;
n=256;
K=40;
mvals = 60:2:140;
qvals = [0.67, 0.75,0.9,1];

success_r = zeros(length(mvals),length(qvals));
success_pro = zeros(length(mvals),length(qvals));
for k=1:100
    X = randn(M,n);
    
    beta0 = zeros(n,1);
    beta0(randperm(n,K)) = randn(K,1);
    y = X*beta0;
    
    for j=1:length(mvals)
        m = mvals(j);
            Xm = X(1:m,:);
            ym = y(1:m);
        for i=1:length(qvals)                   
            progressbar(i+length(qvals)*(j-1),length(mvals)*length(qvals));
            q = qvals(i);
            xr  =  func_IRLQ_lq(Xm,ym,q);
            success_r(j,i) = (norm(xr-beta0)/norm(beta0)<0.01)+success_r(j,i);
            xp  = func_LassoPro_lq(Xm,ym,q);
            res = norm(xp-beta0)/norm(beta0);
            
            t=1; %retry 10 times
            while res>0.01 && t<10 && q<0.99 % && (norm(xr-beta0)/norm(beta0)<0.01)
                xp  = func_LassoPro_lq(Xm,ym,q);
                res = norm(xp-beta0)/norm(beta0);
                t=t+1;      
            end
            if t>1
                disp(num2str(res))
            end
                
            if res<0.01
                success_pro(j,i) = success_pro(j,i)+1;
            end
        end
    end
    
    save('Run200521.mat', 'success_r','success_pro');
    success_r
    success_pro
end

%%
clf
subplot(1,2,1)
plot(mvals, success_r/100,'*-', 'linewidth',2)
title('IRLS')
xlabel('m')
ylabel('successful recovery')
legend({'q=0.67','q=0.75','q=0.9','q=1'},'location','southeast','fontsize',12)
subplot(1,2,2)
% hold on
%%
figure
plot(mvals, success_pro,'-', 'linewidth',2)
% title('Noncvx-Pro')

xlabel('m','fontsize',16)
ylabel('Successful recovery','fontsize',16)
legend({'q=0.67','q=0.75','q=0.9','q=1'},'location','southeast','fontsize',14)