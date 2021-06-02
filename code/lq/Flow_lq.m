%%
% Display the gradient flow evolution.
saveresults = 0;
rep = './results/';
[~,~] = mkdir(rep);
rng(31,'twister')

% CS type problem
n = 20;
m = round(n/2);
X = randn(m,n);
beta0 = zeros(n,1);
% s = 5;
s = 2;
beta0(randperm(n,s)) = (randn(s,1));
y = X*beta0;




%%
clc
clf
v0 = randn(n,1);

regenerate = 0; %regenerate starting point for each q
bmax = 1.1;
niter = 1000;
newcolors=turbo(n);
colororder(newcolors)

qvals = [0.7,0.8,.9,1];
L = length(qvals);
tau = .01/2;
lam =0;%norm(X'*y,'inf')/10;

figure(1)
clf
figure(2)
clf
for i=1:L
    q = qvals(i);
    p = 2*q/(2-q);
    C = (2-q)*q^(q/(2-q))/2;
    
    warning off
    
    %%
    tries = 0; res = 1;
    
    while res>0.01 && tries<5
        if regenerate
            v0 = randn(n,1);
        end
        Beta_VarPro_lq = [];
        v = v0;
        % randn('state', 13);
        V = [];
        
        for it=1:niter
            V(:,end+1) = v;
            
            a = -(lam*eye(m)+ X*diag(v.^2)*X' ) \y ;
            beta = -v.*v.*(X'*a);
            Beta_VarPro_lq(:,end+1) = beta;
            
            %gradient, for q=1, same as g = v - (X'*a).^2 .* v;
            g = C*p*abs(v).^(p-1).*sign(v) - (X'*a).^2 .* v;
            v = v - tau*g;
            
        end
        res = norm(beta - beta0);
        tries = tries+1;
        if ~regenerate
            tries = 100;
        end
    end
    res
    %%
    figure(1)
    if ~saveresults
    subplot(2,ceil(L/2),i); 
%     clf
    end
    
    plot(Beta_VarPro_lq', '-', 'LineWidth',2);
    hold on;
    plot([1 niter], [0 0], 'k-', 'LineWidth',2);
    
    set(gca, 'PlotBoxAspectRatio', [1 2/3 1], 'XTick', [], 'YTick', []);
    box on;
    if saveresults
    exportgraphics(gcf,sprintf('%sbetaflow_lq%.e.png',rep,q),'Resolution',300)
    end
    ylabel(sprintf('q=%.2f',q));
    title('\beta')
    hold off
    
    figure(2)
    if ~saveresults
    subplot(2,ceil(L/2),i);
%     clf
    end
    
    plot(V', '-', 'LineWidth',2);
    hold on;
    plot([1 niter], [0 0], 'k-', 'LineWidth',2);
    
    set(gca, 'PlotBoxAspectRatio', [1 2/3 1], 'XTick', [], 'YTick', []);
    box on;
    if saveresults
        exportgraphics(gcf,sprintf('%svflow_lq%.e.png',rep,q),'Resolution',300)
    end
    ylabel(sprintf('q=%.2f',q));
    title('V')
    hold off
end

