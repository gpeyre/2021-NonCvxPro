%%
% Display the gradient flow evolution.
rng(1124,'twister')


rep = './results/';
[~,~] = mkdir(rep);


% CS type problem
n = 20;
p = round(n/2);

    
msgn = 1;
a0 = 0;
i=0;
if 1
%find a starting point so that varpro has to cross 0
while msgn>-.1 || a0<0.05
    i=i+1;
    X = randn(p,n);
    beta0 = zeros(n,1);
    beta0(randperm(n,2)) = [.5;-.5];
    y = X*beta0;
    lambda = max(abs(X'*y))/10;
    v0 = randn(n,1);
    u = ( diag(v0)*(X'*X)*diag(v0) + lambda*eye(n) ) \ (v0.*(X'*y));
    beta = v0.*u;
    mvec = sign(beta0).*sign(X'*(y-X*beta));
    msgn = min(mvec);
    if msgn<-0.1
        indx = find(mvec<0,1);
        a0 = abs(beta(indx));
    end
    
end

end
betainit = v0.*u;
%%
niter = 1800;
% ISTA flow

SoftThr = @(beta,tau)sign(beta) .* max(abs(beta)-tau,0);
% iterative soft thresholding
beta = betainit;
tau = .1 * 1/norm(X)^2;
Beta_Ista = [];
for it=1:niter
    Beta_Ista(:,end+1) = beta;
    beta = SoftThr( beta - tau * X'*(X*beta-y), tau*lambda );
end
beta_ista = beta;

%%
% VarPro Flow

Beta_VarPro = [];
v = v0;
V = [];
tau = .01/3;
for it=1:niter
    V(:,end+1) = v;
    u = ( diag(v)*X'*X*diag(v) + lambda*eye(n) ) \ (v.*(X'*y));
    beta = u .* v;
    Beta_VarPro(:,end+1) = beta;
    g = v + 1/lambda * u .* X'*( X*(v.*u) - y );
    v = v - tau*g;
end
U = Beta_VarPro./V;


clf; hold on;
subplot(2,1,1);
plot(U');
subplot(2,1,2);
plot(V');


%%


newcolors=turbo(n);
colororder(newcolors)
bmax = .5;
clf;

% subplot(1,2,1); 
hold on;
plot(Beta_Ista', '-', 'LineWidth',1);
plot([1 niter], [0 0], 'k-', 'LineWidth',2);
plot(Beta_Ista(indx,:),'r-', 'LineWidth',2)

axis([1 niter -bmax bmax ]);
set(gca, 'PlotBoxAspectRatio', [1 2/3 1], 'XTick', [], 'YTick', []);
box on;
exportgraphics(gcf,'results/flow_ISTA.png','Resolution',300)

ylabel('ISTA');





% subplot(1,2,2); 
clf
hold on;
plot(Beta_VarPro', '-', 'LineWidth',1);
plot([1 niter], [0 0], 'k-', 'LineWidth',3);
plot(Beta_VarPro(indx,:),'r-', 'LineWidth',2)
axis([1 niter -bmax bmax ]);
set(gca, 'PlotBoxAspectRatio', [1 2/3 1], 'XTick', [], 'YTick', []);
box on;
exportgraphics(gcf,'results/flow_NcvxPro.png','Resolution',300)
ylabel('Pro-NonCvx');