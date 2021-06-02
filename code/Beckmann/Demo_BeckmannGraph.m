%%
% Tests for W1 by Solving Beckmann problem
%       min_z <|z|,c> s.t. div(z)=y=mu-nu
% denoting beta=z.c
%       min_beta |beta|_1 s.t. X*beta=y  where   X=div*diag(1/c)
% and the non-convex var-pro dual formulation split beta=u.v
%       min_uv max_a |u|^2/2+|v|^2/2 + <X*(u.v)-y,a>
% optimal u is u=-Xv'*a  where Xv=X*diag(v)
%       min_v f(v) = max_a |u|^2/2 - |Xv'*a|^2/2 - <y,a>
% f is smooth and its gradient is
%       nabla f(v) = v - v . (X'*a)^2
% where a=a(v) is
%       a(v) = -pinv(Xv*Xv')*y

addpath('./toolbox/');
addpath('./off/');
rep = './results/';
[~,~] = mkdir(rep);

% Helpers
distmat = @(x,y)sqrt( max(bsxfun(@plus,dot(x,x,1)',dot(y,y,1))-2*(x'*y), 0) );
normalize = @(a)a/sum(a(:));
dotp = @(u,v)sum(u(:).*v(:));

%name = 'mushroom';
%name = 'mannequin';
name = 'planar';
%name = 'cow';
name = 'nefertiti';
name = 'yeast';
fprintf('Loading graph ...');
[I,J,c,y,T,Pos] = load_graph(name);
fprintf(' done.\n');

%%
% Compute gradient, div=grad'

n = size(Pos,2); % number of vertices
p = length(I); % number of edge
grad = sparse( [1:p 1:p], [I;J]', [ones(p,1) -ones(p,1)], p,n);
% display the input data
% clf; display_flow(Pos,T,I,J, y);

%%
% Solve Primal and Dual exactly.

fprintf('Solving using interior point ...');
[u,z0] = SolveFlowLinprog(I,J,c,y); % to check, y needed to be consistent with varpro
fprintf(' done.\n');
beta0 = z0 .* c;
% display the unregularized solution
Z =  min(abs(z0),max(abs(z0))*.3); % boost contrast for display
clf; display_flow(Pos,T, I,J, y, [],Z); % grad'*z0
drawnow;
saveas(gcf, [rep 'beckmann-' name '-flow.png'], 'png');

%%
% VarPro

disp('Testing VarPro ...');
X = grad' * spdiags(1./c, 0,p,p);
f = @(v,a) norm(v)^2/2 - norm( v .* (X'*a) )^2/2 - dotp(y,a);
%
GradF = @(v,a)deal( f(v,a), v - (X'*a).^2 .* v );
Gradf = @(v)GradF(v, -( X*spdiags(v.^2, 0,p,p)*X' ) \ y );
v0 = randn(p,1)*.1;
niter = 500;
niter = 300;
niter = 100;
options.niter = niter*10;
%  u = -v .* (X'*a);
%  beta = u.*v = -v.*v.*(X'*a);
A = @(v)-pinv( X*diag(v.^2)*X' ) * y;
Beta = @(v)-v.*v.*(X'*A(v));
options.report = @(v,r)struct('err',norm( Beta(v) - beta0, 1 ),'time',toc);
warning off; tic;
[v, R, info] = perform_bfgs(Gradf, v0, options);
z = Beta(v)./c;
warning on;
results_time = {}; results_err = {}; lgd = {}; disp_style = {}; col = {};
[results_time{end+1},results_err{end+1}] = deal( s2v(R, 'time'), s2v(R, 'err') );
lgd{end+1} = 'NonCvx-Pro'; disp_style{end+1} =  '--'; col{end+1} = [0 0 1];


%%
% Using DR
%       min G+H, G(beta)=|beta|_1 H(beta)=i_{X*beta=y}

ProxG = @(beta,tau)sign(beta) .* max(abs(beta)-tau,0);
ProxH = @(beta,tau)beta + X'*( (X*X')\( y - X*beta ) );
% check projector is ok
% beta = randn(p,1); norm( X*ProxH(beta,0) - y )
options.mu = 1; % in ]0,2[
options.gamma = 1; % >0
options.niter = niter*300;
options.verb = 1;
options.report = @(beta,r)struct('err',norm( beta - beta0, 1 ),'time',toc);
beta_init = zeros(p,1);
gamma_list = .1;
gamma_list = [.05 .1 .5]; % yeast
gamma_list = [.1];
disp('Testing DR ...');
clf; hold on;
for j=1:length(gamma_list)
    s = (j-1)/max(length(gamma_list)-1,1);
    options.gamma = gamma_list(j); tic;
    warning off;
    [beta,R] = perform_dr(beta_init,ProxG,ProxH,options);
    warning on;
    [results_time{end+1},results_err{end+1}] = deal( s2v(R, 'time'), s2v(R, 'err') );
    lgd{end+1} = ['DR,\mu=' num2str(gamma_list(j))]; disp_style{end+1} =  '-';
    col{end+1} = (1-s)*[0 0 1] + s*[0 1 1];
end

%%
% Compare with Primal-Dual
% min_x F(K*x) + G(x)
%   min { |beta|_{1,2} : X*beta = y }
% F=i_{y}, K=X, G=|.|_{1,2}
% FS=<.,y>  ProxFS(u)=argmin_v 1/2*|v-u|^2 + tau*<v,y>=
%      v-u+tau*y=0

options.niter = niter*3000;
options.theta=1; % default is 1
ProxFS = @(u,tau)u-tau*y;
% one needs options.sigma*options.tau*norm(full(X))^2<1 for convergence
sigma_list = [.01 .001 .0001]; % for small meshes
sigma_list = [.5 1 10 50]; % for large meshes
sigma_list = .05;
sigma_list = [.01 .1 1]; % for small meshes
sigma_list = [.05 .1 .5]; % yeast
disp('Testing PD ...');
warning off;
for j=1:length(sigma_list)
    s = (j-1)/max(length(sigma_list)-1,1);
    options.sigma = sigma_list(j);
    options.tau = .9/( options.sigma*estimate_matrix_norm(X) ); % BE CAREFULL
    tic;
    [beta,R] = perform_primal_dual(beta_init, X,  X', ProxFS, ProxG, options);
    [results_time{end+1},results_err{end+1}] = deal( s2v(R, 'time'), s2v(R, 'err') );
    lgd{end+1} = ['PD,\sigma=' num2str(options.sigma)]; disp_style{end+1} =  '-';
    % col{end+1} = (1-s)*[1 .8 0] + s*[.5 0 .5];
    col{end+1} = (1-s)*[1 0 0] + s*[0 1 0];
end

%%
% Final display.

tmax = 300;
tmax = Inf;
sel = 2:4;
sel = 5:7;
sel = 1:1:length(lgd);
sel = [1 2 4];
clf; hold on;
for k=sel %
    t = results_time{k};
    transp = 1;
    if strcmp(disp_style{k},'-')
        transp = .5;
    end
    plot(t(t<tmax), log10(results_err{k}(t<tmax)), disp_style{k}, 'color', [col{k} transp], 'LineWidth', 2);
end
legend({lgd{sel}},'Location','southwest');
box on; axis tight;
set(gca, 'PlotBoxAspectRatio', [1 2/3 1], 'FontSize', 15); %, 'XTick', [], 'YTick', []


saveas(gcf, [rep 'beckmann-' name '.eps'], 'epsc');
