%%
% Solve W1 on meshes using P1 finite elements.

addpath('./toolbox/');
addpath('./off/');
rep = './results/';
[~,~] = mkdir(rep);

dotp = @(u,v)sum(u(:).*v(:));

name = 'half-brain';
ext = 'stl';

name = 'brain';
name = 'cow';
name = 'nefertiti';
name = 'cortex-20k';
ext = 'off';


global meshname
meshname = name;
switch ext
    case 'off'
        [Pos,F] = read_off([name '.off']);
    case 'stl'
        [F,Pos] = stlread([name '.stl']); Pos = Pos'; F = F';
end

% center/scale
Pos = Pos-mean(Pos,2);
Pos = Pos / mean( sqrt(sum(Pos.^2)) );
%
opt.name = name;
n = size(Pos,2);
m = size(F,2);
d = size(Pos,1); % dimension, should be 3


switch name
    case 'cortex-20k'
        q = 100; % # point in support
        S = [6543,6894];
        S = [9600,2631]; % medium
        S = [8175 3999]; % close
    case 'nefertiti'
        q = 10;
        S = [271,146];
    otherwise
        error('Please specify pair of vertices.');

end

y = zeros(n,1);
sigma = [.06 .06];
for i=1:2
    Dist = sum( (Pos-Pos(:,S(i))).^2 );
    Dist = Dist/max(Dist);
    mu = exp(-Dist/(2*sigma(i)^2)); mu = mu/sum(mu(:));
    % y(I(1:q))  = (-1)^(i-1) / q;
    [Dist1,I] = sort(Dist, 'ascend');
    y = y + (-1)^(i-1) * mu(:);
end

mm = linspace(0,1,128)';
CM = [(1-mm)*[1 0 0] + mm*[1 1 1]; (1-mm)*[1 1 1] + mm*[0 0 1]];
opt.face_vertex_color = y/max(abs(y));
clf; hold on;
plot_mesh(Pos,F, opt);
caxis([-1 1]); colormap(CM);

% load gradient on the mesh
[grad,Normals] = load_mesh_grad(Pos,F);

% Solve a Basis Pursuit with group lasso:
%   min { |beta|_{1,2} : X*beta = y }

p = size(grad,1)/d;
c = ones(d*p,1);
X = grad' * spdiags(1./c, 0,d*p,d*p);

v = randn(p,1);
a = randn(n,1);
gmult = @(v,q)[v;v;v] .* q; % group multiply
gsqnorm = @(w)w(1:end/3).^2 + w(end/3+1:2*end/3).^2 + w(2*end/3+1:end).^2;
f = @(v,a) norm(v)^2/2 - norm( gmult(v,X'*a) )^2/2 - dotp(y,a);

A = @(v)-( X*spdiags([v.^2;v.^2;v.^2], 0,d*p,d*p)*X' ) \ y;
GradF = @(v,a)deal( f(v,a), v - v .* gsqnorm(X'*a)  );
Gradf = @(v)GradF(v, A(v) );
v0 = randn(p,1)*.1;
%  u = -v .* (X'*a);
%  beta = u.*v = -v.*v.*(X'*a);
Beta = @(v)-gmult(v.^2, X'*A(v) );


niter = 200;
niter = 50;
niter = 1000;

% compute high precision solution
options.niter = niter*5;
options.report = @(v,r)r;
warning off; tic;
[v_star, R, info] = perform_bfgs(Gradf, v0, options);
beta_star = Beta(v_star);
warning on;

% monitor evolution
options.niter = niter;
options.report = @(v,r)struct('err',norm(beta_star-Beta(v), 1),'time',toc);
warning off; tic;
[v, R, info] = perform_bfgs(Gradf, v0, options);
z = Beta(v)./c;
warning on;
results_time = {}; results_err = {}; lgd = {}; disp_style = {}; col = {};
[results_time{end+1},results_err{end+1}] = deal( s2v(R, 'time'), s2v(R, 'err') );
lgd{end+1} = 'NonCvx-Pro'; disp_style{end+1} =  '-.'; col{end+1} = [0 0 1];
clf;
plot(results_time{end}, log10(results_err{end}));

%%
% Display.

% should be equal to mu-nu
D = grad'*z;
% norm(D-y)/norm(y)
% display the vector field
Z = reshape(z, [p 3]); % ./ sqrt(Area(:));
Z = Z ./ max(sqrt(sum(Z.^2,2)));

I = find(sqrt(sum(Z.^2,2))>.02);
opt.face_vertex_color = y/max(abs(y));
clf; hold on;
plot_mesh(Pos,F, opt);
caxis([-1 1]); colormap(CM);
offs = .03;
for i=I(:)'
    a = mean(Pos(:,F(:,i)),2) + offs*Normals(:,i); % face centroid + little normal offset
    h = Z(i,:);
    h = h/norm(h) * min(norm(h),.15)*1.5;
    b = a(:)+h(:);
    % plot3(a(1),a(2),a(3), 'k.', 'MarkerSize', 10);
    plot3(b(1),b(2),b(3), '.', 'MarkerSize', 10, 'color', [0 .5 0]);
    plot3([a(1), b(1)],[a(2), b(2)],[a(3),b(3)], '-', 'LineWidth', 2, 'color', [0 .6 0]);
end
saveas(gcf, [rep 'beckmann-' name '-flow.png']);


%%
% Compare with Douglas-Rachford

beta = randn(m*3,1);

resh = @(beta)reshape(beta, m,3);
flat = @(b)b(:);
Ampl = @(beta)sqrt( sum( reshape(beta, m,3).^2, 2 ) );
GroupThresh = @(b,tau)b ./ (1e-20 + Ampl(b)) .* max(Ampl(b)-tau,0);
ProxG = @(beta,tau)flat( GroupThresh(resh(beta),tau) );
SoftThresh = @(beta,tau)sign(beta) .* max(abs(beta)-tau,0);
ProxH = @(beta,tau)beta + X'*( (X*X')\( y - X*beta ) );
% check projector is ok
% beta = randn(m*3,1); norm( X*ProxH(beta,0) - y )
options.mu = 1; % in ]0,1[
options.gamma = 1; % >0
options.niter = niter*2;
options.niter = niter*6; % large meshes
options.verb = 1;
%options.report = @(beta,r)norm( beta - beta_star, 1 );
options.report = @(beta,r)struct('err',norm(beta_star-beta, 1),'time',toc);
beta_init = zeros(3*m,1);

gamma_list = [.01 .1 .5 1]; % for small meshes
gamma_list = [.001 .01 .05 ]; % for large mesh
style_list = {'--', '-', '-.'};
warning off;
disp('Testing DR ...');
for j=1:length(gamma_list)
    s = (j-1)/(length(gamma_list)-1);
    options.gamma = gamma_list(j); tic;
    [beta,R] = perform_dr(beta_init,ProxG,ProxH,options);
    % [beta,R] = perform_dr(beta_init,ProxH,ProxG,options);
    [results_time{end+1},results_err{end+1}] = deal( s2v(R, 'time'), s2v(R, 'err') );
    lgd{end+1} = ['DR,\mu=' num2str(gamma_list(j))]; disp_style{end+1} =  style_list{j};
    % col{end+1} = [s 0 1-s];
    col{end+1} = [1 0 0]; % (1-s)*[1 0 0] + s*[0 1 0];
end
warning on;



%%
% Compare with Primal-Dual
% min_x F(K*x) + G(x)
%   min { |beta|_{1,2} : X*beta = y }
% F=i_{y}, K=X, G=|.|_{1,2}
% FS=<.,y>  ProxFS(u)=argmin_v 1/2*|v-u|^2 + tau*<v,y>=
%      v-u+tau*y=0

options.niter = niter*200;
options.theta=1; % default is 1
ProxFS = @(u,tau)u-tau*y;
% one needs options.sigma*options.tau*norm(full(X))^2<1 for convergence
sigma_list = [.01 .001 .0001]; % for small meshes
sigma_list = [.01 .1 1]; % for small meshes

sigma_list = [.1 1 10]; % for large meshes
warning off;
disp('Testing PD ...');
for j=1:length(sigma_list)
    s = (j-1)/(length(sigma_list)-1);
    options.sigma = sigma_list(j);
    options.tau = .9/( options.sigma*estimate_matrix_norm(X) ); % BE CAREFULL
    tic;
    [beta,R] = perform_primal_dual(beta_init, X,  X', ProxFS, ProxG, options);
    [results_time{end+1},results_err{end+1}] = deal( s2v(R, 'time'), s2v(R, 'err') );
    lgd{end+1} = ['PD,\sigma=' num2str(options.sigma)]; disp_style{end+1} = style_list{j};
    % col{end+1} = (1-s)*[1 .8 0] + s*[.5 0 .5];
    col{end+1} = [0 0.7 0]; % (1-s)*[1 0 0] + s*[0 1 0];
end


%%
% Final display.

tmax = Inf;
tmax = 300;
clf; hold on;
for k=1:length(lgd)
    t = results_time{k};
    plot(t(t<tmax), log10(results_err{k}(t<tmax)), disp_style{k}, 'color', col{k}, 'LineWidth', 2);
end
legend(lgd,'Location','southwest');
box on; axis tight;
set(gca, 'PlotBoxAspectRatio', [1 2/3 1], 'FontSize', 15); %, 'XTick', [], 'YTick', []
saveas(gcf, [rep 'beckmann-' name '.eps'], 'epsc');
