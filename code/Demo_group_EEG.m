addpath('solvers')

%% L-BFGS-B
%{
Download from: https://github.com/stephenbeckr/L-BFGS-B-C

Unpack it somewhere and run lbfgsb_C/Matlab/compile_mex.m
%}
addpath '../packages/L-BFGS-B-C-master/Matlab'

%% ZeroSR1 code of Becker
%{
Download from: https://github.com/stephenbeckr/zeroSR1
%}
addpath 'packages/Becker'

clearvars
randn('state',234213); rand('state',2342343);

%% Generate some 2D coefficients with sine waves with random frequency and phase
factor_ = 50;
dataset_ = 'MEG';
dataset_ = 'synthetic-randn';
switch dataset_
    case 'synthetic-randn'
        n_samples = 30;
        n_features = 1000;
        n_tasks = 100;
        n_relevant_features = 5;
        support = randperm(n_features, n_relevant_features);
        coef = zeros(n_tasks, n_features);
        times = linspace(0, 2 * pi, n_tasks);
        for k =1:length( support)
            coef(:, support(k)) = sin((1 + randn(1,1)) * times + 3 * randn(1,1));
        end
        X = randn(n_samples, n_features);
        Y = X * coef' + randn(n_samples, n_tasks);
        
    case 'MEG'  
%         GAP Safe screening rules for sparse multi-task and multi-class models (NIPS 2015)
%         by Eugene Ndiaye, Olivier Fercoq, Alexandre Gramfort, Joseph Salmon
%         Download from https://drive.google.com/drive/folders/139nKKy0AkpkZntB80n-LmcuzGgC8pQHi
        L = load('GroupLassoDatasets/meg_Xy_full.mat');
        X = L.X;
        Y = L.Y;
end


%% set up
%

NAMES = {};
OBJECTIVES = {};
TIMES = {};
lammax = max(sqrt(sum(abs(X'*Y).^2,2)));
lam = lammax/factor_;



%% setup up for prox-quasi-Newton and FISTA solvers

c = X'*Y;
normQ   = norm(X*X');
n1 = size(X,2);
n2 = size(Y,2);
N = n1*n2;
Rs = @(w) reshape(w,n1,n2);
vec = @(w) w(:);
fcn         = @(w) norm(X*Rs(w)-Y,'fro')^2/2 + lam*sum( sqrt( sum( (Rs(w).^2), 2)  ));
fcnSimple   = @(w) norm(X*Rs(w)-Y,'fro')^2/2;
gradSimple  = @(w) vec( X'*(X*Rs(w)) - c); % doesn't include non-smooth portion

prox    = @(x0,d,l) prox_l1group_rank1_becker( x0, d, l, lam,n1,n2);


maxfun = @(z) (sqrt(sum(abs(z).^2,2)));
cert = @(w) maxfun((X'*(X*Rs(w)) - c)/lam);


%% SR1

% fcn and grad are defined above now...
if ~exist('zeroSR1','file')
    disp('Cannot find zeroSR1 on your path, so skipping this test');
else
opts = struct('N',N,'verbose',100,'nmax',1000,'tol',1e-8);
opts.BB     = true;
% opts.theta  = 1; opts.SR1 = true;
opts.SR1_diagWeight=0.8;

opts.L      = normQ;

disp('running 0-mem SR1')

tic
[xk1,nit, errStruct,optsOut] = zeroSR1_noLinesearch(fcn,gradSimple,prox,opts);

tm = toc;
NAMES{end+1} = '0-mem SR1';
OBJECTIVES{end+1} = errStruct(:,1);
TIMES{end+1} = tm;
end
%% choose FISTA...
if ~exist('zeroSR1','file')
    disp('Cannot find zeroSR1 on your path, so skipping this test');
else
disp('running FISTA')

opts.BB     = true;
opts.theta  = []; opts.restart=1000; % use [] for FISTA
% opts.theta  = 1;
opts.SR1 = false;

opts.backtrack = true;

tic
[xk2,nit, errStruct,optsOut] = zeroSR1(fcn,gradSimple,prox,opts);
tm = toc;
NAMES{end+1} = 'FISTA w/ BB'; % with linesearch
OBJECTIVES{end+1} = errStruct(:,1);
TIMES{end+1} = tm;

end
%% choose BB...
if ~exist('zeroSR1','file')
    disp('Cannot find zeroSR1 on your path, so skipping this test');
else
disp('running SPG/SpaRSA')

opts.BB     = true;
opts.theta  = 1;
opts.SR1 = false;

opts.backtrack = true;

tic
[xk3,nit, errStruct,optsOut] = zeroSR1(fcn,gradSimple,prox,opts);
tm = toc;
NAMES{end+1} = 'SPG/SpaRSA'; % with linesearch
OBJECTIVES{end+1} = errStruct(:,1);
TIMES{end+1} = tm;
end

%%
disp('running NcvxPro')

tic
opts    = struct('printEvery', 1, 'pgtol', 1e-3,  'm',10);%, 'maxIts', 5000,'factr', 1e2, 'pgtol', 1e-16, 'maxTotalIts',1e6);
[W,f] = func_MultiTaskLassoPro(X,Y,lam,opts);
tm = toc;

NAMES{end+1} = 'NonCvxPro'; % with linesearch
OBJECTIVES{end+1} = f;
TIMES{end+1} = tm;

%%
display_objectives(OBJECTIVES, NAMES, TIMES, 'NonCvxPro')

f = gcf;
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 9., 7.], 'PaperUnits', 'Inches', 'PaperSize', [9., 7.])
exportgraphics(f,sprintf('results/MEG/%s_%d.png',dataset_, factor_),'Resolution',300)
