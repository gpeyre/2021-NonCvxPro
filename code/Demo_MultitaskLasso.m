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
addpath '../packages/Becker'

clearvars
randn('state',234213); rand('state',2342343);

%% first run Demo_MultitaskLasso.py so we can compare against CELER
% otherwise, try  Demo_group_EEG.m instead
factor_ =50;
datapath = 'GroupLassoDatasets/';
celerresultspath = 'CelerGroupLassoResults/';

dataset_ = 'synthetic-wave';
dataset_ = 'synthetic';
% dataset_ = 'MNE';
% dataset_ = 'meg_full';

L = load([datapath,dataset_,'.mat']);

X = L.X;
Y = L.Y;


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
opts = struct('N',N,'verbose',100,'nmax',400,'tol',1e-8);
opts.BB     = true;
% opts.theta  = 1; opts.SR1 = true;
opts.SR1_diagWeight=0.8;

opts.L      = normQ;

disp('running 0-mem SR1')

tic
[xk1,nit, errStruct,optsOut] = zeroSR1_noLinesearch(fcn,gradSimple,prox,opts);

tm = toc;
NAMES{1} = '0-mem SR1';
OBJECTIVES{1} = errStruct(:,1)/lam;
TIMES{1} = tm;
end
%% choose FISTA...
if ~exist('zeroSR1','file')
    disp('Cannot find zeroSR1 on your path, so skipping this test');
else
disp('running FISTA')
opt.nmax = 200;
opts.BB     = true;
opts.theta  = []; opts.restart=1000; % use [] for FISTA
% opts.theta  = 1;
opts.SR1 = false;

opts.backtrack = true;

tic
[xk2,nit, errStruct,optsOut] = zeroSR1(fcn,gradSimple,prox,opts);
tm = toc;
NAMES{end+1} = 'FISTA w/ BB'; % with linesearch
OBJECTIVES{end+1} = errStruct(:,1)/lam;
TIMES{end+1} = tm;

% objvals = errStruct(:,1);
% stem(sum(w2,2))
% semilogy(linspace(0,tm,length(objvals)),objvals)
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
opts.nmax = 150;
tic
[xk3,nit, errStruct,optsOut] = zeroSR1(fcn,gradSimple,prox,opts);
tm = toc;
NAMES{end+1} = 'SPG/SpaRSA'; % with linesearch
OBJECTIVES{end+1} = errStruct(:,1)/lam;
TIMES{end+1} = tm;
end
%% Load celer results
L_celer = load([celerresultspath,dataset_,sprintf('%d.mat',factor_)]);

NAMES{end+1} = 'CELER';
OBJECTIVES{end+1} = L_celer.objectives/lam;
TIMES{end+1} = L_celer.times;
%%
disp('running NcvxPro')

tic
opts    = struct('printEvery', 1, 'pgtol', 1e-3,  'm',10);%, 'maxIts', 5000,'factr', 1e2, 'pgtol', 1e-16, 'maxTotalIts',1e6);
[W,f] = func_MultiTaskLassoPro(X,Y,lam,opts);
tm = toc;

NAMES{end+1} = 'NonCvxPro'; % with linesearch
OBJECTIVES{end+1} = f/lam;
TIMES{end+1} = tm;

%%
display_objectives(OBJECTIVES, NAMES, TIMES, 'NonCvxPro')
% xlim([0,250])
ylim([5e-6,inf])
f = gcf;
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 9., 7.], 'PaperUnits', 'Inches', 'PaperSize', [9., 7.])
exportgraphics(f,sprintf('results/MEG/%s_%d.png',dataset_, factor_),'Resolution',300)
save(sprintf('results/MEG/Matlab_%s_%d.mat',dataset_, factor_), 'OBJECTIVES', 'TIMES','NAMES')
