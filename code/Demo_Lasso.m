%% This script compares various solvers for Lasso against Nonconvex-Pro 
clearvars
close all
addpath toolbox/
addpath solvers/
addpath datasets/


%% ZeroSR1 code of Becker
%{
Download from: https://github.com/stephenbeckr/zeroSR1
%}
addpath '../packages/Becker'

%% L-BFGS-B
%{
Download from: https://github.com/stephenbeckr/L-BFGS-B-C

Unpack it somewhere and run lbfgsb_C/Matlab/compile_mex.m
%}
addpath '../packages/L-BFGS-B-C-master/Matlab'

%% CGIST
%{
Get CGIST from:
  http://tag7.web.rice.edu/CGIST.html
or http://tag7.web.rice.edu/CGIST_files/cgist.zip
%}
addpath('../packages/cgist');

%% l1_ls
%{
Interior point method by Kwangmoo Koh, Seung-Jean Kim, and Stephen Boyd
Download from: https://web.stanford.edu/~boyd/l1_ls/
%}
addpath('../packages/l1_ls_matlab');

%% Setup a problem
% Before running this script, run python script Demo_Lasso.py to pull datasets from libsvm website
% and saves datasets into .mat files which are loaded here.
%
% Otherwise, download the datasets directly from: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
%
randn('state',234213); rand('state',2342343);

% --- fcn setup ---

i_= 2;

fac_ = -1;
% fac_ = 2;
% fac_ = 10;
% fac_ = 20;
datasets = {'mnist',    'a8a',      'w8a',      'leukemia', ...
    'abalone',  'cadata',   'housing',  'connect-4'};
L = load(sprintf('LassoDatasets/%s.mat',datasets{i_}));
L_celer = load(sprintf('CelerLassoResults/%s%d.mat',datasets{i_},fac_));

A = (L.X);
b = (L.Y);
b = b(:);
lambda = L_celer.lam;
[m,N] = size(A)

At = A';
c = At*b;


% --- Plotting and such ---

NAMES       = {};
OBJECTIVES  = {};
TIMES       = {};
% -------------------------


%%

x0 = randn(N,1);

if issparse(A)
    normQ = normest(A)^2;
else
    normQ = norm(A)^2;
end

lambdaVect = lambda*ones(N,1);
if size(A,1) < size(A,2)% NOTE: the non-standard form (not |Ax-b|, rather <x,Qx> )
    fcn         = @(w) norm(A*w)^2/2 - c'*w+ lambda*norm(w,1);
    fcnSimple   = @(w) norm(A*w)^2/2 - c'*w;
    gradSimple  = @(w) At*(A*w) - c; % doesn't include non-smooth portion
else
    Q           = At*A;
    fcn         = @(w) w'*(Q*w)/2 - c'*w + lambda*norm(w,1);
    fcnSimple   = @(w) w'*(Q*w)/2 - c'*w;
    gradSimple  = @(w) Q*w - c; % doesn't include non-smooth portion
end

% for L-BFGS-B, we will add to gradSimple, since we have made new smooth terms
% for SR1
prox    = @(x0,d,l) prox_l1_rank1( x0, d, l, lambda );

% Setup operators for L-BFGS-B
pos     = @(w) w(1:N,:);
neg     = @(w) w(N+1:2*N,:);
dbl     = @(gg) [gg;-gg];
lambdaVect2     = [lambdaVect;lambdaVect];
fcn2    = @(w) fcnSimple( pos(w) - neg(w) ) + lambdaVect2'*w;
grad2   = @(w) dbl(gradSimple(pos(w)-neg(w))) + lambdaVect2;


%% SR1
if ~exist('zeroSR1_noLinesearch','file')
    disp('Cannot find zeroSR1_noLinesearch on your path, so skipping this test');
else
    disp('Solving via SR1 with l1 constraint ...');
    % fcn and grad are defined above now...
    
    opts = struct('N',N,'verbose',50,'nmax',3000,'tol',1e-12);
    opts.x0     = x0;
    opts.BB     = true;
    opts.theta  = 1; opts.SR1 = true;
    opts.SR1_diagWeight=0.8;
    opts.L      = normQ;
    opts.backtrack = false;
    %%
    try
        opts = rmfield(opts,{'theta','backtrack'});
        tic
        
        [xk,nit, errStruct,optsOut] = zeroSR1_noLinesearch(fcn,gradSimple,prox,opts);
        
        tm = toc;
        
        NAMES{end+1} = '0-mem SR1';
        OBJECTIVES{end+1} = errStruct(:,1);
        TIMES{end+1} = tm;
        
    catch E
        warning('SR1 failed');
    end
end

%% FISTA + BB...
if ~exist('zeroSR1','file')
    disp('Cannot find zeroSR1 on your path, so skipping this test');
else
    disp('FISTA')
    opts.nmax = 5000;
    opts.BB     = true;
    opts.theta  = []; opts.restart=1000; % use [] for FISTA
    opts.SR1 = false;
    opts.backtrack = true;
    tic
    [xk,nit, errStruct,optsOut] = zeroSR1(fcn,gradSimple,prox,opts);
    tm = toc;
    NAMES{end+1} = 'FISTA w/ BB'; % with linesearch
    OBJECTIVES{end+1} = errStruct(:,1);
    TIMES{end+1} = tm;
end
%% SPG/SpaRSA
if ~exist('zeroSR1','file')
    disp('Cannot find zeroSR1 on your path, so skipping this test');
else
    disp('SpaRSA')
    opts.nmax = 5000;
    
    opts.BB     = true;
    opts.theta  = 1;
    opts.SR1 = false;
    opts.backtrack = true;
    tic
    [xk,nit, errStruct,optsOut] = zeroSR1(fcn,gradSimple,prox,opts);
    tm = toc;
    NAMES{end+1} = 'SPG/SpaRSA'; % with linesearch
    OBJECTIVES{end+1} = errStruct(:,1);
    TIMES{end+1} = tm;
end
%% Run L-BFGS-B

%{
Solve min L(x) + lambda*||x||_1 by formulating as:
  min_{z,y} L(z-y) + ones(2N,1)'*[z,y]
s.t.
z,y >= 0. i.e. "x" is z - y
  

if we switch to simple x >= 0 formulation, then it solves it in 2 steps!!

%}
if ~exist('lbfgsb','file')
    disp('Cannot find lbfgsb on your path, so skipping this test');
else
    disp('Solving via L-BFGS-B...');
    
    fun     = @(x)fminunc_wrapper( x, fcn2, grad2);
    opts    = struct( 'factr', 1e4, 'pgtol', 1e-12, 'm', 5, 'maxIts', 5000, 'maxTotalIts',1e6 );
    opts.x0 = [max(x0,0); -min(x0,0)];
    opts.printEvery     = 50;
    opts.factr          = 1e1; % more accurate soln
    if N > 200
        opts.factr = 1e-2;
        opts.pgtol = 1e-14;
    end
    
    tic
    [x2, ~, info] = lbfgsb(fun, zeros(2*N,1), inf(2*N,1), opts );
    tm = toc;
    x   = pos(x2) - neg(x2);
    
    NAMES{end+1} = 'L-BFGS-B';
    OBJECTIVES{end+1} = info.err(:,1);
    TIMES{end+1} = tm;
    
end
%% run cgist
if ~exist('cgist','file')
    disp('Cannot find cgist on your path, so skipping this test');
else
    % solves ||Ax-f||^2 + lambda*|x|_1
    % So, from <x,Qx>/2 -c'*x format, we have
    %
    regularizer = 'l1';
    opts = [];
    %     opts.tol = 1e-10;
    opts.record_objective = true;
    opts.record_iterates = false; % big!
    opts.errFcn = [];
    opts.guess = x0;
    tic
   
        [xk, multCount, subgradientNorm, out] = cgist(A,[],b,lambda,regularizer,opts);

    tm = toc;
    % need to subtract norm(b)^2/2 to get objective fcn to line up
    out.objectives = out.objectives - norm(b)^2/2;
    
    NAMES{end+1} = 'CGIST';
    OBJECTIVES{end+1} = out.objectives;
    TIMES{end+1} = tm;
end


%% l1_ls, Interior point method for l1 by Koh, Kim and Boyd
if ~exist('l1_ls','file')
    disp('Cannot find l1_ls on your path, so skipping this test');
else
    tic
    % rel_tol=1e-7;
        [x_ip,status,history] = l1_ls(A,b,2*lambda);
    ip_toc = toc;
    f_ip = history(2,:)/2-norm(b)^2/2;
    
    NAMES{end+1} = 'interior-point';
    OBJECTIVES{end+1} = f_ip;
    TIMES{end+1} = ip_toc;
    
end
%% Varpro with eta trick
opts = struct('printEvery', 1,'maxIts', 10000, 'm', 5, 'x0', abs(x0) ); %options for inner lbfgs

if m<N
    tic
    [x_lbfgs2,f_lbfgs2,g_lbfgs2]  = func_QuadVar(A,A',b,lambda,N,opts);
    lbfgs_toc2=toc
else
    tic
    Q = A'*A;
    [x_lbfgs2,f_lbfgs2,g_lbfgs2]  = func_QuadVar_Cov(Q,A,b,lambda,opts);
    lbfgs_toc2=toc
end

NAMES{end+1} = 'Quad-variational';
OBJECTIVES{end+1} = f_lbfgs2-norm(b)^2/2;
TIMES{end+1} = lbfgs_toc2;


%% %alternating minimisation on nonconvex splitting
if strcmp(datasets{i_},'abalone')
    maxits = 3000;
else
    maxits = 1000;
end
tic
[x_alt,res_alt,f_alt]  = func_LassoAlternating(A,At,b,lambda,maxits,N,x0);
altern_toc=toc

NAMES{end+1} = 'Non-cvx-Alternating-min';
OBJECTIVES{end+1} = f_alt;
TIMES{end+1} = altern_toc;

%% L-BFGS to minimise the nonconvex splitting, minimising both variables simultaneously
opts = struct('printEvery', 1,'maxIts', 10000, 'm', 5 ); %options for inner lbfgs

opts.factr          = 1e1; % more accurate soln
if N > 200
    opts.factr = 1e-2;
    opts.pgtol = 1e-14;
end

tic
[x,f_lbfgs2,g]  = func_LassoLbfgsnoncvx(A,b,lambda,opts);
lbfgs_toc2=toc;
NAMES{end+1} = 'Non-cvx-LBFGS';
OBJECTIVES{end+1} = f_lbfgs2;
TIMES{end+1} = lbfgs_toc2;

%% Add recorded CELER times

NAMES{end+1} = 'CELER';
OBJECTIVES{end+1} =L_celer.objectives-norm(b)^2/2;
TIMES{end+1} = L_celer.times;


%% Varpro with nonconvex splitting


opts = struct('printEvery', 1,'maxIts', 10000, 'm', 5,'x0',x0); %options for inner lbfgs
[m1,m2] = size(A);
if m1>m2
    Q = A'*A;
    tic
        [x_lbfgs,f_lbfgs,g_lbfgs]  = func_LassoPro_cov(Q,c,lambda,opts);
    OBJECTIVES{end+1} = f_lbfgs;
else
    tic
    [x_lbfgs,f_lbfgs,g_lbfgs]  = func_LassoPro(A,A',b,lambda,N,opts);
    OBJECTIVES{end+1} = f_lbfgs*lambda-norm(b)^2/2;
end

lbfgs_toc=toc
NAMES{end+1} = 'Noncvx-Pro';
TIMES{end+1} = lbfgs_toc;

%% PLOT EVERYTHING

newcolors=turbo(length(OBJECTIVES));
all_marks = {'o','+','*','<','x','p','d','^','v','>','.','s','h','o','+','*','<'};
fig_=figure(1); clf;
obj_best = Inf;
for k = 1:length(OBJECTIVES)
    if min( OBJECTIVES{k})<obj_best
        disp(k)
        
    end
    obj_best = min(obj_best, min( OBJECTIVES{k}) );
end
mh_ = [];
for k = 1:length(NAMES)
    tGrid = linspace(0,TIMES{k},length(OBJECTIVES{k}));
    if strcmp(NAMES{k},'Noncvx-Pro')
        h=semilogy( tGrid, cummin( OBJECTIVES{k} - obj_best)/(m*lambda), 'b-.','DisplayName', NAMES{k}  );
        set(h,'linewidth',4);
        hold on
    else
        obj_ = cummin(  OBJECTIVES{k} - obj_best)/(m*lambda);
        
        sp_ = ceil(length(obj_)/10);
        if TIMES{k}>2 %may need to adjust this for a nice figure
            sp_ = ceil(length(obj_)/10);
        end
        if TIMES{k}>20
            sp_ = ceil(length(obj_)/10);
        end
        h = semilogy( tGrid(1:sp_:end), obj_(1:sp_:end),'Color', newcolors(k,:),'LineStyle', 'none' , 'Marker',  all_marks{k}, 'Markersize',10  );
        hold on
        h1=semilogy( tGrid, obj_, 'Color', newcolors(k,:),'DisplayName', NAMES{k} );
        set(h1,'linewidth',1.5);
        set(h,'linewidth',2);
    end
    mh_ = [mh_, h];
    
end
legend(mh_,NAMES, 'location','southeast')
xlabel('time in seconds','fontsize',18);
ylabel('objective value error','fontsize',18);
set(gca,'fontsize',18)
% xlim([0,15])
ylim([1e-11,inf ])


f = gcf;
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 9., 7.], 'PaperUnits', 'Inches', 'PaperSize', [9., 7.])
exportgraphics(f,sprintf('results/LassoBench/libsvm-%s%d.png',datasets{i_},fac_),'Resolution',300)
