%% L-BFGS-B
%{
Download from: https://github.com/stephenbeckr/L-BFGS-B-C

Unpack it somewhere and run lbfgsb_C/Matlab/compile_mex.m
%}
addpath '../../packages/L-BFGS-B-C-master/Matlab'

%
randn('state',234213); rand('state',2342343);
rng(234213, 'twister')
%% load schools data
datasetpath = 'datasets';
dataset = 'synthetic';
dataset = 'robot';
% dataset = 'parkinsons';
% dataset = 'schools';

switch dataset
    case 'synthetic'
        m = 10000; %samples
        T = 5; %tasks %lamOpt =14.3919
        T = 10; %lamOpt = 14.3919
        T = 50; %lamOpt = 22.4460
        T = 100; %lamOpt = 22.4460

        d = 30; %features
        indx = sort(randperm(m,T)); indx(end+1) = m+1;
        while min(diff(indx))<10 %at least 10 examples per task
        indx(1:T) = sort(randperm(m,T)); 
        end
        X = rand(d,m);
        s=5;
        W0 = randn(d,T); W0(randperm(d,d-s),:) = 0;
        y = randn(m,1);
        for i=1:T
            i1 = indx(i); i2 = indx(i+1)-1;
            y(i1:i2) = y(i1:i2) + X(:,i1:i2)'*W0(:,i);
        end
        lamvals =exp(linspace(0,4,10));
%         lamOpt = 10;
        fac_ = -1;
%         fac_ = 10;
        lamMax = max(checkOptimality(X,y,indx(1:end-1),W0*0,1));
        if fac_>0
            lamOpt = lamMax/fac_;   
        else %crossvalidate
            opts    = struct('printEvery', -1,  'm', 5, 'maxIts', 50 );
            trainer = @(Xtr,ytr,indtr,lambda) func_MultifeatPro(Xtr,ytr,indtr,lambda,opts);
            [Wopt,lamOpt,optVal,errorcurve] = func_train_mkfeat_noncvx(X,y,indx(1:end-1),5,lamvals,trainer);
            lamOpt
        end
    case 'schools'
        %download from https://home.ttic.edu/~argyriou/code/
        Ldata = load([datasetpath,'/school_splits/school_b.mat']);
        split_path = [datasetpath, '/school_splits/school_%d_indexes.mat'];
        lamvals =exp(linspace(3,7,20));%schools
        lamOpt = 251.2167;
        crossvalidate = 1;
    case 'parkinsons'
        %download from﻿http://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring
        %to make the datasplits, run toolbox/readData_parkinsons.m
        Ldata = load([datasetpath,'/parkinsons/fulldata.mat']);
        split_path = [datasetpath,'/parkinsons/split%d.mat'];
        lamvals =exp(linspace(-1,2.5,30));%parkinsons
        lamOpt = 2.0694;
        crossvalidate = 1;
    case 'robot'
        %download The SARCOS data from ﻿http://www.gaussianprocess.org/gpml/data/
        %to make the datasplits, run toolbox/readData_robot.m
        Ldata = load([datasetpath,'/robot/fulldata.mat']);
        split_path = [datasetpath,'/robot/split%d.mat'];
        lamvals =exp(linspace(-3,2,30));%robot
        lamOpt =  3.6325;
        Data = load([datasetpath,'/robot/sarcos_inv.mat']).sarcos_inv;
        Data2 = load([datasetpath,'/robot/sarcos_inv_test.mat']).sarcos_inv_test;
        Data = [Data;Data2];
        X0 = Data(:,1:21);
        y0 = Data(:,22:28);
        
        XtX = X0'*X0;
        Xty = X0'*y0;
        crossvalidate = 1;
end

%
%%
if ~strcmp(dataset, 'synthetic')
    X = Ldata.x;
    y = Ldata.y;
    y = y(:,1);
    indx = Ldata.task_indexes;
    
    T = length(indx);
end

%% set options

%LBFGS
opts    = struct('printEvery', -1,  'm', 5, 'maxIts', 500 );

% %alternating
opts.maxits = 100;
opts.tol = 1e-6;
opts.eps = 1e-4;
opts.decrease_eps = 0;

%% find lambda value by cross validation, averaged across 10 random data splits.
if ~strcmp(dataset, 'synthetic') && crossvalidate
    if strcmp(dataset,'robot')
        trainer = @(Xtr,ytr,indtr,lambda) func_MultiTaskLearning_Alternating(Xtr,ytr,indtr,lambda,opts);
    else
        trainer = @(Xtr,ytr,indtr,lambda) func_MultifeatPro(Xtr,ytr,indtr,lambda,opts);
    end
    OptVals = [];
    weight = [];
    Curve = zeros(size(lamvals));
    lamOpt = 0;
    for i=1:10
        Lindx = load(sprintf(split_path,i));
        tr = Lindx.tr;
        Xtr = X(:,tr);
        ytr = y(tr);
        tr_indexes = Lindx.tr_indexes;
        tst = Lindx.tst;
        Xtst = X(:,tst);
        ytst = y(tst);
        tst_indexes = Lindx.tst_indexes;
        T = length(indx);
        [Wopt,lamOpti,optVal,errorcurve] = func_train_mkfeat_noncvx(Xtr,ytr,tr_indexes,10,lamvals,trainer);
        [~,rmse_i] = testErrors(Wopt, Xtst,ytst,y,tst_indexes,indx);
        
        rmse_i
        OptVals(i) = rmse_i;
        Curve = errorcurve+Curve;
        lamOpt = lamOpt+lamOpti;
        
    end
    lamOpt = lamOpt/10;
    %%
    figure
    plot(lamvals,Curve/10, 'k','linewidth',2)
    xlabel('\lambda','Fontsize',24)
    ylabel('RMSE','Fontsize',24)
    axis tight
    f = gcf;
    set(gcf, 'Units', 'Inches', 'Position', [0, 0, 9., 7.], 'PaperUnits', 'Inches', 'PaperSize', [9., 7.])
    
    exportgraphics(f,sprintf('results/validation%s.jpg',dataset),'Resolution',300)
end
%%
NAMES ={};
TIMES = {};
OBJECTIVES = {};

lambda0 =lamOpt;

%% FISTA
opts.maxits = 50000; %schools %parkinsons
opts.maxits = 5000;
opts.tol = 1e-12;
lambda =lambda0;


%first run without recording objectives to get time
tic
opts.recordobj = 0;
if strcmp(dataset, 'robot')
    [W_FISTA,fval_FISTA]  = func_MTL_fista_XtX(XtX,Xty,lambda,opts);
else
    [W_FISTA,fval_FISTA] = func_MTL_fista(X,y,indx,lambda,opts);
end
t_FISTA = toc

%then run to record objectives
opts.recordobj = 1;
if strcmp(dataset, 'robot')
    [W_FISTA,fval_FISTA]  = func_MTL_fista_XtX(XtX,Xty,lambda,opts);
else
    [W_FISTA,fval_FISTA] = func_MTL_fista(X,y,indx,lambda,opts);
end
OBJECTIVES{1} = fval_FISTA/(size(X,2)*lamOpt);
NAMES{1} = 'FISTA';
TIMES{1} = t_FISTA;




%% alternating minimisation
if strcmp(dataset, 'robot')
    epsvals = 4:2:10;
else
    epsvals = 2:5;
end
for j = 1:length(epsvals)
    dd = epsvals(j);
    lambda = lambda0;
    opts.decrease_eps = 0;
    opts.maxits = 10000; %parkinsons %schools
    opts.maxits = 100;
%     if j<3
%     opts.maxits = 100;
%     end
    
    opts.tol = -1e-10;
    opts.eps = 10^-dd;
    tic
    if strcmp(dataset, 'robot')
        [W_IRLS,f_IRLS] = func_MultiTaskLearning_IRLS_XtX(XtX,Xty,lambda,opts);
    else
        [W_IRLS,f_IRLS] = func_MultiTaskLearning_Alternating(X,y,indx,lambda,opts);
    end
    toc_IRLS=toc
    
    OBJECTIVES{end+1} = f_IRLS/(size(X,2)*lamOpt);
    NAMES{end+1} = sprintf('IRLS-%d', dd);
    TIMES{end+1} = toc_IRLS;
end

%%
Opts    = struct('printEvery', -1,  'm', 5, 'maxIts', 1000 ,'factr', 1e2); %options for inner lbfgs
lambda =lambda0;
tic
if strcmp(dataset, 'robot')
    [W_PRO,f_PRO] = func_multifeatPro_XtX(XtX,Xty,lambda,Opts);
else
    [W_PRO,f_PRO,o] = func_MultifeatPro(X,y,indx,lambda,Opts);
end

toc_PRO = toc

OBJECTIVES{end+1} = f_PRO/(size(X,2)*lamOpt);
NAMES{end+1} = 'Noncvx-Pro';
TIMES{end+1} = toc_PRO;

%%
display_objectives(OBJECTIVES, NAMES, TIMES, 'Noncvx-Pro-')
%%
% xlim([0,.2])
ylim([1e-9,inf])
f = gcf;
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 9., 7.], 'PaperUnits', 'Inches', 'PaperSize', [9., 7.])

if strcmp(dataset, 'synthetic')
    
    exportgraphics(f,sprintf('results/mtfeat-%s-m%d-T%d-d%d-fac%d.png',dataset,m,T,d,fac_),'Resolution',300)
    
else
    exportgraphics(f,sprintf('results/mtfeat-%s.png',dataset),'Resolution',300)
end


%% display coefficients

figure(4)
clf
c1 = min([W_PRO(:);W_FISTA(:);W_IRLS(:)]);
c2 = max([W_PRO(:);W_FISTA(:);W_IRLS(:)]);

subplot(1,3,1)
imagesc(W_PRO);
title('Noncvx-Pro')
caxis([c1 c2]);
subplot(1,3,2)

imagesc(W_IRLS);
title('IRLS')

caxis([c1 c2]);

subplot(1,3,3)

imagesc(W_FISTA);
title('FISTA')

caxis([c1 c2]);
colorbar('Position',[0.04 .05 0.03 .9])
f = gcf;
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 16., 3.], 'PaperUnits', 'Inches', 'PaperSize', [16.25, 3.])

exportgraphics(f,sprintf('results/map%s.png',dataset),'Resolution',300)
