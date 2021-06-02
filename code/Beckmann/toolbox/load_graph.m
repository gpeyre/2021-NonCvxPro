function [I,J,c,y,T,Pos] = load_graph(name)

% load_graph - loading the graph
%
%   [I,J,c,y,T,Pos] = load_graph(name)

% number of point in the support
q = 1; % just dirac mass
q = 25;
q = 14;
%q = 6;
T = []; I = [];
switch name
    case 'yeast'
        %  Analyzing Gene Expressions in Baker's Yeast
        % DeRisi, JL, Iyer, VR, Brown, PO. 
        % "Exploring the metabolic and genetic control of gene expression on a genomic scale." 
        % Science. 1997
        % 7-D points
        load data/yeastdata
        sub = 1; % sub-sampling
        % keep only genes with whole data
        X = X(not(isnan(sum(X,2))),:)';
        X = X(:,1:sub:end);
        Pos = 20*pca(X)'; Pos = Pos(1:2,:);
        n = size(Pos,2);
        % K-nn graph
        D = distmat(Pos,Pos);
        [C,V] = sort(D,'ascend'); 
        k = 5; % NN
        J = V(2:k+1,:); J = J(:);
        c = C(2:k+1,:); c = c(:)*0+1;
        I = repmat(1:n, [k 1]); I = I(:);
        % clean graph
        [I,J] = deal([I;J],[J;I]);         
        [I,J] = deal(I(I<J), J(I<J));
        IJ = unique([I,J],'rows');
        I = IJ(:,1);
        J = IJ(:,2);
        %
        c = .1*rand(size(I))+1;
        t = .07;
        c = D(I + (J-1)*n); c = t+(1-t)*c/max(c);
        % create a measure
        s = .3;
        %
        q = 40;
        % gamma_list = [.1]; sigma_list = [.1]; 
        m = {{[-1;-.9]} {[.9;-.7], [.9,2.15]}}; 
        %
        q = 100;
        % gamma_list = [.1]; sigma_list = [.1]; 
        m = {{[-1;-.8]} { [.9;-.7] [.85,.35] [.9 1.1] [.9,2.15] [.3 2.2]}}; 
        qlist = {[100], [10 10 25 25 25]};
        %
        y = zeros(n,1); 
        for k=1:2
            for l=1:length(m{k})
                q1 = round(q/length(m{k}));
                q1 = qlist{k}(l);
                [~,i] = min( sum( (Pos-m{k}{l}(:)).^2 ) );
                y(V(1:q1,i)) = (-1)^k / sum(qlist{k});
            end
        end
        y = y-mean(y);
    case 'planar'
        n = 200; % large
        n = 10; %mini
        n = 80; % small 
        Pos = randn(2,n);
        Pos(2,:) = Pos(2,:)*.75;
        T = delaunay(Pos(1,:),Pos(2,:))';
    otherwise
        global meshname
        meshname = name;
        [Pos,T] = read_off([name '.off']);
end
d = size(Pos,1);
if isempty(I)
    [I,J,c,y] = graph_from_triangle(Pos,T,q);
end

end