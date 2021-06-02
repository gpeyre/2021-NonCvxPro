function display_flow(X,T,I,J, mu, mu1, f)

% mu is target distribution, mu1 is current one, f is some flow function.

if nargin<6
    mu1 = [];
end
if nargin<7
    f = [];
end

d = size(X,1);
n = max([I;J]);
p = length(I);
lw = 8; 
col = [0 .5 0];
if isempty(f)
    f = ones(p,1);
    lw = 2; 
    col = [0 0 0];
end
if isempty(mu1)
    mu1 = mu;
    mu = [];
end

switch d
    case 2
        myplot = @(z, c, l)plot(z(1,:), z(2,:), 'color', c, 'LineWidth', l);
        myscatter  = @(X,s,CM)scatter( X(1,:),X(2,:), s, CM, 'filled' );
        myscatter1 = @(X,s,CM)scatter( X(1,:),X(2,:), s, 'k', 'LineWidth', 2 );
    case 3
        myplot = @(z, c, l)plot3(z(1,:), z(2,:), z(3,:),'color', c, 'LineWidth', l);
        myscatter  = @(X,s,CM)scatter3( X(1,:),X(2,:),X(3,:), s, CM, 'filled' );
        myscatter1 = @(X,s,CM)scatter3( X(1,:),X(2,:),X(3,:), s, 'k', 'LineWidth', 2 );
    otherwise
        error('Only implemented in 2D and 3D.');
end

hold on;
if d==3
    global meshname
    opt.name = meshname;
    plot_mesh(X,T, opt);
end


f = f/max(f);
hold on;
for k=1:length(I)
    c = f(k);
    % plot([X(1,I(k));X(1,J(k))],[X(2,I(k));X(2,J(k))], 'color', c*col + (1-c)*( col*.3+.7 ), 'LineWidth', .01 + lw*c);
    if c>1e-2
        myplot([X(:,I(k)),X(:,J(k))], c*col + (1-c)*( col*.3+.7 ), .01 + lw*c);
    else
        myplot([X(:,I(k)),X(:,J(k))], [1 1 1]*0, .01 + lw*c);        
    end
end
% current flow
m = .5 + .5*mu1/max(abs(mu1)); CM = m*[1 0 0] + (1-m)*[0 0 1];
myscatter( X, .01 + 75*abs(mu1)/max(abs(mu1)), CM );
% target flow
if not(isempty(mu))
    m = .5 + .5*mu/max(abs(mu)); CM = m*[1 0 0] + (1-m)*[0 0 1];
    myscatter1( X, .01 + 150*abs(mu)/max(abs(mu)) );
end
axis equal; axis off;

end