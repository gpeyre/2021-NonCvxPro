function [I,J,w,mu] = graph_from_triangle(X,T,q)

% generate a graph from triangulation

n = size(X,2);

% helpers
normalize = @(x)x/sum(x(:));



I = [T(1,:);T(2,:);T(2,:);T(3,:);T(3,:);T(1,:)];
J = [T(2,:);T(1,:);T(3,:);T(2,:);T(1,:);T(3,:)];

% 
[I,J] = deal(I(I<J), J(I<J));

%
IJ = unique([I,J],'rows');
I = IJ(:,1);
J = IJ(:,2);


% edge length
% w = sqrt( (X(1,I)-X(1,J)).^2 + (X(2,I)-X(2,J)).^2 );

w = sqrt( sum( (X(:,I)-X(:,J)).^2 ) );
w = w(:);

% source/sink
[~,s] = min(X(1,:));
d = sqrt( (X(1,s)-X(1,:)).^2 + (X(2,s)-X(2,:)).^2 );
[~,t] = max(d);

% Input measures

v = [s t]; Mu = {};
for i=1:2
    % distance to target
    x = X(:,v(i));
%    d = sqrt( (x(1)-X(1,:)).^2 + (x(2)-X(2,:)).^2 );
    d = sqrt( sum( (x-X).^2 ) );
    [d,Is] = sort(d, 'ascend'); d = d(1:q); Is = Is(1:q);
    Mu{i} = zeros(n,1);
    Mu{i}(Is) = normalize(sqrt( 1-d/(.1+d(end)) ));
end
alpha = Mu{1}; beta = Mu{2};
mu = alpha-beta;


end