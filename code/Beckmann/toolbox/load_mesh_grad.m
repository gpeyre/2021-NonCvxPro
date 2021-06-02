function [grad,N] = load_mesh_grad(Pos,F)

m = size(F,2); % number faces
n = size(Pos,2);

XF = @(i)Pos(:,F(i,:));
% Compute un-normalized normal through the formula \(e_1 \wedge e_2 \) where \(e_i\) are the edges.
Na = cross( XF(2)-XF(1), XF(3)-XF(1) );
% area of faces
amplitude = @(X)sqrt( sum( X.^2 ) );
Area = amplitude(Na)/2;
% Compute the set of unit-norm normals to each face.
normalize = @(X)X ./ repmat(amplitude(X), [3 1]);
N = normalize(Na);
% Populate the sparse entries of the matrices for the operator implementing \( \sum_{i \in f} u_i (N_f \wedge e_i) \).
I = []; J = []; V = []; % indexes to build the sparse matrices
for i=1:3
    % opposite edge e_i indexes
    s = mod(i,3)+1;
    t = mod(i+1,3)+1;
    % vector N_f^e_i
    wi = cross(XF(t)-XF(s),N);
    % update the index listing
    I = [I, 1:m];
    J = [J, F(i,:)];
    V = [V, wi];
end
% Sparse matrix with entries \(1/(2A_f)\).
dA = spdiags(1./(2*Area(:)),0,m,m);
% Compute gradient.
GradMat = {};
for k=1:3
%    GradMat{k} = sqrt(dA)*sparse(I,J,V(k,:),m,n);
    GradMat{k} = (dA)*sparse(I,J,V(k,:),m,n);   
end
% \(\nabla\) gradient operator.
% Grad = @(u)[GradMat{1}*u, GradMat{2}*u, GradMat{3}*u]';
grad = [GradMat{1}; GradMat{2}; GradMat{3}];

end