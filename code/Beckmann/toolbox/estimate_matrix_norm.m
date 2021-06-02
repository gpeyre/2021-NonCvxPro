function L = estimate_matrix_norm(X)

% estimate spectral norm of X*X'

if size(X,1)<size(X,2)
    C = X*X';
else
    C = X'*X;
end
n = size(C,1);
if n<1000
    L = norm(full(C));
else
    % use power iterations
    u = randn(n,1);  u = u/norm(u);
    L = [];
    for it=1:100
        v = C*u;
        L(end+1) = sum( u .* v );
        u = v/norm(v);
    end
    L = L(end);
end

end