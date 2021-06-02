function [P,f,g,Err,Costs] = sinkhorn(C,a,b,epsilon,options)

a = a(:); b = b(:)';
n = length(a);

mina = @(H,epsilon)-epsilon*log( sum(a .* exp(-H/epsilon),1) );
minb = @(H,epsilon)-epsilon*log( sum(b .* exp(-H/epsilon),2) );
mina = @(H,epsilon)mina(H-min(H,[],1),epsilon) + min(H,[],1);
minb = @(H,epsilon)minb(H-min(H,[],2),epsilon) + min(H,[],2);

niter = getoptions(options, 'niter', 1000);
tol = getoptions(options, 'tol', 0);

Costs = {};

suma = @(x)sum(x(:));
 
f = zeros(n,1);
Err = [];
for it=1:niter
    g = mina(C-f,epsilon);
    P = a .* exp((f+g-C)/epsilon) .* b; 
    Costs{1}(it,1) = suma(C.*P); % sharp sinkh
    Costs{1}(it,2) = suma(C.*P) + epsilon * suma( P .* log(P./(a*b)) ); % primal
    Costs{1}(it,3) = sum(a.*f) + sum(b.*g); % dual 
    %    
    f = minb(C-g,epsilon);
    P = a .* exp((f+g-C)/epsilon) .* b;   
    Costs{2}(it,1) = suma(C.*P); % sharp sinkh
    Costs{2}(it,2) = suma(C.*P) + epsilon * suma( P .* log(P./(a*b)) ); % sharp sinkh
    Costs{2}(it,3) = sum(a.*f) + sum(b.*g); % dual
    Err(it) = norm(sum(P,1)-b,1); 
    if Err(end)<tol
        break
    end
end

end