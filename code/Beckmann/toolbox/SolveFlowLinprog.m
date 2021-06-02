function [u,f] = SolveFlowLinprog(I,J,w,mu)

% Solve Primal and Dual W1 flow problems 
%       u <=> s/2
%       f <=> fp-fm

n = max([I;J]);

% gradient operator
p = length(I);
grad = sparse( [1:p 1:p], [I;J]', [ones(p,1) -ones(p,1)], p,n);

% max <u,mu> s.t.  |grad(u)| <= w
cvx_solver sdpt3 % SeDuMi %
cvx_begin quiet % sdp quiet
cvx_precision high;
variable u(n,1); %  real;
% norm( grad*u, Inf ) <= 1;
abs( grad*u ) <= w;
maximize( sum( u .* mu ) );
cvx_end


% min |w.*f|_1 s.t.   div(f)=mu-nu
cvx_solver sdpt3 % SeDuMi %
cvx_begin quiet % sdp quiet
cvx_precision high;
variable f(p,1); %  real;
(grad'*f) - mu  == 0;
minimize( sum( abs(f).*w ) );
% maximize( sum( abs(f(:)).*w(:) ) );
cvx_end

end
