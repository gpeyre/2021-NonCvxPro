function [t,f,g] = func_lbfgs(Efun,t0,mopts)
opts    = struct('printEvery', 100, 'x0', t0,'factr', 1e4, 'pgtol', 1e-16, 'm', 10, 'maxIts', 500, 'maxTotalIts',1e9 );
opts.pos = 0;
if nargin==3
    fields = fieldnames(mopts);
    for k = 1:numel(fields)
        aField     = fields{k};
        opts.(aField) = mopts.(aField);
    end
end
%run lbfgs
fun     = @(x)fminunc_wrapper2( x, Efun);

k = length(t0);

if opts.pos
    l =  zeros(k,1);
else
    l = -inf(k,1);
end
u = inf(k,1);% there is no upper bound

[t, ~, R] = lbfgsb(fun, l, u, opts );
f = R.err(:,1); %function values
g = R.err(:,2);%gradient values in inf norm

end



