function [f,g] = fminunc_wrapper2(x,F)
% [f,g,h] = fminunc_wrapper( x, F, G, H, errFcn )
% for use with Matlab's "fminunc"
%
% [fHist,errHist] = fminunc_wrapper()
%       will return the function history
%       (and error history as well, if errFcn was provided)
%       and reset the history to zero.
persistent errHist fcnHist nCalls
if nargin == 0
   f = fcnHist(1:nCalls);
   g = errHist(1:nCalls);
   fcnHist = [];
   errHist = [];
   nCalls  = 0;
   return;
end
if isempty( fcnHist )
    [errHist,fcnHist] = deal( zeros(100,1) );
end

fg = F(x);
f = fg(1);
g = fg(2:end);
% Record this:
nCalls = nCalls + 1;
if length( errHist ) < nCalls
    % allocate more memory
    errHist(end:2*end) = 0;
    fcnHist(end:2*end) = 0;
end
fcnHist(nCalls) = f;



