function gH=HausdorffGain(x,L,R,weighted)
% Proximity Isolation Forest, v1.0, 2022
% (c) A. Mensi
% 
% gH=HausdorffGain(x,L,R,weighted) computes the symmetrized Haussdorf
% distance between two sets of objects identified respectively by the vectors L and R
%
% x is the matrix of pairwise distances between the objects of both sets
% 
% L and R are two vectors of indexes indicating the objects in x which
% belong to the L set and to the R set
%
% weighted is an optional parameter to weigh the divergence on the node size

nL=length(L);  nR=length(R); N=nL+nR;
HDLR=max(min(x(L,R),[],2));
HDRL=max(min(x(R,L),[],2));
if weighted
    gH=nL/N*HDLR+nR/N*HDRL;
else
    gH=(HDLR+HDRL)/2;
end
end