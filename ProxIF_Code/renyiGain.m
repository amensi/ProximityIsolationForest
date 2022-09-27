function rG=renyiGain(L,R,NNs,K,alpha,weighted)
% Proximity Isolation Forest, v1.0, 2022
% (c) A. Mensi
% 
% rG=renyiGain(L,R,NNs,K,alpha,weighted) estimates the symmetrized Rényi
% divergence between the candidate children of n.
%
% L and R are the indexes of the objects in n belonging to the putative left and right child node respectively. 
%
% NNs and K are respectively the matrix of nearest neighbor indexes of node n and the
% number of neighbors to use to build the KNN graph. 
%
% alpha is the order of the Rényi divergence
% 
% weighted is an optional parameter to weigh the divergence on the node size

rdR=renyiDivEstimator(L,R,NNs,K,alpha);
rdL=renyiDivEstimator(R,L,NNs,K,alpha);
N=length(L)+length(R);
if weighted
    rG=(length(R)*rdR+length(L)*rdL)/N;
else
    rG=(rdR+rdL)/2;
end
end