function rd=renyiDivEstimator(X,Y,NNs,K,alpha)
% Proximity Isolation Forest, v1.0, 2022
% (c) A. Mensi
% 
% rd=renyiDivEstimator(X,Y,NNs,K,alpha) estimates the Rényi
% divergence of the set of objects identified by Y from the set of objects identified by X
%
% X and Y are two vectors of indexes indicating respectively the two sets
% under analysis
%
% NNs is the matrix of nearest neighbors computed for XUY
%
% K is the number of neighbors to use to build the KNN graph. 
%
% alpha is the order of the Rényi divergence

N=length(X); M=length(Y);
eta=M/N;

labels=zeros(N+M,1);
labels(X)=2; labels(Y)=1;
Ne=NNs(Y,2:K+1); %to exclude the object itself
labNe=labels(Ne); Mi=sum(labNe==1,2); Ni=sum(labNe==2,2);
rd=1/(alpha-1)*log(eps+(eta^alpha)/M*sum((Ni./(Mi+1)).^alpha));
end