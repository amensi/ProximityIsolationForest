function forest=ProxIF_training(train,param)
% Proximity Isolation Forest, v2.0, 2022
% (c) A. Mensi 
%
% forest=ProxIF_training(train,param) creates a Proximity Isolation Forest (ProxIF) of param.T
% Proximity Isolation Trees from train.
%
% train is a distance matrix NxN containing pairwise distances.
% 
% param is a structure containing the following fields:
%   -T: the number of Proximity Isolation Trees (ProxIT) in a ProxIF.
%           DEFAULT: T=500.
%   -S: the number of objects used to build a ProxIT. The objects are randomly subsamples without replacement
%       from train for each ProxIT in an independent way.
%           DEFAULT: S=128.
%   -D: the maximum depth reachable in each ProxIT. It can either
%       be 'log', i.e. ceil(log_2(S)), or 'max', i.e. S-1.
%           DEFAULT: 'log'
%   -crit: the training strategy to adopt to build each ProxIT. It can
%       be one among 'R-1P', 'R-2P', 'O-1PSD', 'O-2PSD', 'O-2PSP', 'O-1PH', 'O-2PH',
%       'O-1PRD', 'O-2PRD'. We recall that: '1P' stands for one prototype
%       whereas '2P' for two prototypes; 'R' stands for random', whereas
%       'O' for optimized; 'SD' and 'SP' stand for the two types of
%       scatter, whereas 'H' stands for Hausdorff distance and 'RD' for
%       Rényi Divergence. 
%           DEFAULT: 'O-2PH'.
%   -max_comp: the number of tests to be evaluated in each node. . 
%           DEFAULT: R=20.
%   -alpha: optional field to be set when param.crit is either 'O-1PRD' or 'O-2PRD'.    
%           It is the order of the Rényi divergence to be estimated.            
%           DEFAULT: alpha=0.9999.
% #Comments delimeted by '#' make references to the pseudocode presented in
% the supplementary material of the related paper#

if nargin<2
    param.T=0; %Assigning one field randomly that I know to exist
end
if size(train,1)~=size(train,2)
    error("You should provide a pairwise distance matrix as input");
end

%Checking correctness of forest parameters (and eventually setting them)
N=size(train,1); % #Line 5 of Algorithm 4#
param=setProxIFParameters(param,N); % #This function also makes the check of Line 6 of Algorithm 4#

%Training
tree=cell(1,param.T);
% #FOR loop below implements Lines 7-14 of Algorithm 4#
for t=1:param.T
    tree{t}.height=0; 
    if param.S~=N
        tree{t}.idx=randsample(N,param.S); %Necessary to correctly retrieve distances during the testing phase.
    else
        tree{t}.idx=1:N; 
    end
        tree{t}= ProxIT(train,tree{t},param);
end
    forest.tree=tree;
    forest.param=param;
end



