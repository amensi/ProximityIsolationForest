function tree=ProxIT(train,tree,param)
% Proximity Isolation Forest, v1.0, 2022
% (c) A. Mensi
%
% tree=ProxIT(train,tree,param) recursive function that builds a ProxIT.
%
% The function returns a ProxIT tree structure that can be traversed if distances
% to the training objects are known.
%
% train is the original distance matrix containing pairwise distances between all training objects
%
% tree is a nested tree structure where each level corresponds to a node.
% The most external level corresponds to the root node. The structure, at
% each level, contains the following fields:
%    -idx: contains the indexes of the objects that reached the current
%    node. Necessary to accurately retrieve distances to the prototypes
%    during the testing phase. Less computationaly expensive than storing
%    the whole submatrix.
%   -nsamples: number of objects that have reached the current node.
%   numel(idx) gets the same information.
%   -left (right): pointer to the left (right) child node of the current node.
%   Empty if current node is a leaf.e.
%    -imp: contains the value of the optimization function. Empty if
%    criterion is random or current node is a leaf.
%   -If criterion is 1P, i.e. param.thr=1:
%       -tree.proto contains the index of the prototype defining the test
%       of the current node.
%       -tree.thr contains the index of the threshold  defining the test
%       of the current node.
%   -If criterion is 2P, i.e. param.thr=0:
%       -tree.protoL contains the indx of the left prototype defining the test
%       of the current node.
%       -tree.protoR contains the index of the right prototype  defining the test
%       of the current node.
%
% param is a structure containing several fields which refer to the
% parameters of the ProxIF model. For more information write 'help
% 'ProxIF_training'
% #Comments delimeted by '#' make references to the pseudocode presented in
% the supplementary material of the related paper#

N=length(tree.idx);
tree.nsamples=N;

%Checking whether the current node is a leaf or not
% #Lines 47-53 implement Lines 5-7 of Algorithm 1#
if N==1 || tree.height>=param.max_depth || length(unique(train(tree.idx,tree.idx)))==1
    tree.left=[]; tree.right=[]; tree.imp=[];
    if param.thr %Needed to set the fields according to whether the criterion is 1P or 2P
        tree.proto=[]; tree.thr=[];
    else
        tree.protoL=[]; tree.protoR=[];
    end
else %Node is not a leaf. 
    %Defining the test of the current node and getting the indexes of its
    %children. 
    % #Lines 58-69 implement Lines 9-14 of Algorithm 1#
    if param.thr %Chosen learning strategy is 1P
        [tree.proto,tree.thr,tr_left,tr_right,tree.imp]=OnePLearning(train,tree,param); 
    else %Chosen learning strategy is 2P
        [tree.protoL,tree.protoR,tr_left,tr_right,tree.imp]=TwoPLearning(train,tree,param);
    end
    if isempty(tr_left) || isempty(tr_right) %Deals with those cases where all chosen tests can only create one child node.
        tree.left=[]; tree.right=[]; tree.imp=[];
        if param.thr
            tree.proto=[]; tree.thr=[];
        else
            tree.protoL=[]; tree.protoR=[];   
        end
    else
        %Setting fields of the left and right child node + recursive calls
        %to ProxIT
        % #Lines 74-79 implement Lines 15-20 of Algorithm 1#
        tree.left.height=tree.height+1;
        tree.left.idx=tree.idx(tr_left);
        tree.left=ProxIT(train,tree.left,param);
        tree.right.height=tree.height+1;
        tree.right.idx=tree.idx(tr_right);
        tree.right=ProxIT(train,tree.right,param);
    end  
end
end


