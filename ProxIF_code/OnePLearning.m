function [proto_best,thr_best,tr_left,tr_right,I]=OnePLearning(train,tree,param)
% Proximity Isolation Forest, v1.0, 2022
% (c) A. Mensi
% [proto_best,thr_best,I,tr_left,tr_right]=OnePLearning(train,tree,param,mComp)
% learning step adopted when a 1P criterion is chosen.
% The function returns the elements defining the split of the node: proto_best and thr_best, i.e., the prototype and
% threshold, along with the indexes of the objects ending up in the left and
% right child node (tr_left abd tr_right).
% It also returns the value of the impurity function corresponding to the best split.
%
% If the criterion is R-1P then proto_best and thr_best are chosen randomly; otherwise
% proto_best and thr_best are the best prototype and threshold among mComp
% pairs of candidates according to the maximization of an impurity-like
% function.
% #Comments delimeted by '#' make references to the pseudocode presented in
% the supplementary material of the related paper#

N=tree.nsamples;
proto_best=[]; thr_best=[]; tr_left=[]; tr_right=[]; I=[];

% #The first branch of the following IF statement implements Lines 5-9 of Algorithm 2#
if strcmpi(param.crit,'R-1P') %Random criterion
    proto_best=tree.idx(randperm(N,1));
    range=sort(train(tree.idx,proto_best),'ascend');
    if length(unique(range))~=1 %True if all objects at the same distance from P.
        a=range(1); b=range(end);
        while isempty(tr_left)|| isempty(tr_right)
            thr_best=a+(b-a).*rand(1);
            tr_left=find(train(tree.idx,proto_best)<=thr_best);
            tr_right=find(train(tree.idx,proto_best)>thr_best);
        end
    end

% #The second branch of the IF statement implements Lines 10-24 of Algorithm 2#
else %Optimized criterion
    %Choosing the tests to evaluate among all possible tests. This is not needed if criterion is R-1P.
    % #The IF statement below implements Lines 11-12 of Algorithm 2#
    maxComp=(N*N-N); %Do not divide by two in order to manage unsymmetric matrices.
    if param.max_comp>maxComp
        mComp=maxComp;
    else
        mComp=param.max_comp;
    end

    % #The block of code below implements Lines 13-14 of Algorithm 2#
    [P1,P2]=ndgrid(1:N);
    protosIdx=[P1(:) P2(:)];
    protosIdx(protosIdx(:,1)==protosIdx(:,2),:)=[];
    protosIdx=protosIdx(randperm(maxComp,mComp),:);
    protos=tree.idx(protosIdx(:,1));
    thrs=tree.idx(protosIdx(:,2));

    %Test Evaluation
    % #The FOR loop below implements Lines 15-24 of Algorithm 2#
    I=-Inf;
    for pp=1:length(protos)
        %Creation of the putative child nodes
        % #The following block of code implements Lines 17-18 of Algorithm 2#
        if  size(unique(train(tree.idx,protos(pp)),'rows'),1)==1
            continue
        end
        thr=train(thrs(pp),protos(pp));
        trleft=find(train(tree.idx,protos(pp))<=thr);
        trright=find(train(tree.idx,protos(pp))>thr);
        if isempty(trleft) || isempty(trright)
            continue
        end

        %Function that evaluates the created children according to param.criterion
        I_tmp=EvaluateTest(train,tree,trleft,trright,param); % #Implements Lines 19-21 of Algorithm 2#
        % #IF statement below implements Lines 22-24 of Algorithm 2#
        if I_tmp>I %The impurity of the test under evaluation is better than the current best one.
            I=I_tmp;
            tr_left=trleft; tr_right=trright;
            proto_best=protos(pp); thr_best=thr;
        end
    end
end
end