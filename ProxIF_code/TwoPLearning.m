function [protoL_best,protoR_best,tr_left,tr_right,I]=TwoPLearning(train,tree,param)
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
protoL_best=[]; protoR_best=[]; tr_left=[]; tr_right=[]; I=[];

% #Lines 23-27 implement Lines 5-9 of Algorithm 3#
if strcmpi(param.crit,'R-2P') %Random criterion
    temp=randperm(N,2);
    protoL_best=tree.idx(temp(1)); protoR_best=tree.idx(temp(2));
    tr_left=find(train(tree.idx,protoL_best)<=train(tree.idx,protoR_best));
    tr_right=find(train(tree.idx,protoL_best)>train(tree.idx,protoR_best));

    % #Lines 29-70 implement Lines 10-24 of Algorithm 3#
else %Optimized criterion
    %Choosing the tests to evaluate among all possible tests. This is not needed if criterion is R-2P.
    % #Lines 32-37 implement Lines 11-12 of Algorithm 3#
    maxComp=(N*N-N); %Do not divide by two in order to manage unsymmetric matrices.
    if param.max_comp>maxComp
        mComp=maxComp;
    else
        mComp=param.max_comp;
    end

    % #Lines 40-45 implement Lines 13-14 of Algorithm 3#
    [P1,P2]=ndgrid(1:N);
    protosIdx=[P1(:) P2(:)];
    protosIdx(protosIdx(:,1)==protosIdx(:,2),:)=[];
    protosIdx=protosIdx(randperm(maxComp,mComp),:);
    protosL=tree.idx(protosIdx(:,1));
    protosR=tree.idx(protosIdx(:,2));

    %Test Evaluation
    % #Lines 49-70 implement Lines 15-24 of Algorithm 3#
    I=-Inf;
    for pp=1:length(protosL)
        %Creation of the putative child nodes
        % #Lines 52-61 implement Lines 17-18 of Algorithm 3#
        LL=protosL(pp); RR=protosR(pp);
        if train(RR,LL)<=train(RR,RR) || train(LL,RR)<=train(LL,LL)
            continue
        end
        trleft=find(train(tree.idx,LL)<=train(tree.idx,RR));
        trright=find(train(tree.idx,LL)>train(tree.idx,RR));
        if isempty(trleft) || isempty(trright)
            continue
        end
        %Function that evaluates the created children according to param.criterion
        I_tmp=EvaluateTest(train,tree,trleft,trright,param); % #Implements Lines 19-21 of Algorithm 3#
        % #Lines 65-69 Implement Lines 22-24 of Algorithm 3#
        if I_tmp>=I %The impurity of the test under evaluation is better than the current best one.
            I=I_tmp;
            tr_left=trleft; tr_right=trright;
            protoR_best=RR; protoL_best=LL;
        end
    end
end
end