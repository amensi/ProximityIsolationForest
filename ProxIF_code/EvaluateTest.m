function I=EvaluateTest(train,tree,trleft,trright,param)
% Proximity Isolation Forest, v1.0, 2022
% (c) A. Mensi 
%
% I=EvaluateTest(train,tree,trleft,trright,param) returns the value of the optimization
% function chosen according to param.criterion
%
% It evaluates the goodness of the split of the current node intothe putative child nodes identified by the
% submatrices
% train(tree.idx(trleft),tree.idx(trleft)) and
% train(tree.idx(trright),tree.idx(trright))

N=numel(tree.idx);
NL=numel(trleft); NR=numel(trright);
switch param.crit %According to the chosen criterion a different type of function is optimized
    case {'O-1PSD','O-2PSD'}
        muL=train(tree.idx(trleft),tree.idx(trleft));
        muR=train(tree.idx(trright),tree.idx(trright));
        muL=mean(muL(:)); muR=mean(muR(:));
        I=NL./N.*muL+NR./N*muR;
        I=-I; %Needed since it would be a minimization problem.
    case 'O-2PSP'
        nodeN=(mean(train(tree.idx,LL))+mean(train(tree.idx,RR)))/2;
        nodeL=mean(train(tree.idx(trleft),LL)); nodeR=mean(train(tree.idx(trright),RR));
        I=nodeN-(NL./N.*nodeL+NR./N.*nodeR);
    case {'O-1PRD', 'O-2PRD'}
        K=floor(sqrt(N));
        [~,NN]=sort(train(tree.idx,tree.idx),2,'ascend');
        I=renyiGain(trleft,trright,NN,K,param.alpha,0);
    case {'O-1PH', 'O-2PH'}
        I=HausdorffGain(train(tree.idx,tree.idx),trleft,trright,0);
end
end


