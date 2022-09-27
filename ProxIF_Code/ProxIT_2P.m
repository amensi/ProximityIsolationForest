function tree=ProxIT_2P(train,tree,param)
% Proximity Isolation Forest, v1.0, 2022
% (c) A. Mensi 
%
% tree=ProxIT_2P(train,tree,param) recursive function that builds a ProxIT using a 2P training strategies.
% 
% train is the original distance matrix containing pairwise distances between all training objects 
%
% tree is a tree structure which building procedure is ongoing
%
% param is a structure containing several fields which refer to the
% parameters of the ProxIF model. For more information write 'help
% 'ProxIF_training'
%
% The function returns a ProxIT tree structure that can be traversed if distances
% to the training objects are known.

mComp=param.max_comp;

N=length(tree.idx); maxComp=(N*N-N)/2;
if param.max_comp>maxComp
    mComp=maxComp;
end

tree.nsamples=N;
if N>1
    [P1,P2]=ndgrid(1:N);
    protosIdx=[P1(:) P2(:)];
    protosIdx(protosIdx(:,1)==protosIdx(:,2),:)=[];
    protosIdx=unique(sort(protosIdx,2),'rows');
    protosIdx=protosIdx(randperm(maxComp,mComp),:); 
    protosL=tree.idx(protosIdx(:,1));
    protosR=tree.idx(protosIdx(:,2));
end
if N==1 || tree.height>=param.max_depth || size(unique(train(tree.idx,tree.idx),'rows'),1)==1
    tree.left=[];
    tree.right=[]; tree.imp=[];
    tree.protoL=[]; tree.protoR=[];
else
    if strcmpi(param.crit,'R-2P')  %Random
        LL=protosL; RR=protosR; ss=[];
        tr_left=find(train(tree.idx,LL)<=train(tree.idx,RR));
        tr_right=find(train(tree.idx,LL)>train(tree.idx,RR));
        tree.protoR=RR; tree.protoL=LL;

    else %optimized
        if strcmpi(param.crit,'O-2PSD')
            ss=Inf;
        else
            ss=-Inf; 
        end
        for pp=1:length(protosL)
            LL=protosL(pp); RR=protosR(pp);
            if train(RR,LL)<=train(RR,RR) || train(LL,RR)<=train(LL,LL)
                continue
            end
            trleft=find(train(tree.idx,LL)<=train(tree.idx,RR)); NL=length(trleft);
            trright=find(train(tree.idx,LL)>train(tree.idx,RR)); NR=length(trright);
            if isempty(trleft) || isempty(trright)
                continue
            end
            switch param.crit
                case 'O-2PH'
                    ss_tmp=HausdorffGain(train(tree.idx,tree.idx),trleft,trright,0);
                case 'O-2PRD'
                    K=floor(sqrt(N));
                    [~,NN]=sort(train(tree.idx,tree.idx),2,'ascend');
                    ss_tmp=renyiGain(trleft,trright,NN,K,param.alpha,0);
                case 'O-2PSD'
                    muL=train(tree.idx(trleft),tree.idx(trleft));
                    muR=train(tree.idx(trright),tree.idx(trright));
                    muL=mean(muL(:)); muR=mean(muR(:));
                    ss_tmp=NL./N.*muL+NR./N.*muR;
                case 'O-2PSP'
                    nodeN=(mean(train(tree.idx,LL))+mean(train(tree.idx,RR)))/2; 
                    nodeL=mean(train(tree.idx(trleft),LL)); nodeR=mean(train(tree.idx(trright),RR));
                    ss_tmp=nodeN-(NL./N.*nodeL+NR./N.*nodeR); 
            end
            if strcmpi(param.crit,'O-2PSD')
                if ss_tmp<=ss
                    ss=ss_tmp;
                    tr_left=trleft; tr_right=trright;
                    tree.protoR=RR; tree.protoL=LL; 
                end
            else
                if ss_tmp>=ss
                    ss=ss_tmp;
                    tr_left=trleft; tr_right=trright;
                    tree.protoR=RR; tree.protoL=LL;
                end
            end

        end
    end

    if isempty(tr_left) || isempty(tr_right) 
        tree.left=[];
        tree.right=[]; tree.imp=[];
        tree.protoL=[]; tree.protoR=[];
    
    else
        tree.imp=ss;
        
        tree.left.height=tree.height+1;
        tree.left.idx=tree.idx(tr_left);
        tree.left=ProxIT_2P(train,tree.left,param);

        tree.right.height=tree.height+1;
        tree.right.idx=tree.idx(tr_right);
        tree.right=ProxIT_2P(train,tree.right,param);
    end
end
end

