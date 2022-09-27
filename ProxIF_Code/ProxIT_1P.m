function tree=ProxIT_1P(train,tree,param)
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
    protos=tree.idx(protosIdx(:,1));
    thrs=tree.idx(protosIdx(:,2));
end


if N==1 || tree.height>=param.max_depth || length(unique(train(tree.idx,tree.idx)))==1
    tree.left=[];
    tree.right=[];
    tree.imp=[];
    tree.proto=[]; tree.thr=[];
else    
    tr_left=[]; tr_right=[];
    if strcmpi(param.crit,'R-1P')
        %protos in this case is a vector of length 1 since max_comp=1
        ss=[]; range=sort(train(tree.idx,protos),'ascend'); range(1)=[];
        if length(unique(range))~=1 %True in the following cases: two objects, all at the same distance from P.
            a=range(1); b=range(end);
            while isempty(tr_left)|| isempty(tr_right)
                tree.thr=a+(b-a).*rand(1);
                tr_left=find(train(tree.idx,protos)<=tree.thr);
                tr_right=find(train(tree.idx,protos)>tree.thr);
            end
            tree.proto=protos;
        end
    else
        if strcmpi(param.crit,'O-1PSD')
            ss=Inf;
        else
            ss=-Inf;
        end
        for pp=1:length(protos)
            if  size(unique(train(tree.idx,protos(pp)),'rows'),1)==1
                continue
            end
            thr=train(thrs(pp),protos(pp));
            trleft=find(train(tree.idx,protos(pp))<=thr); NL=size(trleft,1);
            trright=find(train(tree.idx,protos(pp))>thr);NR=size(trright,1);
            if isempty(trleft) || isempty(trright) 
                continue
            end
            switch param.crit
                case 'O-1PSD'
                    muL=train(tree.idx(trleft),tree.idx(trleft));
                    muR=train(tree.idx(trright),tree.idx(trright));
                    muL=mean(muL(:)); muR=mean(muR(:));
                    ss_tmp=NL./N.*muL+NR./N*muR;
                case 'O-1PRD'
                    K=floor(sqrt(N));
                    [~,NN]=sort(train(tree.idx,tree.idx),2,'ascend');
                    ss_tmp=renyiGain(trleft,trright,NN,K,param.alpha,0);
                case 'O-1PH'
                    ss_tmp=HausdorffGain(train(tree.idx,tree.idx),trleft,trright,0);
            end
            if strcmpi(param.crit,'O-1PSD')
                if ss_tmp<ss
                    ss=ss_tmp;
                    tr_left=trleft; tr_right=trright;
                    tree.proto=protos(pp); tree.thr=thr;
                end
            else
                if ss_tmp>ss
                    ss=ss_tmp;
                    tr_left=trleft; tr_right=trright;
                    tree.proto=protos(pp); tree.thr=thr;
                end
            end
        end
    end

    if isempty(tr_left) || isempty(tr_right)
        tree.left=[];
        tree.right=[]; tree.imp=[];
        tree.proto=[]; tree.thr=[];
    else
        tree.imp=ss;

        tree.left.height=tree.height+1;
        tree.left.idx=tree.idx(tr_left);
        tree.left=ProxIT_1P(train,tree.left,param);

        tree.right.height=tree.height+1;
        tree.right.idx=tree.idx(tr_right);
        tree.right=ProxIT_1P(train,tree.right,param);
    end
end
end

