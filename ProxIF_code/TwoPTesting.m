function score=TwoPTesting(forest,test,c_l)
% Proximity Isolation Forest, v1.0, 2022
% (c) A. Mensi
%
% score=OnePTesting(forest,test,c_l) returns the anomaly score at tree level for
% each object in test 
%
% forest is a trained Proximity isolation Forest (2P learning strategy)
%
% test is a matrix of dimensions MxN where M is the number of testing
% objects and N the number of training objects
%
% c_l is a vector containing the normalization factor defined in Liu et
% al., 2012
%
% score is a vector of anomaly scores at tree level, one per tree and per testing object, which value
% is proportional to the depth of the reached leaf


param=forest.param;
score=zeros(size(test,1),param.T);
hlim=ceil(param.max_depth);
for t=1:param.T
    for n=1:size(test,1)
        node=forest.tree{t};
        while ~isempty(node.left) %&& ~isempty(node.right)  one is enough
            if test(n,node.protoL)<=test(n,node.protoR)
                node=node.left;
            else
                node=node.right;
            end
            score(n,t)=score(n,t)+1;
            if score(n,t)==hlim
                break
            end
        end
        score(n,t)=score(n,t)+c_l(node.nsamples); 
    end
end
end