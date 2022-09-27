function score=ProxIF_testing(forest,test,c_l)
% Proximity Isolation Forest, v1.0, 2022
% (c) A. Mensi
%
% score=proxIF_testing(forest,test,c_l) returns the anomaly score for
% each object in test
%
% forest is a trained Proximity isolation Forest
%
% train is a matrix of dimensions MxN where M is the number of testing
% objects and N the number of training objects
%
% c_l is a vector containing the normalization factor defined in Liu et
% al., 2012
%
% score is a vector of anomaly scores, one per testing object, which value
% is 0<=score<=1

param=forest.param;
if nargin<3
    c_l=zeros(param.S,1);
    for i=1:param.S
        c_l(i)=average_unsuccessful_search(i);
    end
end
c_n=c_l(param.S);

hlim=ceil(param.max_depth);
tmp_score=zeros(size(test,1),param.T);

for t=1:param.T
    for n=1:size(test,1)
        node=forest.tree{t};
        while ~isempty(node.left) %&& ~isempty(node.right)  one is enough
            if param.thr
                if test(n,node.proto)<=node.thr
                    node=node.left;
                else
                    node=node.right;
                end
            else
                if test(n,node.protoL)<=test(n,node.protoR)
                    node=node.left;
                else
                    node=node.right;
                end
            end
            tmp_score(n,t)=tmp_score(n,t)+1;
            if tmp_score(n,t)==hlim
                break
            end
        end
        tmp_score(n,t)=tmp_score(n,t)+c_l(node.nsamples);
    end
end
score=(2.^(-mean(tmp_score,2)./c_n))';
end
