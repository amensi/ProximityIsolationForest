function score=ProxIF_wrap(train,test,param)
% Proximity Isolation Forest, v1.0, 2022
% (c) A. Mensi
%
% score = ProxIF_wrap(train,test,param) builds a ProxIF on the train
% distance matrix and test the built model on the test distance matrix.
%
% train is a pairwise distance matrix NxN
%
% test is a pairwise dstance matrix MxN
%
% param is a structure containing several fields which refer to the
% parameters of the ProxIF model. For more information write 'help
% 'ProxIF_training'
%
% score is a vector of anomaly scores, one per testing object, which value
% is 0<=score<=1

if size(test,2)~=size(train,2)
    error("You should have the same number of columns! You are comparing to the training samples");
end
forest=ProxIF_training(train,param);
c_l=zeros(forest.param.S,1);
for i=1:forest.param.S
    c_l(i)=average_unsuccessful_search(i);
end
score=ProxIF_testing(forest,test,c_l);
end