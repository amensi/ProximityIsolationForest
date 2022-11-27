function score=ProxIF_testing(forest,test,c_l)
% Proximity Isolation Forest, v2.0, 2022
% (c) A. Mensi
%
% score=proxIF_testing(forest,test,c_l) returns the anomaly score for
% each object in test
%
% forest is a trained Proximity isolation Forest
%
% test is a matrix of dimensions MxN where M is the number of testing
% objects and N the number of training objects
%
% c_l is a vector containing the normalization factor defined in Liu et
% al., 2012
%
% score is a vector of anomaly scores, one per testing object, which value
% is 0<=score<=1
% #Comments delimeted by '#' make references to the pseudocode presented in
% the supplementary material of the related paper#

param=forest.param;
if nargin<3
    c_l=zeros(param.S,1);
    for i=1:param.S
        c_l(i)=average_unsuccessful_search(i);
    end
end
c_n=c_l(param.S);

if param.thr %Learning criterion is 1P. 
    tmp_score=OnePTesting(forest,test,c_l); %Lines 5-15 of Algorithm 5
else %Learning criterion is 2P.
    tmp_score=TwoPTesting(forest,test,c_l); %Lines 16-26 of Algorithm 5
end

score=(2.^(-mean(tmp_score,2)./c_n))'; %Lines 27-28 of Algorithm 5
end