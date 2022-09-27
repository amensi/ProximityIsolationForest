%Example of how to build and test a Proximity Isolation Forest model
clear all;close all;clc

load('WoodyPlantsPartitioned.mat')
N=numel(labels_train);
param.max_comp=20;
param.T=500; param.S=min(128,N);
param.D='log'; param.crit='O-2PH';
%You can also comment the lines above, parameters will be set in the training function according to a fixed parametrization.
% Just remove param from the next command. 

forest=ProxIF_training(train,param);
c_l=zeros(forest.param.S,1);
for i=1:forest.param.S
    c_l(i)=average_unsuccessful_search(i);
end
score=ProxIF_testing(forest,test,c_l);

%alternatively...
%score=ProxIF_wrap(train,test,param)

%positive class is 1, i.e. the outlier one, since the anomaly score is greater for outliers
[~,~,~,auc]=perfcurve(labels_test,score,1); 
