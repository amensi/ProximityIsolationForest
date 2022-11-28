%%% EXAMPLE OF HOW TO BUILD AND TEST A Proximity Isolation Forest MODEL %%%
clear all;close all;clc

%Number of iterations of the procedure
nIt=10;

%Inlier and outlier class labels
inl=0; outl=1;

%Adding the path
addpath(genpath('./ProxIF_code/'));

%Loading the dataset
load('zongker_od.mat')

%Finding the number and indexes of outliers and inliers in the dataset
N=size(data,1); NOutl=sum(labels==outl); NInl=N-NOutl;
inlIdx=find(labels==inl); outlIdx=find(labels==outl);

%Setting the number and type of objects within the training partitions
Ntr=ceil(N.*0.5);
NtrO=ceil(Ntr.*0.05); %maximum number of outliers
%If outliers are less than 5% of the training data, we are going to use
%half for training and half for testing
if NtrO>=NOutl
    NtrO=ceil(NOutl./2);
end
NtrI=Ntr-NtrO;

%Setting the forest parameters.
%You can also not set the fields of param, parameters will be set in the
%training function according to a fixed parametrization.
param.max_comp=20;
param.T=500;
param.D='log'; param.crit='O-2PH';
param.S=min(128,Ntr);

auc=zeros(nIt,1);
fprintf("Iteration..")
for it=1:nIt
    fprintf("%d..",it);

    %Partitioning the data
    ordI=inlIdx(randperm(NInl)); ordO=outlIdx(randperm(NOutl));
    trainIdx=[ordI(1:NtrI); ordO(1:NtrO)];
    testIdx=[ordI(NtrI+1:end); ordO(NtrO+1:end)];
    train=data(trainIdx,trainIdx);
    test=data(testIdx,trainIdx);
    labels_test=labels(testIdx);

    %Training and testing procedure
    score=ProxIF_wrap(train,test,param);

    %Alternative to the usage of the ProxIF_wrap function:
%     forest=ProxIF_training(train,param);
%     c_l=zeros(forest.param.S,1);
%     for i=1:forest.param.S
%         c_l(i)=average_unsuccessful_search(i);
%     end
%     score=ProxIF_testing(forest,test,c_l);

    [~,~,~,auc(it)]=perfcurve(labels_test,score,outl);
end

fprintf("\nZongker AUC ProxIF-F Table 6 0.8496 VS found AUC %5.4f", mean(auc));