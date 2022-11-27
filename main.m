%Example of how to build and test a Proximity Isolation Forest model
clear all;close all;clc

%NOTE: The user hjas to download:
% - prdisdata and distools from http://prtools.tudelft.nl/Guide/37Pages/distools.html 
% - prtools from http://prtools.tudelft.nl/Guide/37Pages/software.html
%Put the extracted folders in the current one (or modify the path below)

%Number of iterations of the procedure
nIt=10;

%Inlier and outlier class labels
inl=0; outl=1;

%Adding the paths
addpath(genpath('./ProxIF/'));
addpath(genpath('./prdisdata/')); addpath(genpath('./distools/')); addpath(genpath('./prtools/'));

%Loading the dataset
dataset_name='zongker';
dataset=load(dataset_name);
data=dataset.d; temp_labels=dataset.lab; N=size(data,1);
%Zongker is a similarity matrix so we need to transform it into a
%dissimilarity one.
if strcmpi(dataset_name,'zongker')
    data=dissimt(data,'SIM2DIS');
end

%Labeling procedure: objects belonging to the class with the highest
%scatter are labeled as outliers
classes=unique(temp_labels);
withinScatter=zeros(length(classes),1);
for cl=1:length(classes)
    temp=data(temp_labels==classes(cl),temp_labels==classes(cl));
    withinScatter(cl)=mean(temp,'all');
end
[~,idx]=max(withinScatter);
labels=zeros(N,1);
labels(temp_labels~=classes(idx))=inl;
labels(temp_labels==classes(idx))=outl;
NOutl=sum(labels==outl); NInl=N-NOutl;
inlIdx=find(labels==inl); outlIdx=find(labels==outl);

%Setting the number and type of objects within the training partitions
Ntr=ceil(N.*0.5); NtrO=ceil(Ntr.*0.05); %maximum number of outliers
%If outliers are less than 5% of the training data, then we are going to use
%half for training and half for testing
if NtrO>=NOutl
    NtrO=ceil(NOutl./2);
end
NtrI=Ntr-NtrO;

%Setting the forest parameters
%You can also comment Lines 57-60: parameters will be set in the
%training function according to a fixed parametrization.
%Just remove param when calling ProxIF_wrap or  ProxIF_training
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

    % The commented lines below are an alternative to Line 27
%     forest=ProxIF_training(train,param);
%     c_l=zeros(forest.param.S,1);
%     for i=1:forest.param.S
%         c_l(i)=average_unsuccessful_search(i);
%     end
%     score=ProxIF_testing(forest,test,c_l);

    [~,~,~,auc(it)]=perfcurve(labels_test,score,outl);
end

fprintf("\nZongker AUC Table 6 0.8496 VS found AUC %5.4f", mean(auc));