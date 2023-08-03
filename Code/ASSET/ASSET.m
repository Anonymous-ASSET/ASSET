function [ PD,PF,Precision,F1,AUC,MCC] = ASSET(Source0, trainTarget, test, isDiscrete, sigma)
%ASSET Summary of this function goes here
%   Detailed explanation goes here:
% INPUTS:
%   (1) Source0     - A n*(d+1) source dataset or a cell array consisted of multiple source datasets where each dataset is a n_i*(d+1) matrix, the last column is the label {0-nondefective,1-defective};
%   (2) trainTarget - A n_tt*(d+1) matrix, i.e., the training target datset; test data from target project,the last column is the label {0-nondefective,1-defective}; 
%   (3) test        - A n_ut*(d+1) matrix, i.e., the testing dataset;
%   (4) isDistance - {0,1} - 1 denotes that each metric is  discrete, otherwise continuous.
% OUTPUTS:
%   some performance measure, e.g., AUC, MCC；
%

warning('off');

% Default value
if ~exist('isDiscrete','var')||isempty(isDiscrete)
    isDiscrete = 0; 
end
if ~exist('sigma','var')||isempty(sigma)
    sigma = 1; 
end


Source = [];
if iscell(Source0) % if it is a 'cell' type variable
    for i=1:numel(Source0)
        temp = Source0{i};
        temp = SMOTE_02(temp,1);
        Source = [Source;temp]; 
    end
else % if it is not a 'cell' type variable
    Source = Source0;
    Source = SMOTE_02(Source,1); % 
end

defRatioSrc = sum(Source(:,end)==1)/size(Source, 1); % Calculate the defective ratio of source dataset
defRatioTT = sum(trainTarget(:,end)==1)/size(trainTarget, 1); % Calculate the defective ratio of source dataset
if defRatioSrc > 0.6 || size(Source, 1) < 40 || ((defRatioTT>0.6)&&(size(trainTarget,1)>10)) 
    %% Transform mat to arff
    trainData = trainTarget;
    label = cell(size(trainData,1),1);
    temp = trainData(:,end);
    for j=1:size(trainData,1)
        if (temp(j)==1)
            label{j} = 'true';
        else
            label{j} = 'false';
        end
    end
    featureNames = cell(size(trainData,2),1);
    for j=1:(size(trainData,2)-1)
        featureNames{j} = ['X', num2str(j)];
    end
    wekaOBJtrain = matlab2weka('train', featureNames, [num2cell(trainData(:,1:end-1)),label], size(trainData,2));
    
    label = cell(size(test,1),1);
    temp = test(:,end);
    for j=1:size(test,1)
        if (temp(j)==1)
            label{j} = 'true';
        else
            label{j} = 'false';
        end
    end
    wekaOBJtest = matlab2weka('train', featureNames, [num2cell(test(:,1:end-1)),label], size(test,2));
    
    
    %% Random Forest
    RF = trainWekaClassifier(wekaOBJtrain,'trees.RandomForest'); % RF:Train the classifier
    % RF:Test the classifier
    [predLabelRF, classProbs]= wekaClassify(wekaOBJtest,RF);
    pro_pos = classProbs(:,2);
    try
        [ PD,PF,Precision,F1,AUC,MCC] = Performance( test(:,end),classProbs(:,2));
    catch
        PD=nan;PF=nan;Precision=nan; F1=nan;AUC=nan;MCC=nan;
    end
    
    return
end

numInsTest = size(test,1);
numInsTrainTar = size(trainTarget,1);
target = [trainTarget; test];

if size(Source,2)~=size(target,2)
    error('Datastes must have same number of metrics!')
end

% 0 -> 1e-4 for the following logorithm
epsilon = 1e-4;
temp = Source(:,1:end-1);
temp(find(temp==0))=epsilon;
Source(:,1:end-1) = temp;

temp = target(:,1:end-1);
temp(find(temp==0))=epsilon;
target(:,1:end-1) = temp;


% Take logorithm (do this only all values are positive) - either log(x) or log10(x), here we use natural logrithm.
Source(:,1:end-1)=log(Source(:,1:end-1)); 
target(:,1:end-1)=log(target(:,1:end-1));


% Maximal Information Coefficient (MIC) (1) the range of values is [0,1]; (2) the bigger the better; 
% (3) Reshef, David N., et al. Detecting novel associations in large data sets. science 334.6062 (2011): 1518-1524. 
myMIC = [];
for i=1:(size(Source,2)-1) % Each metric
    minestats = mine(Source(:,i)',Source(:,end)'); % input must be row vector
    myMIC(i) = minestats.mic; %[0,1]
end

% % Normalization: 
myMIC = myMIC / sum(myMIC); 


% maximun and minimum of each feature in target
targetCopy = target;
trainTarget = targetCopy(1:numInsTrainTar,:);
target = targetCopy(numInsTrainTar+1:end,:);
Max = max(target(:,1:end-1),[],1);% the maximum value of each feature
Min = min(target(:,1:end-1),[],1);

% maximun and minimum of each feature in traingTarget
MaxPos = max(trainTarget(trainTarget(:,end)==1,1:end-1),[],1);
MinPos = min(trainTarget(trainTarget(:,end)==1,1:end-1),[],1);
MaxNeg = max(trainTarget(trainTarget(:,end)==0,1:end-1),[],1);
MinNeg = min(trainTarget(trainTarget(:,end)==0,1:end-1),[],1);



% Calculate similarity and weight of each training instance based on target
s = zeros(size(Source,1),1);% s - the similarity of each training instance
w = zeros(size(Source,1),1);% w - the weight of each training instance
for i=1:size(Source,1) % each source instance   
    % Weighting
    tem=0;
    for j=1:size(Max,2) % each feature
        if Source(i,j)>=Min(1,j)&&Source(i,j)<=Max(1,j)
            tem=tem+1*myMIC(j);
        end
    end 
    
    tem0=0;
    if Source(i,end)==1
        for j=1:size(MaxPos,2) % each feature
            if Source(i,j)>=MinPos(1,j)&&Source(i,j)<=MaxPos(1,j)
                tem0=tem0+1*myMIC(j);
            end
        end
    else
        for j=1:size(MaxNeg,2) % each feature
            if Source(i,j)>=MinNeg(1,j)&&Source(i,j)<=MaxNeg(1,j)
                tem0=tem0+1*myMIC(j);
            end
        end
    end
    % s(i,1)=tem;
%     s(i,1)=tem+tem0; 
    s(i,1)=numInsTest/(numInsTrainTar+numInsTest)*tem+numInsTrainTar/(numInsTrainTar+numInsTest)*tem0; %
    w(i,1)=s(i,1)/(sum(myMIC)-s(i,1)+1)^2;%
end

% Calculate the prior probability of each classes (i.e., positive class and negative class)
label = Source(:,end);                     % the label of source
num_pos = length(find(label==1));          % the number of  positive instances
num_neg = length(find(label==min(label))); % the number of  negative instances
pri_prob_pos = (sum(w(find(label==1))) + 1) / (sum(w) + numel(unique(label)));
pri_prob_neg = (sum(w(find(label==min(label)))) + 1) / (sum(w) + numel(unique(label)));

pred_label = zeros(size(target,1),1);
pro_pos    = zeros(size(target,1),1);

for i=1:size(target,1) % Each instance in Target   
    met=target(i,1:end-1);% i-th instance in Target
    walls=[];
    idx1=[];
    idx2=[];
    n=0;
    pos_cond_met = [];
    neg_cond_met = [];
    for j=1:length(met) % Each feature
        if isDiscrete % 
            n=numel(unique(Source(:,j))); % the total number of unique values of j-th metric in Source
            idx1 = find((Source(:,j)==met(j))&(label==1));          % the total number of positive instances in source whose j-th metric value equals to met(j)
            idx2 = find((Source(:,j)==met(j))&(label==min(label))); % the total number of negative instances in source whose j-th metric value equals to met(j)
        else % 
            walls = fayyadIrani(Source(:,j),label); % Call fayyadIrani() to generate walls.
            walls = sort([min(Source(:,j)) + min(Source(:,j))/2,walls,max(Source(:,j))]); %  
            n = length(walls)-1; % the number of intervals    
            
            % [infimum, supremum] is the most nearest wall interval which includes met(j) in ideal condition.
            supremum = walls(min(find(roundn(walls,-6)>=roundn(met(j),-6))));% find supremum of j-th metric in walls.
            infimum = walls(max(find(roundn(walls,-6)<=roundn(met(j),-6)))); % find infimum of j-th metric in walls. If infimum is empty,... 
            
             % To void supremum or infimum is empty when met(j) is larger than max(walls) or met(j) is smaller than min(walls).
            if isempty(supremum)
                supremum = max(walls);
            end
            if isempty(infimum)
                infimum = min(walls);
            end
            
            idx1 = find((Source(:,j)>infimum)&(Source(:,j)<=supremum)&(label==1)); % the total number of positive instances which belong to the interval (infinum,supremum].
            idx2 = find((Source(:,j)>infimum)&(Source(:,j)<=supremum)&(label==min(label)));         
        end
        
        % Calculate the class conditional probability
        pos_cond_met(j) = (sum(w(idx1)) + 1) / (sum(w(find(label==1))) + n);          % Calculate the class-conditionnal peobability for j-th metric   
        neg_cond_met(j) = (sum(w(idx2)) + 1) / (sum(w(find(label==min(label)))) + n); % Calculate the class-conditionnal peobability for j-th metric 
    end
      
    
    % Calculate posterior probability
    deno = pri_prob_pos * prod(power(pos_cond_met,exp(-myMIC/sigma^2))) + pri_prob_neg * prod(power(neg_cond_met,exp(-myMIC/sigma^2))); % prod([1,2,3])=>6
    pro_pos(i)=pri_prob_pos * prod(power(pos_cond_met,exp(-myMIC/sigma^2))) / deno;
    pro_neg(i)=pri_prob_neg * prod(power(neg_cond_met,exp(-myMIC/sigma^2))) / deno;  
    
    % Predicted labels
    pred_label(i) = double(pro_pos(i)>=pro_neg(i));
    
end

try
    [ PD,PF,Precision,F1,AUC,MCC] = Performance(test(:,end), pro_pos); % Call self-defined Performance()
catch
    PD=nan;PF=nan;Precision=nan;F1=nan;AUC=nan;MCC=nan;
end

end

