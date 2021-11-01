
%_________________________________________________________________________%
%  WOANet: Whale Optimized Deep Neural Network for the Classification
% of COVID-19 from RadiographyImages             %
% It is the main code                                                                        %
%  Developed in MATLAB R2020b                                             %
%                                                                         %
%  Author and programmer: TRIPTI GOEL                                     %
%                                                                         %
%         e-Mail: triptigoel83@gmail.com
%                 triptigoel@ece.nits.ac.in                               %                                          %
%                               
%                                                                         %
%   Main paper: R Murugan, Tripti Goel, Seyedali Mirjalili, 
%               Deba Kumar Chakrabartty, WOANet: Whale Optimized Deep
%               Neural Network for the Classification of COVID-19 from 
%               Radiography Images, Biocybernetics and Biomedical
%               Engineering, 2021,   %
%                                                                         %
%_________________________________________________________________________%

% load CT Dataset of COVID and Non-COVID Patients

imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
tbl = countEachLabel(imds);

minSetCount = min(tbl{:,2});


no_person = 2;

% Split data into training and testing

[trainingSet, testSet] = splitEachLabel(imds, 0.9, 'randomize');

% Define Network
net = resnet50();
inputSize = net.Layers(1).InputSize;
    
 
%Replace last three layers

if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end

layersToRemove = {
    'fc1000'
    'fc1000_softmax'
    'ClassificationLayer_fc1000'
    };

lgraph = removeLayers(lgraph, layersToRemove); % Remove the the last 3 layers. 

% Specify the number of classes the network should classify.
numClassesPlusBackground = 2;

% Define new classfication layers
newLayers = [
    fullyConnectedLayer(numClassesPlusBackground, 'Name', 'rcnnFC')
    softmaxLayer('Name', 'rcnnSoftmax')
    classificationLayer('Name', 'rcnnClassification')
    ];

% Add new layers
lgraph = addLayers(lgraph, newLayers);

% Connect the new layers to the network. 
lgraph = connectLayers(lgraph, 'avg_pool', 'rcnnFC');
layers = lgraph.Layers;
connections = lgraph.Connections;

% Data augmentation
augmentedTrainingSet = augmentedImageDatastore(inputSize(1:2),trainingSet,'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(inputSize(1:2),testSet, 'ColorPreprocessing', 'gray2rgb');
YValidation = testSet.Labels;
save augmentedTrainingSet augmentedTrainingSet; save augmentedTestSet augmentedTestSet;
save YValidation YValidation; 
% Define the variables for WOA Optimization

SearchAgents_no = 20; % Number of search agents
Max_iteration = 20; % Maximum number of iterations

lb = [4 5 0.0001 5];
ub = [32 30 0.1 35];
dim = 4;

[Best_score,Best_pos,cg_curve] = WOA(SearchAgents_no,Max_iteration,lb,ub,dim,@error_rate);

[K1, K2, K3, K4] = Best_pos(1,2);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',K1, ...
    'MaxEpochs',K2, ...
    'InitialLearnRate',K3, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augmentedTestSet, ...
    'ValidationFrequency',K4, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(augmentedTrainingSet,lgraph,options);

[YPred,Scores] = classify(net,augmentedTestSet);
accuracy = sum(YPred == YValidation)/numel(YValidation);


 confMat = confusionmat(YValidation, YPred);
 confMat1 = bsxfun(@rdivide,confMat,sum(confMat,2));
 plotConfMat(confMat, {'COVID', 'Normal'});
 EVAL = Evaluate(YValidation_Expected, label_index_actual);

   
     
   



