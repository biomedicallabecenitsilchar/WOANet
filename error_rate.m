%_________________________________________________________________________%
%  WOANet: Whale Optimized Deep Neural Network forthe Classification
% of COVID-19 from RadiographyImages             %
%                                                                         %
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

function [accuracy] = error_rate(kernel_pars)
   load augmentedTrainingSet; load augmentedTestSet; load YValidation;

   net = resnet50();
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

    options = trainingOptions('sgdm', ...
    'Momentum', abs(kernel_pars(1)),...
    'InitialLearnRate', abs(kernel_pars(2)), ...
    'MaxEpochs',abs(kernel_pars(3)), ...
    'Shuffle','every-epoch', ...
    'MiniBatchSize', 32,...
    'ValidationData',augmentedTestSet, ...
    'ValidationFrequency',abs(kernel_pars(4)), ...
    'Verbose',false, ...
    'Plots','training-progress');

     net = trainNetwork(augmentedTrainingSet,lgraph,options);
    [YPred] = classify(net,augmentedTestSet);
    accuracy = sum(YPred == YValidation)/numel(YValidation);



