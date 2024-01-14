% Load the data for training and testing
load('data.mat');

% Define the number of features and time steps
numFeatures = size(XTrain, 2); % number of features
numTimeSteps = size(XTrain, 1); % number of time steps
numOutputs = 60; % number of output nodes


% Define the RNN architecture
numHiddenUnits = 100;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    fullyConnectedLayer(numOutputs)
    regressionLayer
    ];


% Set the training options
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 32, ...
    'SequenceLength', 'longest', ...
    'Shuffle', 'never', ...
    'InitialLearnRate', 0.01, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 50, ...
    'LearnRateDropFactor', 0.1, ...
    'GradientThreshold', 1, ...
    'Verbose', 1, ...
    'Plots', 'training-progress');

% Train the RNN model
net = trainNetwork(XTrain', YTrain', layers, options);

% Test the RNN model
YPred = predict(net, XTest');

% Calculate the root-mean-squared-error (RMSE)
rmse = sqrt(mean((YPred - YTest').^2));

% Display the RMSE
fprintf('RMSE: %f\n', rmse);
