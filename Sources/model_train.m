%% This code is used to train machine learning model and save them
clc;clear all; close all;
%% Load database
load('DATABASE.mat');

%% Prepare training data
features_name = cell2mat(name_data(2:end, :))'; % features
labels_names = name_data(1, :);  % labels
features_id = cell2mat(id_data(2:end, :))';
labels_ids = id_data(1, :);

%% shuffling the dataset
% idx = randperm(n);
% % name data
% features_name_shuffled = features_name(idx, :);
% labels_names_shuffled = labels_names(idx);
%
% % id data
% features_id_shuffled = features_id(idx, :);
% labels_ids_shuffled = labels_ids(idx);

%%
features_name_train = features_name;
labels_names_train = categorical(labels_names);
features_id_train = features_id;
labels_ids_train = categorical(labels_ids);

%% Random Forest models
% Define template for bagged trees
t = templateTree('MaxNumSplits', 20, 'MinLeafSize', 5); % Initial settings
hyperopts = struct(...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', 30, ... % Limit iterations for tuning
    'ShowPlots', false, ... % Display optimization progress
    'Verbose', 1,... % Show progress in command window
    'KFold', 5);


% Define parameters to optimize
paramToOptimize = {...
    'NumLearningCycles', 'MaxNumSplits', 'MinLeafSize'};

% Train ensemble with hyperparameter optimization
ml_name = fitcensemble(features_name_train, string(labels_names_train), ...
    'Method', 'Bag', ...
    'Learners', t, ...
    'OptimizeHyperparameters', paramToOptimize, ...
    'HyperparameterOptimizationOptions', hyperopts);

ml_id = fitcensemble(features_id_train, string(labels_ids_train), ...
    'Method', 'Bag', ...
    'Learners', t, ...
    'OptimizeHyperparameters', paramToOptimize, ...
    'HyperparameterOptimizationOptions', hyperopts);
%% KNN Model
% ml_name=fitcknn(features_name_train, labels_names_train, ...
%     'OptimizeHyperparameters', {'NumNeighbors', 'Distance', 'DistanceWeight', 'Standardize'}, ...
%     'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', 'expected-improvement-plus', ...
%                                                'KFold', 5));
% 
% ml_id=fitcknn(features_id_train, labels_ids_train, ...
%     'OptimizeHyperparameters', {'NumNeighbors', 'Distance', 'DistanceWeight', 'Standardize'}, ...
%     'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', 'expected-improvement-plus', ...
%                                                'KFold', 5));
%% SVM Model
% % SVM template with initial settings
% svmTemplate = templateSVM(...
%     'KernelFunction', 'rbf', ...        % RBF kernel
%     'Standardize', true, ...            % Automatic feature scaling
%     'ScoreTransform', 'fitPosterior');              % Enable posterior probabilities
% % hyperparameter optimization options
% hyperopts = struct(...
%     'AcquisitionFunctionName', 'expected-improvement-plus', ... % Bayesian optimization strategy
%     'MaxObjectiveEvaluations', 30, ...
%     'ShowPlots', false, ...             % Display optimization progress (turned off)
%     'Verbose', 1);                     % Show progress in command window
% 
% % parameters to optimize
% paramToOptimize = {'BoxConstraint', 'KernelScale'};
% 
% % Train with hyperparameter optimization
% ml_name = fitcecoc(features_name_train,labels_names_train, ...
%     'Learners', svmTemplate, ...
%     'ClassNames', categories(labels_names_train), ...
%     'OptimizeHyperparameters', paramToOptimize, ...
%     'HyperparameterOptimizationOptions', hyperopts, ...
%     'FitPosterior', true,'Coding', 'onevsone');
% 
% ml_id =  fitcecoc(features_id_train,labels_ids_train, ...
%     'Learners', svmTemplate, ...
%     'ClassNames', categories(labels_ids_train), ...
%     'OptimizeHyperparameters', paramToOptimize, ...
%     'HyperparameterOptimizationOptions', hyperopts, ...
%     'FitPosterior', true,'Coding', 'onevsone');

%% Evaluate model accuracy using cross-validation

%Perform 5-fold cross-validation
cv_name = crossval(ml_name, 'KFold', 5);
cv_id = crossval(ml_id, 'KFold', 5);

% Compute cross-validation loss
loss_name = kfoldLoss(cv_name);
loss_id = kfoldLoss(cv_id);

% Convert loss to accuracy
accuracy_name = (1 - loss_name) * 100;
accuracy_id = (1 - loss_id) * 100;

fprintf('Cross-Validation Accuracy for Name Model: %.2f%%\n', accuracy_name);
fprintf('Cross-Validation Accuracy for ID Model: %.2f%%\n', accuracy_id);

% Save the trained models
try
    save('ml_models.mat', 'ml_name', 'ml_id'); % change the file name according to the model used
    disp('Model saved to ml_models.mat');
catch err
    error('Failed to save models to rf_models.mat. Error: %s', err.message);
end