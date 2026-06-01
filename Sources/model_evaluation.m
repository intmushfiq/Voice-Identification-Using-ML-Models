%% This code is used to evaluate the model based on pre-recorded test data
clc; close all; clear all;

%% Load model and dataset
load('ml_models_knn.mat', 'ml_name', 'ml_id');
load('TEST_DATABASE.mat');

%% Prepare test data
features_name_test = cell2mat(name_data(2:end, :))'; % Features for name model
labels_names_test = name_data(1, :)'; % Labels for name model (transposed to column)
features_id_test = cell2mat(id_data(2:end, :))'; % Features for ID model
labels_ids_test = id_data(1, :)'; % Labels for ID model (transposed to column)

%% Evaluation matrices

%% Uncomment them when evaluating knn model and comment them when evaluating svm or rf model
labels_names_test = categorical(labels_names_test);  
labels_ids_test = categorical(labels_ids_test);
%% 

% name model
pred_name = predict(ml_name,features_name_test);
% Plot and save confusion matrix
figure('Name', 'Confusion Matrix - Name Model');
cmChart1 = confusionchart(labels_names_test, pred_name);
title('Confusion Matrix - Name Model');
saveas(cmChart1, 'confusion_matrix_name.png');
% classification report
stats1 = confusionmatStats(labels_names_test, pred_name);
T1 = table(stats1.classLabels(:), ...
          stats1.precision(:), ...
          stats1.recall(:), ...
          stats1.F1score(:), ...
          stats1.accuracy(:), ...
          'VariableNames', {'Class', 'Precision', 'Recall', 'F1score', 'Accuracy'});

% Display the table
disp('--- Classification Report For Name Model---');
disp(T1);
% Display macro averages
fprintf('\nMacro Average Precision: %.2f\n', stats1.macroAvgPrecision);
fprintf('Macro Average Recall:    %.2f\n', stats1.macroAvgRecall);
fprintf('Macro Average F1-score:  %.2f\n', stats1.macroAvgF1score);

fprintf('\n');
% id model
pred_id = predict(ml_id,features_id_test);
confMat_id = confusionmat(labels_ids_test, pred_id);

% classification report
figure('Name', 'Confusion Matrix - Id Model');
cmChart2 = confusionchart(labels_ids_test, pred_id);
title('Confusion Matrix - Id Model');
saveas(cmChart2, 'confusion_matrix_Id.png');

stats2 = confusionmatStats(labels_ids_test, pred_id);
T2 = table(stats2.classLabels(:), ...
          stats2.precision(:), ...
          stats2.recall(:), ...
          stats2.F1score(:), ...
          stats2.accuracy(:), ...
          'VariableNames', {'Class', 'Precision', 'Recall', 'F1score', 'Accuracy'});

% Display the table
disp('--- Classification Report For Id Model---');
disp(T2);
% Display macro averages
fprintf('\nMacro Average Precision: %.2f\n', stats2.macroAvgPrecision);
fprintf('Macro Average Recall:    %.2f\n', stats2.macroAvgRecall);
fprintf('Macro Average F1-score:  %.2f\n', stats2.macroAvgF1score);