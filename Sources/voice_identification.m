%% This code is used for voice authentication
clear all; close all; clc;

%% Load trained models
try
    load('ml_models_rf.mat', 'ml_name', 'ml_id'); % import the model to be used
    if ~exist('ml_name', 'var') || ~exist('ml_id', 'var')
        error('Models ml_name and/or ml_id not found in ml_models.mat.');
    end
catch err
    error('Failed to load rf_models.mat. Ensure it exists. Error: %s', err.message);
end

fs = 44100; % sampling rate
b = 8; % bits per sample
d = 4; % recording duration

promptUser = true; % to facilitate multiple varification
while promptUser
    %% User Input
    name = input('Your Name:','s');
    id = input('Your ID:','s');

    %% Record and process name
    myrecord = audiorecorder(fs, b, 1);
    disp('PRESS ANY KEY AND SAY YOUR NAME'); pause;
    disp('Recording Started');
    recordblocking(myrecord, d);
    disp('Recording Ended');
    xi = getaudiodata(myrecord);
    xi = xi(:, 1);

    % checking if the voice was correctly recorded
    sound(xi, fs);
    speechSegment = endpointdetection(xi, fs);
    % Normalize amplitude to (-1,1) range
    end1 = speechSegment / max(abs(speechSegment));
    if isempty(end1)
        error('No speech detected in ID recording. Please try again.');
    end
    % Extract MFCCs
    coeffs = mfcc(end1, fs, 'NumCoeffs', 13);

    % Normalize MFCCs
    coeffs_norm = (coeffs - mean(coeffs,1)) ./ std(coeffs,[],1);

    % Calculate delta and delta-delta
    delta = diff([coeffs_norm(1,:); coeffs_norm],1,1);
    delta = [delta; delta(end,:)]; % pad last row
    deltaDelta = diff([delta(1,:); delta],1,1);
    deltaDelta = [deltaDelta; deltaDelta(end,:)]; % pad

    % Combine features by averaging over frames
    featureVector = [mean(coeffs_norm,1), mean(delta,1), mean(deltaDelta,1)];

    [pred_name, score] = predict(ml_name, featureVector);
    %confidence_name = max(score)*100;
    pred_name = string(pred_name); % convert categorical to string
    %% Record and process ID
    myrecord = audiorecorder(fs, b, 1);
    disp('PRESS ANY KEY AND SAY YOUR ID (last 3 digit)'); pause;
    disp('Recording Started');
    recordblocking(myrecord, d);
    disp('Recording Ended');
    xi = getaudiodata(myrecord);
    xi = xi(:, 1);

    % checking if the voice was correctly recorded
    sound(xi, fs); pause;

    speechSegment = endpointdetection(xi, fs);
    % Normalize amplitude to (-1,1) range
    end1 = speechSegment / max(abs(speechSegment));
    if isempty(end1)
        error('No speech detected in ID recording. Please try again.');
    end
    % Extract MFCCs
    coeffs = mfcc(end1, fs, 'NumCoeffs', 13);

    % Normalize MFCCs
    coeffs_norm = (coeffs - mean(coeffs,1)) ./ std(coeffs,[],1);

    % Calculate delta and delta-delta
    delta = diff([coeffs_norm(1,:); coeffs_norm],1,1);
    delta = [delta; delta(end,:)]; % pad last row
    deltaDelta = diff([delta(1,:); delta],1,1);
    deltaDelta = [deltaDelta; deltaDelta(end,:)]; % pad

    % Combine features by averaging over frames
    featureVector = [mean(coeffs_norm,1), mean(delta,1), mean(deltaDelta,1)];

    [pred_id,score] = predict(ml_id, featureVector);
    %confidence_id = max(score)*100;
    pred_id = string(pred_id); % convert categorical to string

    %% Check result
    flag = 1;
    if ~strcmp(pred_name,id)
        flag = 0;
    elseif ~strcmp(pred_id,id)
        flag = 0;
    end

    if flag == 1
        disp('ACCESS GRANTED');
        msgbox('WELCOME TO OUR BANK', 'SUCCESS');
    else
        disp('ACCESS DENIED');
        msgbox('SORRY! TRY AGAIN', 'ERROR', 'error');
    end

    fprintf('Predicted user from name varification: %s\n',pred_name);
    fprintf('Predicted user from id varification: %s\n',pred_id);

    retry = input('Try again? (y/n): ', 's');
    if strcmpi(retry,'n')
        promptUser = false;
    end
    clc; % clears command window after every verification
end
