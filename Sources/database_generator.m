%% This code extracts mfcc and stores them in a database from the recorded voice samples
clc;clear all;close all;
%% COLLECTING NAME DATA
folder='C:\Users\ACER\Desktop\DSP Project\Codes\NAME_DATABASE';
filePattern = fullfile(folder, '*.wav');
theFiles = dir(filePattern);
n=length(theFiles);

%% feature extraction
for i = 1 : n
    baseFileName = theFiles(i).name;
    % extracting user name
    j = 0;
    ch = baseFileName(end-j);
    while ch ~= '_'
        j=j+1;
        ch = baseFileName(end-j);
    end
    username = baseFileName(1:end-1-j);
    file = fullfile(folder, baseFileName);
    [s, fs] = audioread(file);
    speechSegment = endpointdetection(s, fs);
    % Normalize amplitude to (-1,1) range
    end1 = speechSegment / max(abs(speechSegment));
    % Extract MFCCs
    coeffs = mfcc(end1, fs, 'NumCoeffs', 13);

    % Normalize MFCCs
    coeffs_norm = (coeffs - mean(coeffs,1)) ./ std(coeffs,[],1);

    % Calculate delta and delta-delta
    delta = diff([coeffs_norm(1,:); coeffs_norm],1,1);  % first derivative
    delta = [delta; delta(end,:)]; % ensuring same size as coeffs
    deltaDelta = diff([delta(1,:); delta],1,1); % second derivative
    deltaDelta = [deltaDelta; deltaDelta(end,:)]; % ensuring same size as coeffs

    % Combine features by averaging over frames
    featureVector = [mean(coeffs_norm,1), mean(delta,1), mean(deltaDelta,1)];

    name_data(:, i) = [username; num2cell(featureVector')]; % first row contains name
end

%% COLLECTING ID DATA
folder='C:\Users\ACER\Desktop\DSP Project\Codes\ID_DATABASE';
filePattern = fullfile(folder, '*.wav');
theFiles = dir(filePattern);
n=length(theFiles);

%% feature extraction
for i = 1 : n
    baseFileName = theFiles(i).name;
    % extracting user id
    j = 0;
    ch = baseFileName(end-j);
    while ch ~= '_'
        j=j+1;
        ch = baseFileName(end-j);
    end
    userid = baseFileName(1:end-1-j);
    file = fullfile(folder, baseFileName);
    [s, fs] = audioread(file);
    speechSegment = endpointdetection(s, fs);
    % Normalize amplitude to (-1,1) range
    end1 = speechSegment / max(abs(speechSegment));
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

    id_data(:, i) = [userid; num2cell(featureVector')]; % first row contains id
end

%% saving the database
try
    save DATABASE name_data id_data;
    disp("Database saved to DATABASE.mat");
catch err
    error('Failed to save database. Error: %s', err.message);
end