%% This code is used to collect voice samples from user
clear all;close all;clc;

% parameters

Fs = 44100; % Sampling rate
b = 8; % Bits per sample
d = 4; % recording duration
inputFolder_name = 'NAME_DATABASE'; % name of the folder to save name data
inputFolder_id = 'ID_DATABASE'; % name of the folder to save id data

% Check and create folder
if ~exist(inputFolder_name, 'dir')
    mkdir(inputFolder_name);
end
if ~exist(inputFolder_id, 'dir')
    mkdir(inputFolder_id);
end
p = 5; % how many samples are previously recorded for the user

promptUser = true;
while promptUser
    a = input('Your ID:','s');
    basename = [a,'_']; % audio files are saved using id as base name
    n = input('Number of Samples:'); % number of samples; half of the samples are name and half are id

    disp('Press enter to start recording.');
    for k=1:n*2
        pause() % sample is taken after a key if pressed
        if k<=n
            myrecord = audiorecorder(Fs,b,1);
            fprintf('Name sample number %d\n',k);
            disp('State Your Name');
            recordblocking(myrecord,d);
            disp('Recording Ended');
            xi = getaudiodata(myrecord);
            folder_destination=['C:\Users\ACER\Desktop\DSP Project\Codes\',inputFolder_name];
            filename=[basename,num2str(k+p),'.wav'];
            file_dest=fullfile(folder_destination,filename);
            audiowrite(file_dest,xi, Fs);
        else
            myrecord = audiorecorder(Fs,b,1);
            fprintf('ID sample number %d\n',k-n);
            disp('State Your ID (last 3 digit)');
            recordblocking(myrecord,d);
            disp('Recording Ended');
            xi = getaudiodata(myrecord);
            folder_destination=['C:\Users\ACER\Desktop\DSP Project\Codes\',inputFolder_id];
            filename=[basename,num2str(k-n+p),'.wav'];
            file_dest=fullfile(folder_destination,filename);
            audiowrite(file_dest,xi, Fs);
        end
    end
    retry = input('Continue Recording? (y/n): ', 's');
    if strcmpi(retry,'n')
        promptUser = false;
    end
end
disp('Data Collection Completed');