function speech = endpointDetection(xi, fs)

% Noise reduction (Wiener)
audioData = wiener2(xi, [5 1]);

% endpoint detection using short-time energy and zero crossing rate
frameLength = round(0.025*fs);
frameShift = round(0.010*fs);

numFrames = floor((length(audioData)-frameLength)/frameShift)+1;
ste = zeros(numFrames,1);
zcr = zeros(numFrames,1);

for k = 1:numFrames
    frame = audioData((k-1)*frameShift+1 : (k-1)*frameShift+frameLength);
    ste(k) = sum(frame.^2);
    zcr(k) = sum(abs(diff(sign(frame)))) / (2*length(frame));
end

% Thresholds (mean-based)
steThresh = 0.1 * max(ste);
zcrThresh = 0.1 * max(zcr);

voicedFrames = find(ste > steThresh & zcr > zcrThresh);
if isempty(voicedFrames)
    speech = [];
    return;
end

% Find continuous voiced segment
startFrame = voicedFrames(1);
endFrame = voicedFrames(end);
startSample = (startFrame-1)*frameShift + 1;
endSample = min(length(audioData), (endFrame-1)*frameShift + frameLength);

speech = audioData(startSample:endSample);

% figure (1)
% %Plotting the full signal and only the speech information in the same axis
% subplot(211); plot(xi); title('Recorded Signal'); 
% %The speech information
% subplot(212); plot(speech); title('After Processing')