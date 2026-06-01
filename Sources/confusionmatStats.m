function stats = confusionmatStats(group,grouphat)
    % group = true labels
    % grouphat = predicted labels

    if iscell(group)
        group = categorical(group);
    end
    if iscell(grouphat)
        grouphat = categorical(grouphat);
    end

    [confMat,order] = confusionmat(group,grouphat);
    numOfClasses = size(confMat,1);
    totalSamples = sum(confMat(:));
    
    stats.confusionMatrix = confMat;
    stats.classLabels = order;

    for class=1:numOfClasses
        TP = confMat(class,class);
        FP = sum(confMat(:,class)) - TP;
        FN = sum(confMat(class,:)) - TP;
        TN = totalSamples - TP - FP - FN;

        stats.precision(class) = TP / (TP + FP + eps);
        stats.recall(class)    = TP / (TP + FN + eps);
        stats.F1score(class)   = 2 * (stats.precision(class) * stats.recall(class)) / (stats.precision(class) + stats.recall(class) + eps);
        stats.accuracy(class)  = (TP + TN) / (TP + TN + FP + FN);
    end

    stats.macroAvgPrecision = mean(stats.precision);
    stats.macroAvgRecall = mean(stats.recall);
    stats.macroAvgF1score = mean(stats.F1score);
