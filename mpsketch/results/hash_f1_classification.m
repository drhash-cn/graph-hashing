clear all
clc

datasets = {'m10', 'pubmed'};
methods={'mpsketch', 'mpsketchbaseline'};

turns = 5;
iterations=5;
trainRatios = [0.5 0.6 0.7 0.8 0.9];

for idata =1:length(datasets)
    
    data = datasets{idata};
    
    for i_method = 1:length(methods)
        method = methods{i_method};
        if strcmp(method, 'mpsketchbaseline') && strcmp(data, 'pubmed')
            continue
        end
        
        acc = zeros(iterations, length(trainRatios), turns);
        micro_f1 = zeros(iterations, length(trainRatios), turns);
        macro_f1 = zeros(iterations, length(trainRatios), turns);
        elapsed = zeros(iterations, length(trainRatios), turns);
        runtimes = zeros(iterations, turns);
        cpus = zeros(iterations, length(trainRatios), turns);
        
        
        for turn =1:turns
            for iteration = 1:iterations
    
                load([ data, '/', data,'.', method, '.fingerprints.iteration.', num2str(iteration),'.turn.', num2str(turn), '.mat'])
                load (['../data/', data, '/', data, '.mat'])
                runtimes(iteration, turn) = runtime;
                instanceNum = size(labels, 1);
                
                labelNum = length(unique(labels));
                labelName = unique(labels);
                
                for iTrain = 1: length(trainRatios)
                    
                    trainSamples = sort(randperm(instanceNum, round(instanceNum*trainRatios(iTrain))));
                    testSamples = setdiff(1:instanceNum, trainSamples);
                    
                    trainData = fingerprints(trainSamples,:);
                    trainLabels = labels(trainSamples,:);
                    
                    testData = fingerprints(testSamples,:);
                    testLabels = labels(testSamples,:);
                    
                    tic;
                    trainKernel = 1-squareform(pdist(trainData, 'hamming'));
                    model = svmtrain(trainLabels, [(1:length(trainSamples))', trainKernel], '-t 4 -q');
                    testKernel = 1-pdist2(testData, trainData, 'hamming');
                    [predictedLabels, accuracy, ~]= svmpredict(testLabels, [(1:length(testSamples))', testKernel], model, '-q');
                    elapsed(iteration, iTrain, turn) = toc;
                    acc(iteration, iTrain, turn) = accuracy(1);
                    
                    tp = zeros(1, length(labelName));
                    fp = zeros(1, length(labelName));
                    fn = zeros(1, length(labelName));
                    for iLabel=1:length(labelName)
                        testPositive = find(testLabels == labelName(iLabel));
                        predictedPositive = find(predictedLabels == labelName(iLabel));
                        tp(iLabel) = length(intersect(predictedPositive, testPositive));
                        
                        testNegative = find(testLabels ~= labelName(iLabel));
                        fp(iLabel) = length(intersect(predictedPositive, testNegative));
                        
                        predictedNegative = find(predictedLabels ~= labelName(iLabel));
                        fn(iLabel) = length(intersect(predictedNegative, testPositive));

                    end
                 
                    micro_f1(iteration, iTrain, turn) = (2* nansum(tp))/(2*nansum(tp)+nansum(fp)+ nansum(fn));
                    macro_f1(iteration, iTrain, turn) = (1/length(labelName)) * nansum((2*tp)./(2*tp+fp+fn));
                    cpus(iteration, iTrain, turn) = elapsed(iteration, iTrain, turn)+runtimes(iteration, turn);
                    
                end
            end
        end
        mean_micro_f1 = mean(micro_f1, 3);
        std_micro_f1 = std(micro_f1, 0, 3);
        mean_macro_f1 = mean(macro_f1, 3);
        std_macro_f1 = std(macro_f1, 0, 3);
        mean_cpus = mean(cpus, 3);
        std_cpus = std(cpus, 0, 3);
        mean_elapsed = mean(elapsed, 3);
        std_elapsed = std(elapsed, 0, 3);
        mean_runtimes = mean(runtimes,2);
        std_runtimes = std(runtimes, 0, 2);
        
        save([ data, '/', data, '.', method, '.results.mat'], ...
            'micro_f1', 'macro_f1', 'runtimes', 'cpus', 'elapsed', ...
            'mean_micro_f1', 'mean_macro_f1', 'mean_runtimes','mean_cpus', 'mean_elapsed', ...
            'std_micro_f1', 'std_macro_f1', 'std_runtimes','std_cpus', 'std_elapsed')
    end
end
