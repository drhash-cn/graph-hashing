clear all
clc

datasets = {'m10', 'pubmed'};
method='mpsketch';

turns = 5;
iterations=5;
trainRatios = [0.5 0.6 0.7 0.8 0.9];

ks = [100,150,250,300];

for idata =1:length(datasets)
    
    data = datasets{idata};
    
    
    parameters_micro_f1 = zeros(length(ks)+1, iterations, length(trainRatios), turns);
    parameters_macro_f1 = zeros(length(ks)+1, iterations, length(trainRatios), turns);
    parameters_elapsed = zeros(length(ks)+1, iterations, length(trainRatios), turns);
    parameters_runtimes = zeros(length(ks)+1, iterations, turns);
    parameters_cpus = zeros(length(ks)+1, iterations, length(trainRatios), turns);
    
    
    for turn =1:turns
        for iteration = 1:iterations
            
            for ik = 1:length(ks)
                k = ks(ik);
                
                load([ data, '/', data,'.', method, '.fingerprints.iteration.', num2str(iteration), '.k.', num2str(k),'.turn.', num2str(turn), '.mat'])
                load (['../data/', data, '/', data, '.mat'])
                
                if k==100 || k==150
                    iK = ik;
                else
                    iK = ik+1;
                end
                
                parameters_runtimes(iK, iteration, turn) = runtime;
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
                    parameters_elapsed(iK, iteration, iTrain, turn) = toc;
                    
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
                    
                    parameters_micro_f1(iK, iteration, iTrain, turn) = (2* nansum(tp))/(2*nansum(tp)+nansum(fp)+ nansum(fn));
                    parameters_macro_f1(iK, iteration, iTrain, turn) = (1/length(labelName)) * nansum((2*tp)./(2*tp+fp+fn));
                    parameters_cpus(iK, iteration, iTrain, turn) = parameters_elapsed(iK, iteration, iTrain, turn)+parameters_runtimes(iK, iteration, turn);
                end
            end
        end
    end
    
    load([ data, '/', data,'.', method, '.results.mat'])
    parameters_micro_f1(3,:,:,:) = micro_f1;
    parameters_macro_f1(3,:,:,:) = macro_f1;
    parameters_cpus(3,:,:,:) = cpus;
    parameters_elapsed(3,:,:,:) = elapsed;
    parameters_runtimes(3,:,:) = runtimes;
    
    mean_micro_f1 = mean(parameters_micro_f1, 4);
    std_micro_f1 = std(parameters_micro_f1, 0, 4);
    mean_macro_f1 = mean(parameters_macro_f1, 4);
    std_macro_f1 = std(parameters_macro_f1, 0, 4);
    mean_cpus = mean(parameters_cpus, 4);
    std_cpus = std(parameters_cpus, 0, 4);
    mean_elapsed = mean(parameters_elapsed, 4);
    std_elapsed = std(parameters_elapsed, 0, 4);
    mean_runtimes = mean(parameters_runtimes,3);
    std_runtimes = std(parameters_runtimes, 0, 3);
    
    save([ data, '/', data, '.', method, '.parameters.results.mat'], ...
        'micro_f1', 'macro_f1', 'runtimes', 'cpus', 'elapsed', ...
        'mean_micro_f1', 'mean_macro_f1', 'mean_runtimes','mean_cpus', 'mean_elapsed', ...
        'std_micro_f1', 'std_macro_f1', 'std_runtimes','std_cpus', 'std_elapsed')
    
end
