clear all

ratios = [0.5, 0.6, 0.7, 0.8, 0.9];

times = 10000;

datasets = {'blog', 'yelp'};
method='mpsketch';

turns =5;
iterations=5;
ks = [100,150,250,300];

for i_data =1:length(datasets)
    
    data = datasets{i_data};
    load(['../data/',data, '/',data, '.mat']);
    nodeNum = size(network,1);
    network(1:nodeNum + 1:end) = 0;
    
    
    count = 1;
    nonexistence = zeros(2, times);
    tic;
    while count <= times
        edgeIds = randi(nodeNum, 2, 1);
        if network(edgeIds(1), edgeIds(2)) == 0
            nonexistence(:, count) = edgeIds;
            count = count+1;
        end
    end
    cpu1=toc;
    
    parameters_auc = zeros(length(ks)+1, iterations, length(ratios), turns);
    parameters_runtimes = zeros(length(ks)+1, iterations, length(ratios), turns);
    parameters_elapsed= zeros(length(ks)+1, iterations, length(ratios), turns);
    parameters_cpus = zeros(length(ks)+1, iterations, length(ratios), turns);
    
    for iteration = 1: iterations
        for dense= 1:5
            for ik = 1:length(ks)
                k = ks(ik);
                for iturn=1:turns
                    
                    display([data, ', ', method, ', iteration: ', num2str(iteration), ', dense: ', num2str(ratios(dense)), ', turn: ', num2str(iturn), '.k.', num2str(k)])
                    
                    load(['./',data, '/', data, '.', num2str(ratios(dense)), '.',method, '.fingerprints.iteration.', num2str(iteration), '.k.', num2str(k), '.turn.', num2str(iturn),'.mat']);
                    load(['../data/', data, '/', data, '_', num2str(ratios(dense)), '.mat'])
                    
                    if k==100 || k==150
                        iK = ik;
                    else
                        iK = ik+1;
                    end
                    tic;
                    % train network
                    trainGraph(1:nodeNum + 1:end) = 0;
                    % hamming kernel is for minhash and its variation
                    nonexistence_similarity = sum(fingerprints(nonexistence(1,:),:)==fingerprints(nonexistence(2,:),:), 2)/size(fingerprints, 2);
                    
                    
                    % test network, missing links
                    testGraph(1:nodeNum + 1:end) = 0;
                    [iTest, jTest] = find(testGraph==1);
                    testedEdges = [iTest, jTest];
                    clear iTest
                    clear jTest
                    testedEdges = testedEdges(testedEdges(:,1)>testedEdges(:,2),:);
                    testedEdges = testedEdges(randi(size(testedEdges, 1), 1, times), :);
                    
                    missing_similarity = sum(fingerprints(testedEdges(:,1),:)==fingerprints(testedEdges(:,2),:), 2)/size(fingerprints, 2);
                    
                    
                    % AUC
                    greatNum = sum(missing_similarity > nonexistence_similarity);
                    equalNum = sum(missing_similarity == nonexistence_similarity);
                    
                    
                    parameters_elapsed(iK, iteration, dense, iturn) = cpu1+toc;
                    parameters_auc(iK, iteration, dense, iturn) = (greatNum + 0.5*equalNum)/ times;
                    parameters_runtimes(iK, iteration, dense, iturn) = runtime;
                    parameters_cpus(iK, iteration, dense, iturn) = parameters_elapsed(iK, iteration, dense, iturn)+parameters_runtimes(iK, iteration, dense, iturn);
                    
                end
            end
        end
    end
    
    load([ data, '/', data,'.', method, '.results.mat'])
    parameters_auc(3,:,:,:) = auc;
    parameters_cpus(3,:,:,:) = cpus;
    parameters_elapsed(3,:,:,:) = elapsed;
    parameters_runtimes(3,:,:,:) = runtimes;
    
    mean_auc = mean(parameters_auc, 4);
    std_auc = std(parameters_auc, 0, 4);
    mean_cpus = mean(parameters_cpus, 4);
    std_cpus = std(parameters_cpus, 0, 4);
    mean_elapsed = mean(parameters_elapsed, 4);
    std_elapsed = std(parameters_elapsed, 0, 4);
    mean_runtimes = mean(parameters_runtimes, 4);
    std_runtimes = std(parameters_runtimes, 0, 4);
    
    save(['./',data, '/', data, '.', method, '.parameters.results.mat'], ...
        'auc', 'cpus', 'elapsed', 'runtimes', ...
        'mean_auc', 'mean_cpus', 'mean_elapsed', 'mean_runtimes', ...
        'std_auc', 'std_cpus', 'std_elapsed', 'std_runtimes')
    
end

