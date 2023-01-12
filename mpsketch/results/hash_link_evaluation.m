clear all

ratios = [0.5, 0.6, 0.7, 0.8, 0.9];

times = 10000;

datasets = {'blog', 'yelp'};
methods={'mpsketch', 'mpsketchbaseline'};

turns =5;
iterations=5;

for i_data =1:length(datasets)
    
    data = datasets{i_data};
    load(['../data/',data, '/',data, '.mat']);
    nodeNum = size(network,1);
    network(1:nodeNum + 1:end) = 0;
    
    for i_method = 1:length(methods)
        method = methods{i_method};
        
        if strcmp(method, 'mpsketchbaseline') && strcmp(data, 'yelp')
            continue
        end
        
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
        
        auc = zeros(iterations, length(ratios), turns);
        runtimes = zeros(iterations, length(ratios), turns);
        elapsed= zeros(iterations, length(ratios), turns);
        cpus = zeros(iterations, length(ratios), turns);
        
        for iteration = 1: iterations
            for dense= 1:5
                
                for iturn=1:turns
                    
                    display([data, ', ', method, ', iteration: ', num2str(iteration), ', dense: ', num2str(ratios(dense)), ', turn: ', num2str(iturn)])
                    
                    load(['./',data, '/', data, '.', num2str(ratios(dense)), '.',method, '.fingerprints.iteration.', num2str(iteration), '.turn.', num2str(iturn),'.mat']);
                    load(['../data/', data, '/', data, '_', num2str(ratios(dense)), '.mat'])
                    
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
                    elapsed(iteration, dense, iturn) = cpu1+toc;
                    auc(iteration, dense, iturn) = (greatNum + 0.5*equalNum)/ times;
                    runtimes(iteration, dense, iturn) = runtime;
                    cpus(iteration, dense, iturn) = elapsed(iteration, dense, iturn)+runtimes(iteration, dense, iturn);
                    
                end
            end
        end
 
        mean_auc = mean(auc, 3);
        std_auc = std(auc, 0, 3);
        mean_cpus = mean(cpus, 3);
        std_cpus = std(cpus, 0, 3);
        mean_elapsed = mean(elapsed, 3);
        std_elapsed = std(elapsed, 0, 3);
        mean_runtimes = mean(runtimes, 3);
        std_runtimes = std(runtimes, 0, 3);
        
        save(['./',data, '/', data, '.', method, '.results.mat'], ...
            'auc', 'cpus', 'elapsed', 'runtimes', ...
            'mean_auc', 'mean_cpus', 'mean_elapsed', 'mean_runtimes', ...
            'std_auc', 'std_cpus', 'std_elapsed', 'std_runtimes')
        
    end
end









