clear all
clc

datasets = {'blog', 'yelp'};
methods={'mpsketch', 'mpsketchbaseline'};

turns = 5;
iterations=5;

for i_data =1:length(datasets)
    data = datasets{i_data};
    
    for i_method = 1:length(methods)
        method = methods{i_method};
        if strcmp(method, 'mpsketchbaseline') && strcmp(data, 'yelp')
            continue
        end
        
        for iteration = 1:iterations
            
            for dense = 0.5:0.1:0.9
                for turn = 1:turns
                    
                    fingerprints = load([data, '/', data, '.dense.', num2str(dense), '.', method, '.iteration.', num2str(iteration),  '.embedding.turn.', num2str(turn)]);
                    runtime = load([data, '/', data, '.dense.', num2str(dense), '.', method, '.iteration.', num2str(iteration), '.time.turn.', num2str(turn)]);
                    
                    save([data, '/', data, '.', num2str(dense), '.', method, '.fingerprints.iteration.', num2str(iteration), '.turn.', num2str(turn), '.mat'], 'fingerprints', 'runtime', '-v7.3')
                    
                    delete([data, '/', data, '.dense.', num2str(dense), '.', method, '.iteration.', num2str(iteration),  '.embedding.turn.', num2str(turn)]);
                    delete([data, '/', data, '.dense.', num2str(dense), '.', method, '.iteration.', num2str(iteration), '.time.turn.', num2str(turn)]);
                    
                end
            end
        end
    end
end
