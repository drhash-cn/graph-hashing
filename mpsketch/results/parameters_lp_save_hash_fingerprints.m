clear all
clc

datasets = {'blog', 'yelp'};
method='mpsketch';

turns = 5;
iterations=5;

ks = [100,150,250,300];

for i_data =1:length(datasets)
    data = datasets{i_data};
    
    for iteration = 1:iterations
        
        for dense = 0.5:0.1:0.9
            
            for ik = 1:length(ks)
                k = ks(ik);
                for turn = 1:turns
                    display([data, ', ', method, ', iteration: ', num2str(iteration), ', dense: ', num2str(dense), ', k: ', num2str(k), ', turn: ', num2str(turn)])
                    
                    fingerprints = load([data, '/', data, '.dense.', num2str(dense), '.', method, '.iteration.', num2str(iteration), '.k.', num2str(k),  '.embedding.turn.', num2str(turn)]);
                    runtime = load([data, '/time.dense.', num2str(dense), '.', method, '.iteration.', num2str(iteration), '.k.', num2str(k), '.txt.turn.', num2str(turn)]);
                    save([data, '/', data, '.', num2str(dense), '.', method, '.fingerprints.iteration.', num2str(iteration), '.k.', num2str(k), '.turn.', num2str(turn), '.mat'], 'fingerprints', 'runtime', '-v7.3')
                    
                end
                
            end
        end
    end
end

