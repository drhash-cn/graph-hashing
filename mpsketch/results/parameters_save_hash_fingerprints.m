clear all
clc


datasets = {'m10', 'pubmed'};
method='mpsketch';

turns = 5;
iterations=5;
ks = [100,150,250,300];

for i_data =1:length(datasets)
    data = datasets{i_data};
    load(['../data/',data, '/',data, '.mat']);
    
    for iteration=1:iterations
        
        for ik = 1:length(ks)
            k = ks(ik);
            for turn =1:turns
                
                fingerprints = load([data, '/', data, '.', method, '.iteration.', num2str(iteration), '.k.', num2str(k), '.embedding.turn.', num2str(turn)]);
                runtime = load([data, '/time.', method, '.iteration.', num2str(iteration), '.k.', num2str(k), '.txt.turn.', num2str(turn)]);
                
                save([data, '/', data, '.', method, '.fingerprints.iteration.', num2str(iteration), '.k.', num2str(k), '.turn.', num2str(turn), '.mat'], 'fingerprints', 'runtime', '-v7.3')
            end
        end
    end
end
