clear all
clc

method='hashgnnplus';

datasets = {'m10', 'pubmed' };
turns = 5;
iterations = 5;

for i_data =1:length(datasets)
    data = datasets{i_data};
    for iturn =1:turns
        
        for iteration=1:iterations
            
            for k = 100:50:300

                    fingerprints = load(['parameters/',data, '/', data, '.', method, '.iteration.', num2str(iteration), '.k.', num2str(k),  '.embedding.turn.', num2str(iturn)]);
                    runtime = load([ 'parameters/',data, '/',  data, '.', method, '.iteration.', num2str(iteration), '.k.', num2str(k), '.time.turn.', num2str(iturn)]);
                    
                    save(['parameters/',data, '/',  data, '.', method, '.iteration.', num2str(iteration), '.k.', num2str(k), '.fingerprints.turn.', num2str(iturn),'.mat'], 'fingerprints', 'runtime')
                
            end
        end
    end
    
end