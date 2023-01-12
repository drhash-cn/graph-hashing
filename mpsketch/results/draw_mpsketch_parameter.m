clear all
clc

ratios = [0.5, 0.6, 0.7, 0.8, 0.9];

times = 10000;

methods='mpsketch';

datasets = {'m10', 'yelp'};
datanames = {'CiteSeer-M10', 'Yelp'};
turns = 1;
iterations = 5;
ks=100:50:300;
iratio = 5;

set(gcf,'position',[0,0,1000,250]);

load([datasets{1}, '/', datasets{1}, '.mpsketch.parameters.results.mat'])
subplot(1,4,1)
micro_ratio = mean_micro_f1(:,:,iratio);
surf(micro_ratio);
zlim([0.,1])
ylabel('Embedding Size', 'Rotation',-45)
set(gca, 'ytick', [1 2 3 4 5], 'yticklabels', {'100', '150', '200', '250', '300'})
y_pos = get(get(gca, 'ylabel'), 'Position');
set(get(gca, 'ylabel'), 'Position', y_pos + [0 -1 0]);
xlabel('Iteration', 'Rotation',30)
set(gca, 'xtick', [1 2 3 4 5], 'xticklabels', {'1', '2', '3', '4', '5'})
zlabel('Micro-F1')
title([datanames{1}, ' (', num2str(ratios(iratio)*100),'%)'])

subplot(1,4,2)
cpu_ratio = mean_cpus(:,:,iratio);
surf(cpu_ratio);
zlim([0,1000])
ylabel('Embedding Size', 'Rotation',-45)
set(gca, 'ytick', [1 2 3 4 5], 'yticklabels', {'100', '150', '200', '250', '300'})
y_pos = get(get(gca, 'ylabel'), 'Position');
set(get(gca, 'ylabel'), 'Position', y_pos + [0 -1 0]);
xlabel('Iteration', 'Rotation',30)
set(gca, 'xtick', [1 2 3 4 5], 'xticklabels', {'1', '2', '3', '4', '5'})
zlabel('Embedding Time (s)')
title(datanames{1})
m10_accs = micro_ratio;
m10_cpus = cpu_ratio;



load([datasets{2}, '/', datasets{2}, '.mpsketch.parameters.results.mat'])
subplot(1,4,3)
auc_ratio = mean_auc(:,:,iratio);
surf(auc_ratio);
zlim([0.,1])
ylabel('Embedding Size', 'Rotation',-45)
set(gca, 'ytick', [1 2 3 4 5], 'yticklabels', {'100', '150', '200', '250', '300'})
y_pos = get(get(gca, 'ylabel'), 'Position');
set(get(gca, 'ylabel'), 'Position', y_pos + [0 -1 0]);
xlabel('Iteration', 'Rotation',30)
set(gca, 'xtick', [1 2 3 4 5], 'xticklabels', {'1', '2', '3', '4', '5'})
zlabel('AUC')
title([datanames{2}, ' (', num2str(ratios(iratio)*100),'%)'])

subplot(1,4,4)
cpu_ratio = mean_cpus(:,:,iratio);
surf(cpu_ratio);
zlim([0,15000])
ylabel('Embedding Size', 'Rotation',-45)
set(gca, 'ytick', [1 2 3 4 5], 'yticklabels', {'100', '150', '200', '250', '300'})
y_pos = get(get(gca, 'ylabel'), 'Position');
set(get(gca, 'ylabel'), 'Position', y_pos + [0 -1 0]);
xlabel('Iteration', 'Rotation',30)
set(gca, 'xtick', [1 2 3 4 5], 'xticklabels', {'1', '2', '3', '4', '5'})
zlabel('Embedding Time (s)')
title([datanames{2}, ' (', num2str(ratios(iratio)*100),'%)'])
yelp_accs = auc_ratio;
yelp_cpus = cpu_ratio;
