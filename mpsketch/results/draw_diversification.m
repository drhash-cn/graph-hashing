datasets = {'m10', 'blog'};
datanames = {'CiteSeer-M10',  'BlogCatalog'};

methods = {'MPSketch', 'MPSketch-basedline'};
methods1 = {'mpsketch', 'mpsketchbaseline'};


load([datasets{1}, '/', datasets{1}, '.', methods1{1}, '.results.mat'])
mpsketch_micro_f1 = mean_micro_f1(2,:);
load([datasets{1}, '/', datasets{1}, '.', methods1{2}, '.results.mat']);
mpsketchbaseline_micro_f1 = mean_micro_f1(1,:);

load([datasets{2}, '/', datasets{2}, '.', methods1{1}, '.results.mat'])
mpsketch_auc = mean_auc(1,:);
load([datasets{2}, '/', datasets{2}, '.', methods1{2}, '.results.mat'])
mpsketchbaseline_auc = mean_auc(2,:);

iteration = 5;

figure;
set(gcf,'position',[0,0,800,300]);

subplot(1, 2, 1)
p1=plot(1:iteration, mpsketch_micro_f1, '--r+', 'LineWidth',1);
hold on;
p2=plot(1:iteration, mpsketchbaseline_micro_f1, ':^', 'Color',[0.6 0.6 0.6], 'LineWidth',1);
hold off;
xlabel('Training Ratio', 'FontSize', 14)
ylabel('Micro-F1', 'FontSize', 14)
xlim([0 6])
ylim([0.7 1])
set(gca, 'xtick', 1:5, 'xticklabels', { '50%','60%','70%','80%','90%'}, 'FontSize', 14)
legend([p1 p2], {'MPSketch','MPSketch-baseline'}, 'location', 'southeast', 'FontSize', 14)
title(['Node Classification on ', datanames{1}], 'FontSize', 14)

subplot(1, 2, 2)
p1=plot(1:iteration, mpsketch_auc, '--r+', 'LineWidth',1);
hold on;
p2=plot(1:iteration, mpsketchbaseline_auc, ':^', 'Color',[0.6 0.6 0.6], 'LineWidth',1);
hold off;
xlabel('Training Ratio', 'FontSize', 14)
ylabel('AUC', 'FontSize', 14)
xlim([0 6])
ylim([0.7 1])
set(gca, 'xtick', 1:5, 'xticklabels', { '50%','60%','70%','80%','90%'}, 'FontSize', 14)
legend([p1 p2], {'MPSketch','MPSketch-baseline'}, 'location', 'northeast', 'FontSize', 14)
title(['Link Prediction on ', datanames{2}], 'FontSize', 14)
