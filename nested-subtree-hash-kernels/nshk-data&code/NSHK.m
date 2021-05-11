% ======================================================================= %
%                   *** Nested Subtree Hash Kernels ***                   %
%                   Author: Bin Li (bin.li-1@uts.edu.au)                  %
%           QCIS Centre, University of Technology, Sydney (UTS)           % 
% ----------------------------------------------------------------------- %                                 
% Citation: B. Li, X. Zhu, & C. Zhang, "Nested Subtree Hash Kernels for   %
%           Large-scale Graph Classification over Streams", ICDM 2012.    %
% ----------------------------------------------------------------------- %
% Acknowledgement: This code is modified based on N. Shervashidze's code  %
% for "Weisfeiler-Lehman Graph Kernels" in JMLR 12:2539-2561, available   % 
% at http://mlcb.is.tuebingen.mpg.de/Mitarbeiter/Nino/WL/                 %
% ======================================================================= %

function [K,runtime,hashdims] = NSHK(Graphs,hashdims)
% Input: 
%   Graphs - a 1*N structure array of graphs
%   Graphs(i).am - an adjacency matrix of the i'th graph
%   Graphs(i).al - an adjacency list of the i'th graph
%   Graphs(i).nl.values - a column vector of node labels for the i'th graph
% Output: 
%   K - a 1*r cell array of N*N kernel matrices for iter = 1,...,r
%   runtime - total runtime in seconds
%   hashdims - a 1*r vector of the dimensionalities of r hashed spaces

N = size(Graphs,2); % number of graphs
r = size(hashdims,2); % number of iterations

Lists = cell(1,N);
for i = 1:N
  Lists{i} = Graphs(i).al;
end

x = sparse(hashdims(1),N);
t = cputime;
for i = 1:N
    labels{i} = zeros(size(Graphs(i).nl.values,1),1,'uint32');
    for j = 1:length(Graphs(i).nl.values)
      str_label = str2hash(num2str(Graphs(i).nl.values(j)),hashdims(1));
      str_sign = mod(str_label,2)*2-1;
      labels{i}(j) = str_label;
      x(str_label,i) = x(str_label)+str_sign;
    end
end
runtime = cputime-t;
K = full(x'*x);
new_labels = labels;
disp(['Number of hashed dimensions: ',num2str(hashdims(1))]);
clear Graphs;

for iter = 2:r
    x = sparse(hashdims(iter),N);
    t = cputime;
    for i = 1:N
        for j = 1:length(Lists{i})
            long_label = [labels{i}(j),sort(labels{i}(Lists{i}{j}))'];
            long_label_2bytes = typecast(long_label,'uint16');
            long_label_string = char(long_label_2bytes);
            str_label = str2hash(long_label_string,hashdims(iter));
            str_sign = mod(str_label,2)*2-1;
            new_labels{i}(j) = str_label;
            x(str_label,i) = x(str_label)+str_sign;
        end
    end
    runtime = runtime+cputime-t;
    K = K+full(x'*x);
    labels = new_labels;
    disp(['Number of hashed dimensions: ',num2str(hashdims(iter))]);  
end