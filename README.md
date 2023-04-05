# graph_hashing
A toolbox of randomized hashing algorithms for fast Graph Representation and Network Embedding. We provide two sets of graph hashing algorithms as follows:

- Graph kernels for graph classification
    
    This problem provides a graph database which consists of multiple graphs, and contains the following steps:
    
    1. Each graph is represented as the hashcode;  
    2. Pairwise hamming similarity calculation between the hashcodes;  
    3. Hamming-similarity-based Graph classification.
    
    We provide the following algorithms:
    
    - [Nested Subtree Hashing (NSH)](https://github.com/drhash-cn/graph-hashing/tree/main/nested-subtree-hash-kernels). Bin Li, Xingquan Zhu, Lianhua Chi, Chengqi Zhang. (2012). Nested Subtree Hash Kernels for Large-scale Graph Classification over Streams. Proceedings of the 12th International Conference on Data Mining. 399-408.
    - [K-Ary Tree Hashing (KATH)](https://github.com/drhash-cn/graph-hashing/tree/main/kath). Wei Wu, Bin Li, Ling Chen, Xingquan Zhu, Chengqi Zhang. (2018). K-Ary Tree Hashing for Fast Graph Classification. IEEE Transactions on Knowledge and Data Engineering. 30(5):936-949.
    - [SCHash](https://github.com/drhash-cn/graph-hashing/tree/main/schash). Xuan Tan, Wei Wu*, Chuan Luo. (2023). SCHash: Speedy Simplicial Complex Neural Networks via Randomized Hashing. to appear in SIGIR 2023.

- Network embedding for node classification, link prediction and node retrieval, etc.

    This task provides a network, and contains the following steps:
    
    1. Each node is represented as the hashcode;  
    2. Pairwise hamming similarity calculation between the hashcodes;  
    3. Hamming-similarity-based node classification, link prediction and node retrieval, etc.

    We provide the following algorithms:
    
    - [NetHash](https://github.com/drhash-cn/graph-hashing/tree/main/nethash). Wei Wu, Bin Li, Ling Chen, Chengqi Zhang. (2018). Efficient Attributed Network Embedding via Recursive Randomized Hashing. Proceedings of the 27th International Joint Conference on Artificial Intelligence. 2861-2867.
    - [#GNN](https://github.com/drhash-cn/graph-hashing/tree/main/hash-gnn). Wei Wu, Bin Li, Chuan Luo and Wolfgang Nejdl. (2021). Hashing-Accelerated Graph Neural Networks for Link Prediction. Proceedings of the 30th Web Conference. 2910-2920.
    - [MPSketch](https://github.com/drhash-cn/graph-hashing/tree/main/mpsketch). Wei Wu, Bin Li, Chuan Luo, Wolfgang Nejdl and Xuan Tan. (2023). MPSketch: Message Passing Networks via Randomized Hashing for Efficient Attributed Network Embedding. accpted by IEEE Transactions on Cybernetics.
