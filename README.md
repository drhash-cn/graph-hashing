# graph_hashing
A toolbox of randomized hashing algorithms for fast Graph Representation and Network Embedding. We provide two sets of graph hashing algorithms as follows:

- Graph kernels for graph classification
    
    This problem provides a graph database which consists of multiple graphs, and contains the following steps:
    
    1. Each graph is represented as the hashcode;  
    2. Pairwise hamming similarity calculation between the hashcodes;  
    3. Hamming-similarity-based Graph classification.
    
    We provide the following algorithms:
    
    - [Nested Subtree Hashing (NSH)](https://github.com/drhash-cn/graph-hashing/tree/main/nested-subtree-hash-kernels)
    - [K-Ary Tree Hashing (KATH)](https://github.com/drhash-cn/graph-hashing/tree/main/kath)

- Network embedding for node classification, link prediction and node retrieval, etc.

    This task provides a network, and contains the following steps:
    
    1. Each node is represented as the hashcode;  
    2. Pairwise hamming similarity calculation between the hashcodes;  
    3. Hamming-similarity-based node classification, link prediction and node retrieval, etc.

    We provide the following algorithms:
    
    - [NetHash](https://github.com/drhash-cn/graph-hashing/tree/main/nethash)
    - [#GNN](https://github.com/drhash-cn/graph-hashing/tree/main/hash-gnn)
    - [#GNN+](https://github.com/drhash-cn/graph-hashing/tree/main/hash-gnn-plus)
