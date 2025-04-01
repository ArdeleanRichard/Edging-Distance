# A path-based distance computation for non-convexity with applications in clustering
Clustering algorithms are essential in data analysis, but evaluating their performance is challenging when the true labels are not available and non-convex clusters are present. Traditional performance evaluation metrics struggle to identify the correctness of clustering, often evaluating linearly separated clusters with higher scores than the true clusters. We propose a novel way to compute distance such that the structure of the data is taken into consideration; thus, improving the correctness of evaluation for non-convex clusters while not affecting that of convex clusters. We validated the proposed method on several benchmark synthetic datasets of various characteristics: simple convex cluster, overlapped and imbalanced clusters and non-convex clusters. Moreover, besides the true and random labels which are required for any analysis, each of these datasets are paired with labels generated from linear separation to show the ineffectiveness of traditional methods and to verify that the proposed method has overcome this weakness. The applicability of this method is not limited to clustering performance evaluation metrics, as an example, we show a modified version of K-Means using the proposed method that is capable of correctly separating non-convex clusters.

Here, we introduce a novel approach for distance computation, denominated Edging Distance (ED), that was designed specifically for non-convex clusters. The proposed approach addresses the limitations of the Euclidian distance by iterating through the data points to discover the structure of the data. By taking into consideration the structure of the data in the computation of the distance, the proposed method offers a better estimation of the distances in and between clusters that enhances the performance of clustering and clustering evaluation metrics. The structure of the data is integrated into the distance computation based on principles from graph theory, the distance is calculated as a path between the points which can account for the complex shapes of non-convex clusters.


# Citations
We would appreciate it, if you cite the paper when you use this work for the original ED algorithm:

- For Plain Text:
```
Ardelean, ER., Portase, R.L., Potolea, R. et al. A path-based distance computation for non-convexity with applications in clustering. Knowl Inf Syst 67, 1415–1453 (2025). https://doi.org/10.1007/s10115-024-02275-4
```

- BibTex:
```
@ARTICLE{Ardelean2025-kh,
  title     = "A path-based distance computation for non-convexity with
               applications in clustering",
  author    = "Ardelean, Eugen-Richard and Portase, Raluca Laura and Potolea,
               Rodica and D{\^\i}nșoreanu, Mihaela",
  journal   = "Knowl. Inf. Syst.",
  publisher = "Springer Science and Business Media LLC",
  volume    =  67,
  number    =  2,
  pages     = "1415--1453",
  month     =  feb,
  year      =  2025,
  copyright = "https://creativecommons.org/licenses/by/4.0",
  language  = "en"
}
```

# Contact
If you have any questions about SBM, feel free to contact me. (Email: ardeleaneugenrichard@gmail.com)
