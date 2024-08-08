# DR-Submodular
This project focuses on **developing a method to convert social network data, usually represented in traditional graph format, into a bipartite graph.**  The main goal is to utilize this conversion for optimization problems within social networks, particularly focusing on maximizing the impact of marketing campaigns.

Here's a breakdown:

* **Problem:** Companies want to advertise efficiently on social media by reaching the most potential customers within budget. This involves identifying influential users in relevant communities and "seeding" information to them. Existing bipartite graph datasets aren't tailored for social networks. 
* **Solution:** The project converts regular social network graphs into bipartite graphs. 
    * One side of the bipartite graph represents communities within the social network.
    * The other side represents individual users. 
    * The weight of edges connecting communities and users is calculated as the user's influence within that community.
* **Process:**
    1. A traditional social network graph (e.g., Facebook friendships) is constructed.
    2. Community detection algorithms (Greedy Modularity and Directed Louvain) are used to identify communities within this graph.
    3. The identified communities become one set of nodes in the bipartite graph (V₁), individual users become the other set (V₂). 
    4. Edges are created between community nodes (V₁) and user nodes (V₂) if the user belongs to that community. Edge weights reflect the user's influence within the community.
* **Benefits:** This conversion creates a bipartite graph ideal for tackling "influence maximization" problems in social media marketing:
    * Allocate marketing budgets efficiently by targeting influential users within the most relevant communities.
    * Optimize the spread of information by understanding community structures and user influence within them.
* **Future Research:** 
    * Improving the community detection algorithms for speed, accuracy, and scope. 
    * Using this weighted bipartite graph as input for solving  DR-submodular maximization problems on integer lattices – leading to more sophisticated marketing solutions. 

This project provides a practical approach to translate social network data into a format suited for optimizing influence-based marketing campaigns. Its potential applications are diverse, impacting market segmentation, online advertising, and information diffusion within social media platforms. 
