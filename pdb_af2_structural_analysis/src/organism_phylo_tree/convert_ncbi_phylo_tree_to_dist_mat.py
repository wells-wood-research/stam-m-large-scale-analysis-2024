import dendropy

tree = dendropy.Tree.get(
    path="pdb_af2_structural_analysis/data/processed_data/ncbi_phylo_tree.phy",
    schema="newick",
)
pdm = tree.phylogenetic_distance_matrix()
pdm.write_csv("pdb_af2_structural_analysis/data/processed_data/ncbi_dist_mat.csv")
