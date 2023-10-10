# Load the ape package if not already loaded
library(ape)

# Define the directory path where the .nwk files are located
directory_path <- "~/GitRepos/illuminating-protein-structural-universe/pdb_af2_structural_analysis/analysis/clustering/af2/iso_for_0.0/phylo_tree_comparison_output/"

# Get a list of all .nwk files in the directory
nwk_files <- list.files(directory_path, pattern = "\\.nwk$", full.names = TRUE)

# Specify the path to the reference .nwk file
reference_nwk_file <- "~/GitRepos/illuminating-protein-structural-universe/pdb_af2_structural_analysis/data/processed_data/ncbi_phylo_tree.phy"

# Load the reference tree
reference_tree <- read.tree(reference_nwk_file)

reference_tree$node.label <- NULL

# Create an empty list to store the pairwise distances
distances_list <- list()

# Loop through each .nwk file, load the tree, and calculate the distance to the reference tree
for (nwk_file in nwk_files) {
  # Load the tree from the current .nwk file
  current_tree <- read.tree(nwk_file)
  
  # Calculate the topological distance between the current tree and the reference tree
  dist_value <- dist.topo(current_tree, reference_tree)
  
  # Calculating the max score
  maximum_rf_score <- length(tree1$edge) + length(tree2$edge)
  print(maximum_rf_score)
  
  # Normalising score
  normalised_dist_score = dist_value/maximum_rf_score
  
  # Store the distance in the list, using the file name as the key
  file_name <- basename(nwk_file)
  distances_list[[file_name]] <- normalised_dist_score
}

# Print or use the distances_list as needed
print(distances_list)





