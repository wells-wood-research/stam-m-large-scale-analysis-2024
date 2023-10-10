library(TreeDist)

# Define the directory path where the .nwk files are located
directory_path <- "~/GitRepos/illuminating-protein-structural-universe/pdb_af2_structural_analysis/analysis/clustering/af2/iso_for_0.0/phylo_tree_comparison_output/"

# Get a list of all .nwk files in the directory
nwk_files <- list.files(directory_path, pattern = "\\.nwk$", full.names = TRUE)

# Specify the path to the reference .nwk file
reference_nwk_file <- "~/GitRepos/illuminating-protein-structural-universe/pdb_af2_structural_analysis/data/processed_data/ncbi_phylo_tree.phy"

# Load the reference tree
reference_tree <- ape::read.tree(reference_nwk_file)

# Create an empty list to store the pairwise distances
distances_list <- list()

# Loop through each .nwk file, load the tree, and calculate the distance to the reference tree
for (nwk_file in nwk_files) {
  # Load the tree from the current .nwk file
  current_tree <- ape::read.tree(nwk_file)

  distance <- TreeDistance(reference_tree, current_tree)
  
  # Store the distance in the list, using the file name as the key
  file_name <- basename(nwk_file)
  distances_list[[file_name]] <- distance
  
}

print(distances_list)

# Convert the named list to a data frame
distances_df <- as.data.frame(distances_list)

# Specify the file path for the CSV file
csv_file <- "~/GitRepos/illuminating-protein-structural-universe/pdb_af2_structural_analysis/analysis/clustering/af2/iso_for_0.0/phylo_tree_comparison_output/tree_distances.csv"

# Export the data frame to a CSV file
write.csv(distances_df, file = csv_file, row.names = FALSE)


expectedCID <- randomTreeDistances["cid", "mean", "48"]
print(expectedCID)
