import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import gseapy as gp
import os
import seaborn as sns

#import variance stabilised DESeq2 dataset
vsd = pd.read_csv('data/full_vsd_matrix.csv', index_col = 0)

#PCA analysis
print("starting PCA analysis")
from sklearn.decomposition import PCA
tvsd = np.transpose(vsd)  #transpose data to shape (samples, genes)
pca_model = PCA()
coordinates = pca_model.fit_transform(tvsd) 
eigenvalues = pca_model.explained_variance_
ratios = pca_model.explained_variance_ratio_
cumulative_ratios = np.cumsum(ratios)
print("PCA analysis complete")

print("creating scree plot")
#scree plot
idx = np.arange(1, len(ratios)+1)

fig, ax1 = plt.subplots()

ax1.plot(idx,ratios, 'b')      
ax1.plot(idx,ratios, 'bo')     
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Explained Variance', color = 'b')
ax1.set_ylim(0,max(ratios * 1.1))
ax1.tick_params(axis = 'y', labelcolor = 'b')

ax2 = ax1.twinx()
ax2.plot(idx, cumulative_ratios, 'r')   
ax2.plot(idx, cumulative_ratios, 'ro')  
ax2.set_ylabel('Cumulative Variance', color = 'r')
ax2.set_ylim(0,1.03)
ax2.tick_params(axis = 'y', labelcolor = 'r')

plt.tight_layout()
plt.savefig('results/scree_plot.png')
plt.close()
print("scree plot saved")


#set experimental design matrix
design = pd.read_csv('data/metadata.txt', sep = ' ', header = None)
design.columns = ['sample', 'condition']
groups = design['condition'] 

sample_labels = ['sample1', 'sample2', 'sample3', 'sample4', 'sample5', 'sample6']
design.index = sample_labels
groups.index = sample_labels


#PCA plot
print("creating PCA plot")
sns.scatterplot(x = coordinates[:,0], y = coordinates[:,1], s = 100, 
                hue = groups, palette = sns.color_palette('Paired')[0:10])
plt.xlabel(f"PC1 ({ratios[0]*100:.2f}%)") 
plt.ylabel(f"PC2 ({ratios[1]*100:.2f}%)") 
plt.legend(bbox_to_anchor = (1,1))
plt.tight_layout()
plt.savefig('results/PCA_plot.png')
plt.close()
print("PCA plot saved")


#clustering
print("performing hierarchical clustering")
Z = sch.linkage(tvsd, method = 'complete')

dend = sch.dendrogram(Z, 
                      labels = groups, 
                      orientation = 'left', 
                      color_threshold = 0.7 * max(Z[:,2]))

plt.title("Hierarchical Clustering")
plt.xlabel("Distance")
plt.ylabel("Sample")
plt.tight_layout()
plt.savefig('results/hierarchical_clustering.png')
plt.close()
print("hierarchical clustering plot saved")

#differential expression analysis
print("performing differential expression analysis")
def analyse_de(filepath):
    
    data = pd.read_csv(filepath, index_col=0)

    up = (data['log2FoldChange'] > 1) &(data['padj'] < 0.05)
    up_genes = data[up]['external_gene_name']
    
    down = (data['log2FoldChange'] < -1) &(data['padj'] < 0.05)
    down_genes = data[down]['external_gene_name']

    print("Number of upregulated genes:" + str(len(up_genes)))
    print("Number of downregulated genes:" + str(len(down_genes)))

    return up_genes, down_genes

#Het vs WT
up_het_wt, down_het_wt = analyse_de("data/Het_vs_WT/salmon_Het_WT_DEGs_annotated.csv")
#Hom vs WT
up_hom_wt, down_hom_wt = analyse_de("data/Hom_vs_WT/salmon_Hom_WT_DEGs_annotated.csv")
#Hom vs Het
up_het_hom, down_het_hom = analyse_de("data/Het_vs_Hom/salmon_Het_Hom_DEGs_annotated.csv")


#dictionary of DEG lists
comparisons = {
    'Het_vs_WT_Up': list(up_het_wt),
    'Het_vs_WT_Down': list(down_het_wt),
    'Hom_vs_WT_Up': list(up_hom_wt),
    'Hom_vs_WT_Down': list(down_hom_wt),
    'Het_vs_Hom_Up': list(up_het_hom),
    'Het_vs_Hom_Down': list(down_het_hom)}

#save DE gene lists
print("saving differentially expressed gene lists")
def save_gene_list(comparisons):
    base_dir = r'results'
    
    for comp, genes in comparisons.items():
         if 'Up' in comp:
            comparison = comp.replace('_Up', '')
            direction = 'upregulated'
         elif 'Down' in comp:
            comparison = comp.replace('_Down', '')
            direction = 'downregulated'
         else:
            continue 
            
         comp_dir = os.path.join(base_dir, comparison)
         os.makedirs(comp_dir, exist_ok=True)

         fname = f'{direction}_genes_{comparison}.txt'
         file_path = os.path.join(comp_dir, fname)

         with open(file_path, 'w') as f:
            for gene in genes:
                f.write(f"{gene}\n")

save_gene_list(comparisons)
print("differentially expressed gene lists saved")

#enrichment analysis
print("performing enrichment analysis")
def plot_enr(gene_list, comparison_name):

    outdir = os.path.join("results", "enrichment", comparison_name)
    os.makedirs(outdir, exist_ok=True)
    
    enr = gp.enrichr(
        gene_list=gene_list,
        gene_sets=['GO_Biological_Process_2021'],
        organism='Mouse',
        outdir=outdir,
        no_plot=True,
        cutoff=0.05)

    top_terms = (enr.res2d.sort_values('Adjusted P-value').head(25).assign(Gene_Count=lambda df: df['Genes'].str.count(';') + 1))

    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=top_terms,
        x='Gene_Count',
        y='Term',
        palette='viridis')
    plt.xlabel('Number of Genes')
    plt.ylabel('GO Biological Process')
    plt.tight_layout()

    plot_path = os.path.join("results", "enrichment", comparison_name, f'{comparison_name}_barplot.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()

for name, gene_list in comparisons.items():
    plot_enr(gene_list, name)

print("enrichment analysis saved")
