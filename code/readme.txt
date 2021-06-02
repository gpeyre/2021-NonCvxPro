Lasso Demos:
Demo_Lasso.py: runs comparisons against CELER and Noncvx-Pro on libsvm datasets. The libsvm data is saved to the LassoDatasets folder.
Demo_Lasso.m: run this after running Demo_Lasso.py, this loads the results of CELER and then compares against 9 other solvers and Noncvx-Pro in Matlab.

Group Lasso Demos:
mne_datset.py: run this to retrieve a MEG/EEG dataset from the mne website.
Demo_MultitaskLasso.py: runs comparisons against CELER and Noncvx-Pro on synthetic and MEG/EEG datasets. The libsvm data is saved to the GroupLassoDatasets folder.
Demo_MultitaskLasso.m: run this after Demo_MultitaskLasso.py, this loads the results of CELER and then compares against 3 other solvers and Noncvx-Pro in Matlab.
Demo_group_EEG.m: runs comparison in Matlab (does not require first running Demo_MultitaskLasso.py)

Nuclear norm demo is in the Demo_MTFeat folder. See demo_MTfeat.m and follow instructions about how to download the benchmark datasets.

Basis Pursuit / W1 Optimal Transport Demos:
- Beckmann/Demo_BeckmannGraph.m: test on a 3D mesh (default: 20K vertex cortex) with finite element Laplacian.
- Beckmann/Demo_BeckmannMesh.m: test on a graph (default: K-NN graph on baker's yeast genomic expression) with the graph Laplacian.
See README in Beckmann/ for more details.

Nonconvex Lq minimisation demos:
See readme.m in lq/ for further details.
