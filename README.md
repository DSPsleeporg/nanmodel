## README: Codes used in this study
### Overview
NAN model and its derivatives are conductance-based (Hodgkin-Huxley type) neuron models using the mean-field approximation of groups of neurons and it is devived from averaged-neuron (AN) model. The model can recapitulate up-down oscillation (UDO).

### Dependencies
The following packages need to be installed to use codes used in this study:
matplotlib：3.7.5
numpy: 1.21.6 or 1.23.4
pandas: 0.23.4 or 2.0.3
requests: 2.19.0 or 2.31.0
scikit-learn: 1.3.2
seaborn: 0.13.2
scipy: 1.1.0 or 1.5.2
Network analyses were conducted with GPU in NVIDIA HPC SDK 22.11. Codes were written with C++17 and CUDA 11.8.

### Usage
All codes except for codes used in the network analysis are written in python. Codes used in the network analysis are written in C++ and CUDA.

### Description
The largest file CodeRev is divided into Basic, Fig1, 2, 3, 4, 5, FigS1, S2, S3, S4, S5, S6, S7. As the name suggests, these files contain all coded necessary for generating figures in this paper. 

##### Basic
This file contains CSV files (CodeRev/Basic/CSV) for all models used in this study, calculation of trajectory for all types of models used in this study, and the code used for choosing the representative parameter set. Both in the file "Original" and "Detailed", inside the file "anmodel", codes describing the models are stored. Most of the simulations conducted in this study are using codes in the folder "Original", except for some revised models. All simulations are conducted under the condition that the reversal potential is constant, except for simulations using codes in "Detailed" folders, where the reversal potential of Na+, K+ and Ca2+ change dynamically and are dependent on intracellular and extracellular Na+, K+ and Ca2+, respectively. 
Codes used for calculating the network dynamics are stored (CodeRev/Basic/Network). 
In the "anmodel" file (CodeRev/Basic/Original/anmodel), 6 files are stored. To run a parameter search, you can type "python search.py" to the terminal. Here are the descriptions of each file.

__init__.py: module for importing all files for parameter search
analysis.py: module for analysing and classifying the wave pattern
channels.py: module for storinig formulation for all channels incorporated in the models used in this study
models.py: module for solving ordinary differential equations in the models
params.py: module for storing parameters (constants) in the models
search.py: module for running the parameter search

In the "Detailed" file (CodeRev/Basic/Detailed), 6 files are stored (similar to "Original" file described above), and this is used for running the revised models used in the study, which took account the dynamics of intracellular and extracellular Na+, K+, and Ca2+ to calculate the reversal potential.

PCA.ipynb: This is the code used for obtaining the representative parameter set.

##### Fig1, 2, 3, 4, 5, FigS1, S2, S3, S4, S5, S6, S7
By running the file (the names of which end with ".ipynb"), you can get figures presented in this study.

Here are the descriptions of each module.
##### Fig1
1B.ipynb: Code used for generating Figure 1B.
1D.ipynb: Code used for generating Figure 1D.
1E: In the file "anmodel", code used for generating Figure 1E (1E.ipynb) is stored.
1F: In the file "anmodel", codes used for running bifurcation analysis (B0~B7.py) are stored. Get.ipynb is the code used for summarizing the data generated in the bifurcation analysis, and 1F.ipynb is the code used for generating Figure 1F.

##### Fig2
2A-G: In the file "anmodel", code used for generating Figure 1E (2A-G.ipynb) is stored.
2H.ipynb: Code used for generating Figure 2H.
2I.ipynb: Code used for generating Figure 2I.
2J.ipynb: Code used for generating Figure 2J.

##### Fig3
3A: In the file "Experiment", codes used for calculating the parameters of UDO are stored. For instance, file "cav" is stored the file "anmodel". In that file, by running "small.ipynb", parameters of UDO are calculated when the conductance of voltage-gated Ca2+ channels are slightly smaller than the original. By running "large.ipynb", parameters of UDO are calculated when the conductance of voltage-gated Ca2+ channels are slightly larger than the original. Note that csv file the name of which starting with "NAN" stores the information about parameter sets stored in A.csv which is used in this analysis.
In the file "Summary", by running the code "3A.ipynb", Figure 3A is generated. All csv files generated by running ".ipynb" files in "Experiment" folder must be stored in the file "Summary".
3B-G&I-J: In the file "anmodel, codes used for generating Figure 3B-G&J (3B-G&J.ipynb) and Figure 3I (3I.ipynb) are stored.
3H: In the file "Experiment", codes used for calculating ISI are stored. For instance, file "cav" is stored the file "anmodel". In that file, by running "original.ipynb", ISI of the up state in UDO are calculated. By running "modified.ipynb", ISI of awake firing pattern by modulating the conductance of voltage-gated Ca2+ channels are calculated. By running "stats.ipynb", the mean and standard deviation if ISI are calculated. Note that csv file the name of which starting with "NAN" stores the information about parameter sets stored in A.csv which is used in this analysis.
"3H.ipynb" is the code used for generating Figure 3H.

##### Fig4
4B&C&F-K: In the file "anmodel, code used for generating Figure 4B&C&F-K (4B&C&F-K.ipynb) is stored.
4D: In the file "anmodel", codes used for running bifurcation analysis (B0~B6.py) are stored. Get.ipynb is the code used for summarizing the data generated in the bifurcation analysis, and 4D.ipynb is the code used for generating Figure 4D.
4E: In tis file, code used for generating Figure 4E (Hist.ipynb) is stored.
4L-N: In the file "Experiment", codes used for calculating the parameters of UDO are stored. For instance, file "cav" is stored the file "anmodel". In that file, by running "small.ipynb", parameters of UDO are calculated when the conductance of voltage-gated Ca2+ channels are slightly smaller than the original. By running "large.ipynb", parameters of UDO are calculated when the conductance of voltage-gated Ca2+ channels are slightly larger than the original. Note that csv file the name of which starting with "NAN" stores the information about parameter sets stored in A.csv which is used in this analysis.
In the file "Summary", by running the code "4L-N.ipynb", Figure 4L-N and Figure S4C are generated. All csv files generated by running ".ipynb" files in "Experiment" folder must be stored in the file "Summary".
4O: In the file "Experiment", codes used for calculating ISI are stored. For instance, file "cav" is stored the file "anmodel". In that file, by running "original.ipynb", ISI of the up state in UDO are calculated. By running "modified.ipynb", ISI of awake firing pattern by modulating the conductance of voltage-gated Ca2+ channels are calculated. By running "stats.ipynb", the mean and standard deviation if ISI are calculated. Note that csv file the name of which starting with "NAN" stores the information about parameter sets stored in A.csv which is used in this analysis.
"4O.ipynb" is the Code used for generating Figure 4O.

##### Fig5
5A:In the file "anmodel", codes used for running single knockout analysis (SKO~SKO23.ipynb) are stored. After all these calculations are finished by running 5A.ipynb, Figure 5A is generated.
5B: In the file "Experiment", codes used for calculating ISI are stored. For instance, file "kna" is stored the file "anmodel". In that file, by running "original.ipynb", ISI of the up state in UDO are calculated. By running "modified.ipynb", ISI of awake firing pattern by modulating the conductance of KNa channels are calculated. By running "stats.ipynb", the mean and standard deviation if ISI are calculated. Note that csv file the name of which starting with "NAN" stores the information about parameter sets stored in A.csv which is used in this analysis.
"5B.ipynb" is the code used for generating Figure 5B.
5D: By running all codes the name of which end with ".cu", electical activity of neuronal network can be calculated. After that, by running "5D.ipynb", Figure 5D can be generated.
5E: By running "ISI.ipynb" in the file "anmodel", ISI of electrical activity in neuronal network can be calculated, and after that, by running "5E.ipynb", Figure 5E can be generated.

##### FigS1
S1A: In the file "anmodel" in file "LeakK" and "LeakNa", codes used for running bifurcation analysis (B0.py) are stored. Get.ipynb is the code used for summarizing the data generated in the bifurcation analysis, and Visualize.ipynb is the code used for generating Figure S1A.
S1B: In tis file, code used for generating Figure S1B (Hist.ipynb) is stored.
S1C: In the file "anmodel", code used for generating Figure S1C (S1C.ipynb) is stored.

##### FigS2
S2B: In tis file, code used for generating Figure S2B (Hist.ipynb) is stored.
S2C:In the file "anmodel", codes used for running bifurcation analysis (B0~B6.py) are stored. Get.ipynb is the code used for summarizing the data generated in the bifurcation analysis, and S2C.ipynb is the code used for generating Figure S2C.

##### FigS3
In the file "anmodel", code used for generating Figure S3A-E (S3A-E.ipynb) is stored.

##### FigS4
S4A: In the file "anmodel", codes used for running bifurcation analysis (B0~B6.py) are stored. Get.ipynb is the code used for summarizing the data generated in the bifurcation analysis, and S4A.ipynb is the code used for generating Figure S4A.
S4B:In tis file, code used for generating Figure S4B (Hist.ipynb) is stored.
S4C:In the file "Experiment", codes used for calculating the parameters of UDO are stored. For instance, file "cav" is stored the file "anmodel". In that file, by running "small.ipynb", parameters of UDO are calculated when the conductance of voltage-gated Ca2+ channels are slightly smaller than the original. By running "large.ipynb", parameters of UDO are calculated when the conductance of voltage-gated Ca2+ channels are slightly larger than the original. Note that csv file the name of which starting with "NAN" stores the information about parameter sets stored in A.csv which is used in this analysis.
In the file "Summary", by running the code "4L-N.ipynb", Figure 4L-N and Figure S4C are generated. All csv files generated by running ".ipynb" files in "Experiment" folder must be stored in the file "Summary".

##### FigS5
S5A:In the file "anmodel", codes used for running single knockout analysis (SKO~SKO24.ipynb) are stored. After all these calculations are finished by running S5A.ipynb, Figure S5A is generated.
S5B:In the file "kna" and "kca", by running "KNa_KO.ipynb" and "KCa_KO.ipynb" in each folder, the effect of KNa or KCa knockout can be measured. By running "S5B.ipynb", Figure S5B can be generated.
S5C:By running "S5C.ipynb" in the file "anmodel", Figure S5C can be generated.
S5D:In the file "Experiment", codes used for calculating the parameters of UDO are stored. For instance, file "cav" is stored the file "anmodel". In that file, by running "small.ipynb", parameters of UDO are calculated when the conductance of voltage-gated Ca2+ channels are slightly smaller than the original. By running "large.ipynb", parameters of UDO are calculated when the conductance of voltage-gated Ca2+ channels are slightly larger than the original. Note that csv file the name of which starting with "NAN" stores the information about parameter sets stored in A.csv which is used in this analysis.
In the file "Summary", by running the code "S5D.ipynb", Figure S5D is generated. All csv files generated by running ".ipynb" files in "Experiment" folder must be stored in the file "Summary".

##### FigS6
S6A:By running all codes the name of which end with ".cu", electical activity of neuronal network can be calculated. After that, by running "Prepare_Acsv.ipynb" and "S6A.ipynb", Figure S6A can be generated.
S6B:In the file "anmodel", codes used for running bifurcation analysis (B0~B19.py) are stored. Get.ipynb is the code used for summarizing the data generated in the bifurcation analysis, and S6B.ipynb is the code used for generating Figure S6B.

##### FigS7
S7A&B:In the file "anmodel", code used for generating Figure S7A&B (S7A&B.ipynb) is stored.
S7C:In the file "anmodel", code used for generating Figure S7C (S7C.ipynb) is stored.
