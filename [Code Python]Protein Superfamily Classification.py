#!/usr/bin/env python
# coding: utf-8

# <h1 align="center"><font size="5">Protein Superfamily Classification Using Kernel
# Principal Component Analysis And Probabilistic
# Neural Networks</font></h1>

# # Introduction

# Protein superfamily classification focuses on organizing proteins into their super family and also predicting the family of
# newly discovered proteins. Correct classification helps in correct prediction of structure and function of new proteins which
# is one of the primary objective of computational biology and
# proteomics. Correct prediction of newly discovered proteins
# mainly concerns the Biologists or researchers for prediction of
# molecular function, drug discovery, medical diagnosis etc

# Protein classification can be done by classifying a new protein
# to a given family with previously known characteristics. The
# aim of classification is to predict target classes for given input
# protein. There are many approaches available for classification
# tasks, such as statistical techniques, decision trees and the neural networks. Neural networks have been chosen as technical
# tools for the protein super family classification task because:
# 

# • The extracted features of the protein sequences are distributed in a high dimensional space with complex
# characteristics which are difficult to satisfactorily model
# using some parametrized approaches.

# • The rules produced by decision tree techniques are complex and difficult to understand because the features are
# extracted from long character strings

# # Data and Methods

# ## Data collection

# <h1 align="center"><font size="2">Table 1: Details of Training and Test Sequence</font></h1>
# 
# | Protein Superfamily          | No. of training sequences  | No. of test sequences |
# |--------------------|---------|----------|
# | Globin (1000)            |  712      | 288    |
# | Kinase (750)            |        508   | 242    | 
# | Ligase (750)            |           530  | 220   |
# 

# ## Preprocessing

# Extract sequences from .fasta files and save them to .txt files

# In[1]:


import pandas as pd
from Bio import SeqIO
from IPython.display import display

list_kinase = list()
list_globin = list()
list_ligase = list()

#define function to Read from .fasta file and append every sequence to the list
def read_from_fasta_file(fasta_file_path):
    li = list()
    for seq_buffer in SeqIO.parse(fasta_file_path, "fasta"):
        #print(seq_buffer.seq)
        li.append(str(seq_buffer.seq))
    return li

list_kinase = read_from_fasta_file("Dataset/data_kinase")
list_globin = read_from_fasta_file("Dataset/data_globin")
list_ligase = read_from_fasta_file("Dataset/data_ligase")
    
    
print("Number of Kinase sequences : {} \nNumber of Hemoglobine sequences : {} \nNumber of Ligas sequences : {}".format(len(list_kinase),len(list_globin),len(list_ligase)))


#define function to save list as .csv file
def save_list_to_csv(list_to_save,file_name):
    
    dict = {'sequence': list_to_save}
    df = pd.DataFrame(dict)
    df.to_csv('Preprocessing/'+'file_name'+'.csv',index=False)
    

save_list_to_csv(list_kinase,"data_kinase")
save_list_to_csv(list_globin,"data_globin")
save_list_to_csv(list_ligase,"data_ligase")

print('\nFirst rows from data_kinase.csv file')
df = pd.read_csv('Preprocessing/data_kinase.csv')
display(df.head())

print('\nFirst rows from data_globin.csv file')
df = pd.read_csv('Preprocessing/data_globin.csv')
display(df.head())

print('\nFirst rows from data_ligase.csv file')
df = pd.read_csv('Preprocessing/data_ligase.csv')
display(df.head())


# ## Feature extraction

# Feature selection for protein superfamily classification is one of the major tasks, because every protein contains a huge amount of features. In real world a protein has many features and some of
# the features having very less significance in protein superfamily classification. The prime objective of feature selection is to eliminate the less significant feature that helps in accurate prediction and classification . Although, the feature that are eliminated
# may provide additional information that may improve the classification and prediction but adds
# the additional cost in classification and model may give different result. 

# Proteins (also known as polypeptides) are organic compounds made of amino acids arranged in a linear chain or
# folded into a globular form. The amino acids are joined
# together by the peptide bonds between the carboxyl and amino
# groups of adjacent amino acid residues. In general, the genetic
# code specifies 20 standard amino acids such as

# <h1 align="center"><font size="2">Σ = (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y )</font></h1>

# For protein feature selection, the two gram features such as

# <h1 align="center"><font size="2">(AA, AC, · · · AY ), (CA, CC, · · · CY ), · · · (Y A, Y C, · · · Y Y )</font></h1>

# are selected. The total number of possible bi-grams from a set
# of 20 amino acids is 20², that is, 400 .The two gram features
# represent the majority of the protein features

# bi-grams
# reflecting the pattern of substitution of amino acids are also
# extracted. For this purpose, equivalence classes of amino acids
# that substitue for one another are derived from the percent
# accepted mutation matrix (PAM).Exchange grams are
# similar but are based on a many to one translation of the
# amino acid alphabet into a six letter alphabet that represents
# six groups of amino acids, which represent high evolutionary
# similarity. Generally the exchange groups used are :

# <h1 align="center"><font size="2">e1 = {H, R, K}, e2 = {D, E, N, Q}, e3 = {C},
# e4 = {S, T, P, A, G}, e5 = {M, I, L, V }, and e6 = {F, Y, W}</font></h1>

# The total number of possible bi-grams on these six substitution groups is 6² = 36

# Besides that, the amino acid distribution (20) and
# exchange group distribution(6) are also taken into account.
# Consider the amino acid sequence as 

# <h1 align="center"><font size="2">MNNPQMNPQRS</font></h1>

# The extracted two-gram features are
# 

# <h1 align="center"><font size="2">(MN,2),(NN,1),(NP,2)(PQ,2)(QM,1)(QR,1)(RS,1)</font></h1>

# The above
# sequence can be denoted in terms of 6-letter exchange group
# as

# <h1 align="center"><font size="2">e5e2e2e4e2e5e2e4e2e1e4</font></h1>

# The two gram features of exchange group can be denoted as
# 

# <h1 align="center"><font size="2">{(e5e2, 2), (e2e2, 1), (e2e4, 2), (e4e2, 2), (e2e5, 1), (e2e1, 1), (e1e4, 1)}</font></h1>

# Besides the bigram measure and exchange group residues
# some other features are also extracted such as:

# <h1 align="center"><font size="2">X(1), X(2), · · · , X(5) = atomic composition<br>
# X(6) = molecular weight<br>
# X(7) = isoelectric point<br>
# X(8) = average mass of protein sequence<br>
# X(9), X(8), · · · X(28) = amino acid distribution<br>
# X(29), X(30), · · · X(428) = two gram distribution<br>
# X(429), X(430) · · · X(434) = exchange group distribution<br>
# X(435), X(434) · · · X(470) = two gram exchange group distribution<br></font></h1>

# Therefore, for every amino acid sequence, the <b>470 features</b>
# were processed to built the feature vector

# In[255]:


from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import numpy as np
import csv
from IPython.display import display


#Define function to extract features from data files .csv
def extract_features_from_file(dataframe,class_name):
    
    Y=list()
    test = pd.array(dataframe['sequence'])
    #print(test[0:5])

    sequence=list()
    for k in range(0,len(test)):
       
        my_seq = test[k]
        #my_seq = "MLSEQDRNIIKATVPVLEQHGATITSLFYKNMLNEHEELRNVFNRINQARGAQPAALATTVLAAAKHIDDLSVLAPYVNLIGHKHRALQIKPEQYPIVGHYLLQAIKQVLGDAATPEILSAWQKAYGVIADVFIDYEAQLYKEALWEGWQPFKVVSREHVAADIIEFTVAPQPGSGVELSKIPIVAGQYITVNVHPTTQGNKYDALRHYSICSESKDQGIKFAVKLENSYEHADGLVSEYLHHHVTVGDQILLSAPAGDFTLDESLIKQEKTPLVLMSSGVGATPLMAMLERQIKENPKRPIIWIQSSHEESRQAFKQKLEAISEKYDSFQKLVVHTSVQPRIGLPFLQKHVPSDADIYVCGSLPFMTSMLGYLDSLHHRNVHYELFGPKMVTVKA"
        analysed_seq = ProteinAnalysis(my_seq)
        sequence.append(my_seq)

        unigram=analysed_seq.count_amino_acids()
        uni=list(unigram.values())
        #print('unigram ',len(uni))
        #code for bigram 
        a='ACDEFGHIKLMNPQRSTVWY'
        b=[]
        for i in range(0,len(a)):
            for j in range(0,len(a)):
                b.append(a[i]+a[j])

        #c = [b[i:i+2] for i in range(len(b)-1)]

        bi= dict((letter,my_seq.count(letter)) for letter in set(b))
        bi= list(bi.values())
        #print('bigram ',len(bi))

        #forming exchange groups

        my_seq1=list(my_seq)

        for i in range(0,len(my_seq)):
            if my_seq1[i]=='H'or my_seq1[i]=='R'or my_seq1[i]=='K':
                my_seq1[i]='B'

            elif my_seq1[i]=='D'or my_seq1[i]=='E'or my_seq1[i]=='N'or my_seq1[i]=='Q':
                my_seq1[i]='J'

            elif my_seq1[i]=='C':
                my_seq1[i]='O'

            elif my_seq1[i]=='S'or my_seq1[i]=='T'or my_seq1[i]=='P'or my_seq1[i]=='A'or my_seq1[i]=='G':
                my_seq1[i]='U'

            elif my_seq1[i]== 'M'or my_seq1[i]=='I'or my_seq1[i]=='L'or my_seq1[i]=='V':
                my_seq1[i]='X'

            elif my_seq1[i]=='F'or my_seq1[i]=='Y'or my_seq1[i]=='W':
                my_seq1[i]='Z'

        my_seq1="".join(my_seq1)
    
        #print(my_seq1)
        #print()

        #unigram

        aex='BJOUXZ'
        uniex=dict((letter,my_seq1.count(letter)) for letter in set(aex))
        uniex=list(uniex.values())
        #print('uniex ',len(uniex))

        #bigram of exchange groups

        bex=[]
        for i in range(0,len(aex)):
            for j in range(0,len(aex)):
                bex.append(aex[i]+aex[j])

        biex=dict((letter,my_seq1.count(letter)) for letter in set(bex))
        biex=list(biex.values())
        #print('biex ',len(biex))


        #iso=analysed_seq.isoelectric_point()
        #mol=analysed_seq.molecular_weight()
        #print("mooool ",mol)

        X=[0]*470
        #fortest = ProteinAnalysis(my_seq)
        
        X[0] = analysed_seq.count_amino_acids()['C'] 
        #print("CCCC ",X[0])
        X[1] = analysed_seq.count_amino_acids()['H']
        #print("HHHH ",X[1])
        X[2] = analysed_seq.count_amino_acids()['N']
        #print("NNNN ",X[2])
        #X[3] = analysed_seq.count_amino_acids()['O']
        #print("OOOO ",X[3])
        X[4] = analysed_seq.count_amino_acids()['S']
        #print("SSSS ",X[4])
        X[5] = analysed_seq.molecular_weight()
        X[6] = analysed_seq.isoelectric_point()
        
        s=0
        for v in analysed_seq.count_amino_acids().values():
            s = s+v
       
        X[7] = analysed_seq.molecular_weight() / s
        
        X[8:28]=uni[:]

        X[28:428]=bi[:]

        X[428:434]=uniex[:]

        X[434:470]=biex[:]
        
        X.append(class_name)

        Y.append(X) 
        #print(Y[1:5])
    return Y


kinase = pd.read_csv("Preprocessing/data_kinase.csv")
globin = pd.read_csv("Preprocessing/data_globin.csv")
ligase = pd.read_csv("Preprocessing/data_ligase.csv")

features_kinase = extract_features_from_file(kinase,"Kinase")
print("features vector of 1st row (kinase)\n ",features_kinase[0])
print("length of vector ",len(features_kinase[1]),"\n")

features_globin = extract_features_from_file(globin,"Globin")
print("features vector of 1st row (globin)\n ",features_globin[0])
print("length of vector ",len(features_globin[1]),"\n")

features_ligase = extract_features_from_file(ligase,"Ligase")
print("features vector of 1st row (ligase)\n ",features_ligase[0])
print("length of vector ",len(features_ligase[1]))

print('Table length ',len(features_globin))
print('Table length ',len(features_kinase))
print('Table length ',len(features_ligase))


# In[256]:


#Extraxt final dataset with all superclasses and  features

d = np.concatenate((features_globin,features_kinase,features_ligase))
df = pd.DataFrame(d)
#df.to_csv("Preprocessing/Final_Data_Protein_Superclass.csv", sep=',',index=False)

#read from extracted file
#df = pd.read_csv("Preprocessing/Final_Data_Protein_Superclass.csv")
col_Names=[i for i in range(1,471)]
col_Names.append('Family')
df = pd.DataFrame(d,columns=col_Names)
display(df)
df.to_csv("Preprocessing/Final_Data_Protein_Superclass.csv", sep=',',index=False)
df = pd.read_csv("Preprocessing/Final_Data_Protein_Superclass.csv")
df.head()


# ## Methods of classification

# ### Principal Component Analysis(PCA)

# The concept of PCA was developed by Karl Pearson in 1901. Principal component analysis (PCA) is a statistical
# technique used to transform a data space of high dimension
# into a feature space of lower dimension having the most significant features.
# <br>Features (or inputs) that have little variance
# are thereby removed. PCA is an orthogonal transformation
# of the coordinate system in which the data are represented.
# The new transformed coordinate values by which data are
# represented are called principal components. A small number
# of principal components(PCs) are sufficient to represent most
# of the patterns in the data. 

# ### Kernel Principal Component Analysis(KPCA)

# Traditional PCA applies linear transformation which may
# not be effective when data are distributed non-linearly. In such
# a case, nonlinear PCA is used which is also referred as Kernel
# PCA. We apply nonlinear transformation to potentially very
# high-dimensional space by the use of integral operator kernel
# functions.<br> PCA is written here in the form of dot product and any one
# of the kernel trick is then applied. We decide kernel function
# a priori which can be either of the following types:
# <br>• Polynomial Kernel: k(x, y) = (x.y)d
# <br>• RBF kernel : k(x, y) = exp(− (x2−σy2)2 )
# <br>• Sigmoid : k(x, y) = tanh(k(x.y)) + Θ

# In our experiment, RBF kernel is used 

# ### Brief Overview of Probabilistic Neural Networks (PNN)

# The PNN is a multilayer feedforward network having four
# layers namely: input layer, hidden or pattern layer, summation
# layer, output or decision layer. The pattern layer has one
# pattern node for each training sample. The summation node
# or unit receives the outputs from the pattern nodes associated
# with a given class. It simply sums the outputs from the
# pattern nodes that correspond to the category from which
# the training pattern was selected.Thus, the number of nodes in the summation layer is same as the number of classes in
# multi class classification problem. The output node takes the
# decision of classifying the unknown sample to its respective
# class.

# 
# <img src="Preprocessing/Capture.PNG">

# ## Exprementation

# In[5]:


import pandas as pd 
import numpy as np
from IPython.display import display #Displaying Dataframes beautifully
import matplotlib.pyplot as plt #Plotting Graphs


# In[257]:


#Importing the dataset and having a look
data_set = pd.read_csv("Preprocessing/Final_Data_Protein_Superclass.csv")
print("randoms rows from our dataset:")
display(data_set.sample(10))


# In[258]:


#Making the X matrix and Y vector; X -> Input Features Y -> Ouput Vector

df = data_set.copy()
print("Shape of X: ", df.shape)
print("Some random rows from X:")
display(df.sample(10))


# In[259]:



X = np.array(df.drop(['Family'], axis=1))
Y = np.array(data_set['Family'])


print("Shape of X: ", X.shape)
print("Shape of Y: ",Y.shape)

print("random rows from from dataset:")
#display(data_set.loc[2284]['Family'])
display(data_set['Family'][2284])
display(data_set['Family'][969])
display(data_set['Family'][1724])
print("Same rows from Y:")
display(Y[2284])
display(Y[969])
display(Y[1724])


# In[260]:


from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA 
from sklearn.preprocessing import StandardScaler #For Standardization

# Standardizing the features
# KX = X.copy()
X = StandardScaler().fit_transform(X)
print("Shape of X after Standardization: ", X.shape)
display(X[0:5])


# In[342]:


#PCA and KPCA for a total of 2 principal axes
pca_object = PCA(n_components=2)
after_pca = pca_object.fit_transform(X)

print("Shape of Matrix after choosing 2 Principal Componenet from PCA: ", after_pca.shape)
display(after_pca[0:5])

kpca_object = KernelPCA(n_components=2, kernel='rbf',gamma=0.5)
after_kpca = kpca_object.fit_transform(X)

print("Shape of Matrix after choosing 2 Principal Componenet from KPCA: ", after_kpca.shape)
display(after_kpca[0:5])


# In[343]:


print('Plotting 2 PCs of PCA')
plt.plot(after_pca[:,0], after_pca[:,1], 'bo')
plt.ylabel('PC1')
plt.xlabel('PC2')
plt.show()
print('Plotting 2 PCs of KPCA')
plt.plot(after_kpca[:,0], after_kpca[:,1], 'bo')
plt.ylabel('KPC1')
plt.xlabel('KPC2')
plt.show()


# In[344]:


principalDf = pd.DataFrame(data = after_pca
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, data_set['Family']],axis=1)

display(finalDf)


# In[345]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Globin', 'Kinase', 'Ligase']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Family'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# In[346]:


KPCAprincipalDf = pd.DataFrame(data = after_kpca
             , columns = ['kernel principal component 1', 'kernel principal component 2'])

KPCAfinalDf = pd.concat([KPCAprincipalDf, data_set['Family']],axis=1)

display(KPCAfinalDf)


# In[347]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Kernel Principal Component 1', fontsize = 15)
ax.set_ylabel('Kernel Principal Component 2', fontsize = 15)
ax.set_title('2 component KPCA', fontsize = 20)
targets = ['Globin', 'Kinase', 'Ligase']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = KPCAfinalDf['Family'] == target
    ax.scatter(KPCAfinalDf.loc[indicesToKeep, 'kernel principal component 1']
               , KPCAfinalDf.loc[indicesToKeep, 'kernel principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# In[361]:


#PCA and KPCA for a total of 3 principal axes
pca_object = PCA(n_components=3)
after_pca = pca_object.fit_transform(X)

print("Shape of Matrix after choosing 3 Principal Componenet from PCA: ", after_pca.shape)
display(after_pca[0:5])

kpca_object = KernelPCA(n_components=3, kernel='rbf',gamma=1/470)
after_kpca = kpca_object.fit_transform(X)

print("Shape of Matrix after choosing 3 Principal Componenet from KPCA: ", after_kpca.shape)
display(after_kpca[0:5])


# In[269]:


from mpl_toolkits.mplot3d import Axes3D
print('Plotting 3 PCs of PCA')
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(after_pca[:,0], after_pca[:,1], after_pca[:,2])
plt.show()

print('Plotting 3 PCs of KPCA')
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(after_kpca[:,0], after_kpca[:,1], after_kpca[:,2])
plt.show()


# In[270]:


PCAprincipalDf = pd.DataFrame(data = after_pca
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])

PCAfinalDf = pd.concat([PCAprincipalDf, data_set['Family']],axis=1)

display(PCAfinalDf)


# In[271]:



fig = plt.figure()
ax = Axes3D(fig) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('3 component PCA', fontsize = 20)
targets = ['Globin', 'Kinase', 'Ligase']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = PCAfinalDf['Family'] == target
    ax.scatter(PCAfinalDf.loc[indicesToKeep, 'principal component 1']
               , PCAfinalDf.loc[indicesToKeep, 'principal component 2']
               , PCAfinalDf.loc[indicesToKeep, 'principal component 3']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# In[362]:


KPCAprincipalDf = pd.DataFrame(data = after_kpca
             , columns = ['kernel principal component 1', 'kernel principal component 2', 'kernel principal component 3'])

KPCAfinalDf = pd.concat([KPCAprincipalDf, data_set['Family']],axis=1)

display(KPCAfinalDf)


# In[363]:


fig = plt.figure()
ax = Axes3D(fig) 
ax.set_xlabel('Kernel Principal Component 1', fontsize = 15)
ax.set_ylabel('Kernel Principal Component 2', fontsize = 15)
ax.set_zlabel('Kernel Principal Component 3', fontsize = 15)
ax.set_title('3 component KPCA', fontsize = 20)
targets = ['Globin', 'Kinase', 'Ligase']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = KPCAfinalDf['Family'] == target
    ax.scatter(KPCAfinalDf.loc[indicesToKeep, 'kernel principal component 1']
               , KPCAfinalDf.loc[indicesToKeep, 'kernel principal component 2']
               , KPCAfinalDf.loc[indicesToKeep, 'kernel principal component 3']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# In[364]:


#PCA and KPCA for a total of 10 principal axes
pca_object = PCA(n_components=10)
after_pca = pca_object.fit_transform(X)

print("Shape of Matrix after choosing 10 Principal Componenet from PCA: ", after_pca.shape)
display(after_pca[0:5])

kpca_object = KernelPCA(n_components=10, kernel='rbf')
after_kpca = kpca_object.fit_transform(X)

print("Shape of Matrix after choosing 10 Principal Componenet from KPCA: ", after_kpca.shape)
display(after_kpca[0:5])


# In[365]:


from sklearn.model_selection import train_test_split #For splitting TRAIN and TEST dataset

#Train Test Split 
## PCA
X_train_pca, X_test_pca, Y_train_pca, Y_test_pca = train_test_split(after_pca, Y, train_size = 0.7, random_state=1)

print("PCA Shapes ")
print("X Train shape ",X_train_pca.shape)
print("Y Train shape ",Y_train_pca.shape)
print("X Test shape ",X_test_pca.shape)
print("Y Test shape ",Y_test_pca.shape)

## KPCA
X_train_kpca, X_test_kpca, Y_train_kpca, Y_test_kpca = train_test_split(after_kpca, Y, train_size = 0.7, random_state=1)

print("\nKPCA Shapes ")
print("X Train shape ",X_train_kpca.shape)
print("Y Train shape ",Y_train_kpca.shape)
print("X Test shape ",X_test_kpca.shape)
print("Y Test shape ",Y_test_kpca.shape)


# In[366]:


#split into superfamilies (Categories)
print("For Training Set in PCA: ")

train_PCA_g = X_train_pca[Y_train_pca == 'Globin']
train_PCA_k = X_train_pca[Y_train_pca == 'Kinase']
train_PCA_l = X_train_pca[Y_train_pca == 'Ligase']

print("PCA Globin Train Set Shape: ", train_PCA_g.shape)
print("PCA Kinase Train Set Shape: ", train_PCA_k.shape)
print("PCA Ligase Train Set Shape: ", train_PCA_l.shape)

print("For Training Set KPCA: ")

train_KPCA_g = X_train_kpca[Y_train_kpca == 'Globin']
train_KPCA_k = X_train_kpca[Y_train_kpca == 'Kinase']
train_KPCA_l = X_train_kpca[Y_train_kpca == 'Ligase']

print("KPCA Globin Train Set Shape: ", train_KPCA_g.shape)
print("KPCA Kinase Train Set Shape: ", train_KPCA_k.shape)
print("KPCA Ligase Train Set Shape: ", train_KPCA_l.shape)


# In[392]:


groups = data_set.groupby('Family')
number_of_classes = len(groups) #number of classes globin kinase ligase 
dictionary_of_sum = {}
numrber_of_components  = 10 # We have 10 PCAs
sigma = 0.3 #smoothing factor

# Implementation of PNN using PCA

print('_'*10 ,'PNN using PCA','_'*10)
PCA_array_of_classified_points = [] # to save final result PCA

#data points that we wish to classifiy
array_of_points = X_test_pca

# Loop via the number of data points that we wish to classifiy 
for point_want_to_classify in array_of_points:

    # INPUT LAYER OF THE PNN
    
    dictionary_of_sum['Globin'] = 0 
    sum_pdf_g = 0.0
    
    # Loop via the number of training example in globin class
    for globin in train_PCA_g:
        # PATTERN LAYER OF PNN
        sum = 0
        for i in range(0,numrber_of_components):
            sum =sum + np.power(point_want_to_classify[i]-globin[i],2)
        # Summation
        sum_pdf_g =sum_pdf_g + np.exp((-1*sum)/( 2 * np.power(sigma,2) ))
    dictionary_of_sum['Globin']  = sum_pdf_g
    
    dictionary_of_sum['Kinase'] = 0
    sum_pdf_k = 0.0
    
    # Loop via the number of training example in kinase class 
    for kinase in train_PCA_k:
        # PATTERN LAYER OF PNN
        sum = 0
        for i in range(0,numrber_of_components):
            sum =sum+ np.power(point_want_to_classify[i]-kinase[i],2)
        # Summation
        sum_pdf_k =sum_pdf_k + np.exp((-1*sum)/( 2 * np.power(sigma,2) ))
    dictionary_of_sum['Kinase']  = sum_pdf_k
    
    dictionary_of_sum['Ligase'] = 0 
    sum_pdf_l = 0.0
    
    # Loop via the number of training example in ligase class
    for ligase in train_PCA_l:
        # PATTERN LAYER OF PNN
        sum = 0
        for i in range(0,numrber_of_components):
            sum =sum+ np.power(point_want_to_classify[i]-ligase[i],2)
        # Summation
        sum_pdf_l =sum_pdf_l+ np.exp((-1*sum)/( 2 * np.power(sigma,2) ))
    dictionary_of_sum['Ligase']  = sum_pdf_l
    
    # Select the appropriate class (class with the highest value)
    classified_class = str( max(dictionary_of_sum, key=dictionary_of_sum.get) )
    
    # Add classfied clss to the array
    PCA_array_of_classified_points.append(classified_class)

print('shape of PNN PCA file ',np.array(PCA_array_of_classified_points).shape)
# save result to csv file
pd.DataFrame(PCA_array_of_classified_points,columns=['Family']).to_csv(
    "Result/PNN_PCA_file_of_classified_points.csv", sep=',',index=False)
print('File saved to Result/PNN_PCA_file_of_classified_points.csv')

#Read from file
fdf= pd.read_csv("Result/PNN_PCA_file_of_classified_points.csv")
display(fdf.sample(10))


# Implementation of PNN using KPCA

print('_'*10 ,'PNN using KPCA','_'*10)
dictionary_of_sum = {}
KPCA_array_of_classified_points = [] # to save final result PCA

#data points that we wish to classifiy
array_of_points = X_test_kpca

# Loop via the number of data points that we wish to classifiy 
for point_want_to_classify in array_of_points:

    # INPUT LAYER OF THE PNN
    
    dictionary_of_sum['Globin'] = 0 
    sum_pdf_g = 0.0
    
    # Loop via the number of training example in globin class
    for globin in train_KPCA_g:
        # PATTERN LAYER OF PNN
        sum = 0
        for i in range(0,numrber_of_components):
            sum =sum + np.power(point_want_to_classify[i]-globin[i],2)
        # Summation
        sum_pdf_g =sum_pdf_g + np.exp((-1*sum)/( 2 * np.power(sigma,2) ))
    dictionary_of_sum['Globin']  = sum_pdf_g
    
    dictionary_of_sum['Kinase'] = 0
    sum_pdf_k = 0.0
    
    # Loop via the number of training example in kinase class 
    for kinase in train_KPCA_k:
        # PATTERN LAYER OF PNN
        sum = 0
        for i in range(0,numrber_of_components):
            sum =sum + np.power(point_want_to_classify[i]-kinase[i],2)
        # Summation
        sum_pdf_k =sum_pdf_k + np.exp((-1*sum)/( 2 * np.power(sigma,2) ))
    dictionary_of_sum['Kinase']  = sum_pdf_k
    
    dictionary_of_sum['Ligase'] = 0 
    sum_pdf_l = 0.0
    
    # Loop via the number of training example in ligase class
    for ligase in train_KPCA_l:
        # PATTERN LAYER OF PNN
        sum = 0
        for i in range(0,numrber_of_components):
            sum =sum + np.power(point_want_to_classify[i]-ligase[i],2)
        # Summation
        sum_pdf_l =sum_pdf_l + np.exp((-1*sum)/( 2 * np.power(sigma,2) ))
    dictionary_of_sum['Ligase']  = sum_pdf_l
    
    # Select the appropriate class (class with the highest value)
    classified_class = str( max(dictionary_of_sum, key=dictionary_of_sum.get) )
    
    # Add classfied clss to the array
    KPCA_array_of_classified_points.append(classified_class)
    
print('shape of PNN KPCA file ',np.array(KPCA_array_of_classified_points).shape)
# save result to csv file
pd.DataFrame(KPCA_array_of_classified_points,columns=['Family']).to_csv(
    "Result/PNN_KPCA_file_of_classified_points.csv", sep=',',index=False)
print('File saved to Result/PNN_KPCA_file_of_classified_points.csv')

#Read from file
fdf= pd.read_csv("Result/PNN_KPCA_file_of_classified_points.csv")
display(fdf.sample(10))       

        


# ## Mesure Performance

# In[382]:


#Read from CSVs files

pnn_pca_result = pd.read_csv("Result/PNN_PCA_file_of_classified_points.csv")
pnn_kpca_result = pd.read_csv("Result/PNN_KPCA_file_of_classified_points.csv")

print("PCA Results Shape: ", pnn_pca_result.shape)
print("KPCA Results Shape: ", pnn_kpca_result.shape)

print("PNN_PCA Results:")
display(pnn_pca_result.sample(10))
print("PNN_KPCA Results:")
display(pnn_kpca_result.sample(10))


# In[369]:


from sklearn.metrics import jaccard_similarity_score as jss
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder #To convert categorical data to numerica data


# In[383]:


#Converting categorical to numerical; 
## 0 -> Globin 
## 1-> Kinase 
## 2-> Ligase
encoder = LabelEncoder()
Ynum = pd.Series(encoder.fit_transform(Y_test_pca)) 
pnn_kpca_num = pd.Series(encoder.fit_transform(pnn_kpca_result)) 
pnn_pca_num = pd.Series(encoder.fit_transform(pnn_pca_result)) 
display(pnn_pca_num.sample(10))


# In[384]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(Ynum,pnn_pca_num, labels=[0,1,2]))


# ## PNN PCA Performance meaures

# In[385]:


# Compute confusion matrix of PNN PCA
cnf_matrix = confusion_matrix(Ynum,pnn_pca_num, labels=[0,1,2])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix of PNN PCA
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Globin=0','Kinase=1','Ligase=2'],
                      normalize= False,  title='Confusion matrix PNN PCA')


# In[386]:


print (classification_report(Ynum,pnn_pca_num))


# ## PNN KPCA Performance meaures

# In[393]:


# Compute confusion matrix of PNN KPCA
cnf_matrix = confusion_matrix(Ynum,pnn_kpca_num, labels=[0,1,2])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix of PNN KPCA
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Globin=0','Kinase=1','Ligase=2']
                      ,normalize= False,  title='Confusion matrix PNN KPCA')


# In[388]:


print (classification_report(Ynum,pnn_kpca_num))


# <h1 align="center"><font size="4">
# Finally :</font></h1>

# <h1 ><font size="4">
#  - For PNN PCA Results depends on Smoothing factor(Sigma)
#   <br>- For PNN KPCA Result depends on Gamma (by default Gamma = 1/470 where 470 n_features) and Smoothnig factor (Sigma)<br>- The number of Principal Components PCs has no much effect on accuracy 3 to 10 Pcs gave good result but slowly were PCs are increased   
# </font></h1>

# In[ ]:




