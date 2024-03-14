import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the dataset
df = pd.read_csv("Mice Protein Expression data.csv")

# Extract the relevant columns
t_colname_I = df[['DYRK1A_N', 'ITSN1_N', 'BDNF_N', 'NR1_N', 'NR2A_N', 'pAKT_N', 'pBRAF_N', 'pCAMKII_N', 'pCREB_N', 'pELK_N', 'pERK_N', 'pJNK_N', 'PKCA_N', 'pMEK_N', 'pNR1_N', 'pNR2A_N', 'pNR2B_N', 'pPKCAB_N', 'pRSK_N', 'AKT_N', 'BRAF_N', 'CAMKII_N', 'CREB_N', 'ELK_N', 'ERK_N', 'GSK3B_N', 'JNK_N', 'MEK_N', 'TRKA_N', 'RSK_N', 'APP_N', 'Bcatenin_N', 'SOD1_N', 'MTOR_N', 'P38_N', 'pMTOR_N', 'DSCR1_N', 'AMPKA_N', 'NR2B_N', 'pNUMB_N', 'RAPTOR_N', 'TIAM1_N', 'pP70S6_N', 'NUMB_N', 'P70S6_N', 'pGSK3B_N', 'pPKCG_N', 'CDK5_N', 'S6_N', 'ADARB1_N', 'AcetylH3K9_N', 'RRP1_N', 'BAX_N', 'ARC_N', 'ERBB4_N', 'nNOS_N', 'Tau_N', 'GFAP_N', 'GluR3_N', 'GluR4_N', 'IL1B_N', 'P3525_N', 'pCASP9_N', 'PSD95_N', 'SNCA_N', 'Ubiquitin_N', 'pGSK3B_Tyr216_N', 'SHH_N', 'BAD_N', 'BCL2_N', 'pS6_N', 'pCFOS_N', 'SYP_N', 'H3AcK18_N', 'EGR1_N', 'H3MeK4_N', 'CaNA_N']].values

# Impute missing values with median
imputer = SimpleImputer(missing_values=np.NaN, strategy='median')
t_colname_I = imputer.fit_transform(t_colname_I)

# Create a DataFrame with the imputed values
df_1_I = pd.DataFrame(t_colname_I, columns=['DYRK1A_N', 'ITSN1_N', 'BDNF_N', 'NR1_N', 'NR2A_N', 'pAKT_N', 'pBRAF_N', 'pCAMKII_N', 'pCREB_N', 'pELK_N', 'pERK_N', 'pJNK_N', 'PKCA_N', 'pMEK_N', 'pNR1_N', 'pNR2A_N', 'pNR2B_N', 'pPKCAB_N', 'pRSK_N', 'AKT_N', 'BRAF_N', 'CAMKII_N', 'CREB_N', 'ELK_N', 'ERK_N', 'GSK3B_N', 'JNK_N', 'MEK_N', 'TRKA_N', 'RSK_N', 'APP_N', 'Bcatenin_N', 'SOD1_N', 'MTOR_N', 'P38_N', 'pMTOR_N', 'DSCR1_N', 'AMPKA_N', 'NR2B_N', 'pNUMB_N', 'RAPTOR_N', 'TIAM1_N', 'pP70S6_N', 'NUMB_N', 'P70S6_N', 'pGSK3B_N', 'pPKCG_N', 'CDK5_N', 'S6_N', 'ADARB1_N', 'AcetylH3K9_N', 'RRP1_N', 'BAX_N', 'ARC_N', 'ERBB4_N', 'nNOS_N', 'Tau_N', 'GFAP_N', 'GluR3_N', 'GluR4_N', 'IL1B_N', 'P3525_N', 'pCASP9_N', 'PSD95_N', 'SNCA_N', 'Ubiquitin_N', 'pGSK3B_Tyr216_N', 'SHH_N', 'BAD_N', 'BCL2_N', 'pS6_N', 'pCFOS_N', 'SYP_N', 'H3AcK18_N', 'EGR1_N', 'H3MeK4_N', 'CaNA_N'])

# Extract relevant columns for target variable encoding
t_colname_D = df[['Genotype', 'Treatment', 'Behavior', 'class']].values

# Label encode the categorical variables
LE_t_colname_D = LabelEncoder()
for i in range(4):
    t_colname_D[:, i] = LE_t_colname_D.fit_transform(t_colname_D[:, i])

# Create DataFrame with label encoded variables
df_2_D = pd.DataFrame(t_colname_D, columns=['Genotype', 'Treatment', 'Behavior', 'class'])
df_2_D = df_2_D.astype({'Genotype': 'int', 'Treatment': 'int', 'Behavior': 'int', 'class': 'int'})

# Extract the target variable for the model
df_3_D = df.iloc[:, 81]

# Concatenate encoded variables with the features
df_1_I = pd.concat([df_1_I, df_2_D.drop(['class'], axis=1)], axis=1)

# Split the dataset for dependent variable "Behavior"
I_train, I_test, D2_train, D2_test = train_test_split(df_1_I, df_3_D, test_size=0.2, random_state=0)

# Model training using Support Vector Machine (SVM) algorithm
model = SVC(kernel='linear')
model.fit(I_train, D2_train)

# To predict the output "Behavior"
df_fopt1 = model.predict(I_test)

# Evaluate the model
accc = model.score(I_test, D2_test)
print(accc)
