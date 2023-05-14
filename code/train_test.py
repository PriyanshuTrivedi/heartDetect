import pandas as pd
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
warnings.filterwarnings('ignore', 'Solver terminated early.*')

#Importing Dataset
df = pd.read_csv("heart.csv")
raw_df=df

# DATA PREPROCESSING

# --- Creating Dummy Variables for cp, thal and slope ---
cp = pd.get_dummies(df['cp'], prefix='cp')
thal = pd.get_dummies(df['thal'], prefix='thal')
slope = pd.get_dummies(df['slope'], prefix='slope')

# --- Merge Dummy Variables to Main Data Frame ---
frames = [df, cp, thal, slope]
df = pd.concat(frames, axis = 1)

# --- Drop Unnecessary Variables ---
df = df.drop(columns = ['cp', 'thal', 'slope'])

# --- Seperating Dependent Features ---
x = df.drop(['target'], axis=1)
y = df['target']

# --- Data Normalization using Min-Max Method ---
x = MinMaxScaler().fit_transform(x)

processed_df=df

# DATA SPLITTING

# --- Splitting Dataset into 75:25 ---
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=6)