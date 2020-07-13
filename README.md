#What is PUBG?
PUBG Stands for PlayerUnknown’s Battlegrounds. Basically the game is all about battle and the battle royal means all against all. This is similar to a hunger game where you start with nothing and with time you will scavenge and collect weapons and equipment.The game is ultimately a battle to the last player standing, with 100 players on an 8 x 8 km island. Mode of the games are: Solo, Duo or Squad.
To do the analysis on the data we will download the data from Kaggle and here is the source. Let’s have a look at the Data description which was taken from the Kaggle itself.
Feature descriptions (From Kaggle)
1.	* DBNOs – Number of enemy players knocked.
2.	* assists – Number of enemy players this player damaged that were killed by teammates.
3.	* boosts – Number of boost items used.
4.	* damageDealt – Total damage dealt. Note: Self inflicted damage is subtracted.
5.	* headshotKills – Number of enemy players killed with headshots.
6.	* heals – Number of healing items used.
7.	* Id – Player’s Id
8.	* killPlace – Ranking in match of number of enemy players killed.
9.	* killPoints – Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.) If there is a value other than -1 in rankPoints, then any 0 in killPoints should be treated as a “None”.
10.	* killStreaks – Max number of enemy players killed in a short amount of time.
11.	* kills – Number of enemy players killed.
12.	* longestKill – Longest distance between player and player killed at time of death. This may be misleading, as downing a player and driving away may lead to a large longestKill stat.
13.	* matchDuration – Duration of match in seconds.
14.	* matchId – ID to identify match. There are no matches that are in both the training and testing set.
15.	* matchType – String identifying the game mode that the data comes from. The standard modes are “solo”, “duo”, “squad”, “solo-fpp”, “duo-fpp”, and “squad-fpp”; other modes are from events or custom matches.
16.	* rankPoints – Elo-like ranking of player. This ranking is inconsistent and is being deprecated in the API’s next version, so use with caution. Value of -1 takes place of “None”.
17.	* revives – Number of times this player revived teammates.
18.	* rideDistance – Total distance traveled in vehicles measured in meters.
19.	* roadKills – Number of kills while in a vehicle.
20.	* swimDistance – Total distance traveled by swimming measured in meters.
21.	* teamKills – Number of times this player killed a teammate.
22.	* vehicleDestroys – Number of vehicles destroyed.
23.	* walkDistance – Total distance traveled on foot measured in meters.
24.	* weaponsAcquired – Number of weapons picked up.
25.	* winPoints – Win-based external ranking of player. (Think of this as an Elo ranking where only winning matters.) If there is a value other than -1 in rankPoints, then any 0 in winPoints should be treated as a “None”.
26.	* groupId – ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.
27.	* numGroups – Number of groups we have data for in the match.
28.	* maxPlace – Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.
29.	* winPlacePerc – The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match.
So I hope we got a brief about what is the game all about and the dataset as well what we are going to use.
So let’s divide the whole project into a few parts:
1.	Load the dataset
2.	Import the libraries
3.	Clean the data
4.	Perform Exploratory Data analysis
5.	Perform Feature engineering
6.	Build a Linear regression model 
7.	Predict the model
8.	Visualize actual and predicted value using matplotlib and seaborn library
1.	Load the dataset: Load the dataset from dropbox.  We already loaded the dataset into dropbox from Kaggle because it is easy to fetch the dataset from dropbox.
https://www.dropbox.com/s/kqu004pn2xpg0tr/train_V2.csv
To fetch the dataset from dropbox we need to use a command that is !wget and then the link like
!wget https://www.dropbox.com/s/kqu004pn2xpg0tr/train_V2.csv
!wget https://www.dropbox.com/s/5rl09pble4g6dk1/test_V2.csv
So our dataset is divided into two parts:
•	train_v2.csv
•	Test_v2. Csv
1.	Import the libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import gc
import os
import sys
%matplotlib inline
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
3. Use memory saving function:
 As the amount of dataset is too big, we need to use a memory saving function which will help us to reduce the memory usage.
The function also is taken from Kaggle itself:
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #    df[col] = df[col].astype(np.float16)
                #el
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else:
            #df[col] = df[col].astype('category')
 
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
4. Store the training data and use memory saving function to reduce memory usage:
train_data=pd.read_csv("train_V2.csv")
train_data= reduce_mem_usage(train_data)
train_data → This is the variable which holds the training part of the dataset
 Output:
 Memory usage of dataframe is 983.90 MB --> 339.28 MB (Decreased by 65.5%) 
5. Store the test data and use memory saving function to reduce memory usage:
test_data=pd.read_csv("/content/test_V2.csv")
test_data= reduce_mem_usage(test_data)
test_data → This is the variable which holds the testing part of the dataset
Output:
 Memory usage of dataframe is 413.18 MB --> 140.19 MB (Decreased by 66.1%)
6. Now we will check the dataset description as well as the shape of the dataset
•	The shape of training dataset:
o	Input: train_data.shape
o	Output: (4446966, 29)–> 4446966 rows and 29 columns
•	The shape of the testing dataset:
o	Input: test_data.shape
o	output: (1934174, 28)–> 4446966 rows and 28 columns
7. Print the training data: Print top 5 rows of data
train_data.head()
 
head() method returns the first five rows of the dataset
8. Print the testing data: Print top 5 rows of data
test_data.head()
 
Data cleaning:
1.	Checking the null values in the dataset: train_data.isna().any()
Output:
 Id                 False
groupId            False
matchId            False
assists            False
boosts             False
damageDealt        False
DBNOs              False
headshotKills      False
heals              False
killPlace          False
killPoints         False
kills              False
killStreaks        False
longestKill        False
matchDuration      False
matchType          False
maxPlace           False
numGroups          False
rankPoints         False
revives            False
rideDistance       False
roadKills          False
swimDistance       False
teamKills          False
vehicleDestroys    False
walkDistance       False
weaponsAcquired    False
winPoints          False
winPlacePerc        True
dtype: bool

 
So from the output, we can conclude that no column has null values except winPlaceperc.
Get the percentage for each column for null values:
null_columns=pd.DataFrame({'Columns':train_data.isna().sum().index,'No. Null values':train_data.isna().sum().values,'Percentage':train_data.isna().sum().values/train_data.shape[0]})
Output:
   
Exploratory Data Analysis:
Get the Statistical description of the dataset:
train_data.describe()
 
Now we will Find the unique id we have in the dataset:
•	    Input: train_data[“Id”].nunique()
•	    Output:4446966
.nunique() function is used to fetch the unique values from the dataset.
Now we will Find the unique group id and match id we have in the dataset:
•	Input: train_data[“groupId”].nunique()
•	Output: 2026745
•	Input: train_data[“matchId”].nunique()
•	Output: 47965
Match Type in the Game
There are 3 game modes in the game.  
•	— One can play solo
•	—  or with a friend (duo)
•	— or with 3 other friends (squad)
Input:
train_data.groupby(["matchType"]).count()
We use groupby() function to group the data based on the specified column
Output: 
 
Visualize the data using Python’s library:
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
train_data.groupby('matchId')['matchType'].first().value_counts().plot.bar()
Output: 
 
We know PUBG has three types of the match but in the dataset, we got more right?
Because PUBG has a criteria called fpp and tpp. So basically they are used to fixed the visualization like: zoom in or Zoom out.To solve this problem we need to map our data for specific three types of match:
Map the match function:
Input:
new_train_data=train_data
def mapthematch(data):
  mapping = lambda y: 'solo' if ('solo' in y) else 'duo' if('duo' in y) or ('crash' in y) else 'squad'
  data['matchType'] = data['matchType'].apply(mapping)
  return(new_train_data)
data=mapthematch(new_train_data)
data.groupby('matchId')['matchType'].first().value_counts().plot.bar()
Output:
 
So we map our data into three types of match
Find the Illegal match:
Input:
data[data['winPlacePerc'].isnull()]
 
Where WinPlaceperc is null and we will drop the column because the data is not correct.
data.drop(2744604, inplace=True)
Display the histogram of each map type:
   
Visualize the match duration:
data['matchDuration'].hist(bins=50)
 
Team kills based on Match Type
•	Solo
•	Duo
•	Squad
Input:
d=data[['teamKills','matchType']]
d.groupby('matchType').hist(bins=80)
   
Normalize the columns:
data['killsNormalization'] = data['kills']*((100-data['kills'])/100 + 1)
data['damageDealtNormalization'] = data['damageDealt']*((100-data['damageDealt'])/100 + 1)
 
 
data['maxPlaceNormalization'] = data['maxPlace']*((100-data['maxPlace'])/100 + 1)
 
data['matchDurationNormalization'] = data['matchDuration']*((100-data['matchDuration'])/100 + 1)
Let’s compare the actual and normalized data:
New_normalized_column = data[['Id','matchDuration','matchDurationNormalization','kills','killsNormalization','maxPlace','maxPlaceNormalization','damageDealt','damageDealtNormalization']]
 
Feature Engineering:
Before starting to apply the feature engineering let’s see what it is?
Feature engineering process is basically used to create a new feature from the existing data which helps to understand the data more deeply.
 
Create new features:
# Create new feature healsandboosts
data['healsandboostsfeature'] = data['heals'] + data['boosts']
data[['heals', 'boosts', 'healsandboostsfeature']].tail()
 
 Total distance travelled:
data['totalDistancetravelled'] = data['rideDistance'] + data['walkDistance'] + data['swimDistance']
data[['rideDistance', 'walkDistance', 'swimDistance',totalDistancetravelled]].tail()
 
#  headshot_rate feature
data['headshot_rate'] = data['headshotKills'] / data['kills']
Data['headshot_rate']
 
Now we will split our training data into two parts for:
•	Train the model (80%)
•	Test the model (20%)
•	And for validation purpose we will use test_v2.csv
x=data[['killsNormalization', 'damageDealtNormalization','maxPlaceNormalization', 'matchDurationNormalization','healsandboostsfeature','totalDistancetravelled']]
#drop the target variable
y=data['winPlacePerc']
 
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 42) 
Now create your own linear regression model:
linear=LinearRegression()
After the training predict your model using .predict() function with unknown dataset
y_pred=linear.predict(xtest)
 
Lastly we will visualize the actual and the predicted value of the model:
df1 = df.head(25)
df1.plot(kind='bar',figsize=(26,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
 
This brings us to the end of this article. If you are interested in data science and machine learning, click the banner below to get access to these free course

