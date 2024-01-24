# NFL Special Teams Analysis Using Tracking DATA
#By Tyran Johnson


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

```


```python
#Importing file

file_path = '/Volumes/TEST1/Notebook projects/FOOTBALL/tracking2020.csv'
nfl_data = pd.read_csv(file_path)

```


```python
# Viewing data structure

nfl_data.info(), nfl_data.head()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1048575 entries, 0 to 1048574
    Data columns (total 19 columns):
     #   Column     Non-Null Count    Dtype  
    ---  ------     --------------    -----  
     0   game_id    1048575 non-null  int64  
     1   player_id  1003064 non-null  float64
     2   play_id    1048575 non-null  int64  
     3   date_id    1048575 non-null  float64
     4   frame_id   1048575 non-null  int64  
     5   time       1048575 non-null  object 
     6   x          1048575 non-null  float64
     7   y          1048575 non-null  float64
     8   s          1048575 non-null  float64
     9   a          1048575 non-null  float64
     10  dis        1048575 non-null  float64
     11  o          1003064 non-null  float64
     12  dir        1003064 non-null  float64
     13  event      1048575 non-null  object 
     14  name       1048575 non-null  object 
     15  number     1003064 non-null  float64
     16  position   1003064 non-null  object 
     17  team       1048575 non-null  object 
     18  play_dir   1048575 non-null  object 
    dtypes: float64(10), int64(3), object(6)
    memory usage: 152.0+ MB





    (None,
           game_id  player_id  play_id    date_id  frame_id  \
     0  2021010300    42901.0       40  1965782.2         1   
     1  2021010300    42901.0       40  1965782.3         2   
     2  2021010300    42901.0       40  1965782.4         3   
     3  2021010300    42901.0       40  1965782.5         4   
     4  2021010300    42901.0       40  1965782.6         5   
     
                           time      x      y     s     a   dis       o     dir  \
     0  2021-01-03T18:03:02.200  61.21  46.77  0.08  0.13  0.01  186.21  144.73   
     1  2021-01-03T18:03:02.300  61.22  46.77  0.11  0.19  0.01  184.87  126.09   
     2  2021-01-03T18:03:02.400  61.23  46.76  0.10  0.18  0.01  183.38  108.95   
     3  2021-01-03T18:03:02.500  61.24  46.76  0.11  0.22  0.01  180.24   91.57   
     4  2021-01-03T18:03:02.600  61.25  46.76  0.05  0.20  0.01  172.44  119.91   
     
       event          name  number position  team play_dir  
     0  None  Dean Marlowe    31.0       SS  home     left  
     1  None  Dean Marlowe    31.0       SS  home     left  
     2  None  Dean Marlowe    31.0       SS  home     left  
     3  None  Dean Marlowe    31.0       SS  home     left  
     4  None  Dean Marlowe    31.0       SS  home     left  )



### Count of All Special Team Instances in 2020


```python
# Filtering for events that are typically associated with special teams
# Common events for special teams include kickoffs, punts, and field goals
special_teams_events = ['kickoff', 'punt', 'field_goal']

# Filtering the dataset for these events
special_teams_data = nfl_data[nfl_data['event'].isin(special_teams_events)]

# Checking the shape of the filtered data
special_teams_data.shape
```




    (10414, 19)



### Top 10 Fastest Recorded Players on Special Teams


```python
# Calculating the maximum speed reached by each player in special teams plays
max_speeds = special_teams_data.groupby(['player_id', 'name'])['s'].max().reset_index()

# Sorting the data to find the top speeds
max_speeds_sorted = max_speeds.sort_values(by='s', ascending=False).head(10)

# Creating a bar plot for the top 10 fastest times downfield
plt.figure(figsize=(10, 6))
sns.barplot(x='s', y='name', data=max_speeds_sorted, palette="viridis")
plt.xlabel('Maximum Speed (yards/second)')
plt.ylabel('Player Name')
plt.title('Top 10 Fastest Players in Special Teams Plays (2020)')
plt.show()

```


    
![png](output_7_0.png)
    


### Discovering the Average Distance Traveled by Player Position


```python
# Grouping the data by player and calculating the total distance traveled during special teams plays
total_distance_per_player = special_teams_data.groupby(['player_id', 'name', 'position'])['dis'].sum().reset_index()

# Calculating the average distance traveled for each position
average_distance_per_position = total_distance_per_player.groupby('position')['dis'].mean().reset_index()

# Sorting the results for better visualization
average_distance_per_position_sorted = average_distance_per_position.sort_values(by='dis', ascending=False)

# Creating a bar plot for the average distance traveled by position
plt.figure(figsize=(12, 8))
sns.barplot(x='dis', y='position', data=average_distance_per_position_sorted, palette="mako")
plt.xlabel('Average Distance Traveled (yards)')
plt.ylabel('Position')
plt.title('Average Distance Traveled by Player Position in Special Teams Plays (2020)')
plt.show()

```


    
![png](output_9_0.png)
    


### Total Distribution of Special Teams Plays By Type


```python
# Counting the number of each type of special teams play
play_type_counts = special_teams_data['event'].value_counts().reset_index()
play_type_counts.columns = ['play_type', 'count']
```


```python
# Redefining the color palette for consistency with previous visualizations
colors = sns.color_palette("mako", n_colors=play_type_counts['play_type'].nunique())

# Redoing the pie chart with the new color theme
plt.figure(figsize=(8, 8))
plt.pie(play_type_counts['count'], labels=play_type_counts['play_type'], autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Distribution of Special Teams Play Types (2020)')
plt.show()
```


    
![png](output_12_0.png)
    


### Discovering the Average Direction Players by Postion Travel during Special Team Plays


```python
# Calculating the average orientation (o) and direction (dir) for each position
avg_orientation_direction = special_teams_data.groupby('position').agg(
    avg_orientation=('o', 'mean'),
    avg_direction=('dir', 'mean')
).reset_index()

# Visualizing the average orientation and direction
plt.figure(figsize=(14, 8))

# Scatter plot for average orientation and direction
sns.scatterplot(data=avg_orientation_direction, x='avg_orientation', y='avg_direction', hue='position', palette="mako", s=100)
plt.xlabel('Average Orientation (degrees)')
plt.ylabel('Average Direction (degrees)')
plt.title('Average Orientation and Direction by Player Position in Special Teams Plays (2020)')
plt.legend(title='Position', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()
```


    
![png](output_14_0.png)
    


# Average Speeds By Postion on Special Team Plays


```python
# Calculating the average maximum acceleration for each position
avg_max_acceleration = special_teams_data.groupby(['position'])['a'].max().reset_index()

# Sorting the results for better visualization
avg_max_acceleration_sorted = avg_max_acceleration.sort_values(by='a', ascending=False)

# Creating a bar plot for the average maximum acceleration by position
plt.figure(figsize=(12, 8))
sns.barplot(x='a', y='position', data=avg_max_acceleration_sorted, palette="mako")
plt.xlabel('Average Maximum Acceleration (yards/second^2)')
plt.ylabel('Position')
plt.title('Average Maximum Acceleration by Player Position in Special Teams Plays (2020)')
plt.show()

```


    
![png](output_16_0.png)
    


### Discovering what direction and speed most plays occur


```python
# Calculating average speed and total distance for each play direction
play_direction_analysis = special_teams_data.groupby('play_dir').agg(
    avg_speed=('s', 'mean'),
    total_distance=('dis', 'sum')
).reset_index()

# Creating a dual-axis bar and line plot for the comparison
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar plot for average speed
sns.barplot(x='play_dir', y='avg_speed', data=play_direction_analysis, ax=ax1, color='lightblue', alpha=0.6)
ax1.set_ylabel('Average Speed (yards/second)')
ax1.set_xlabel('Play Direction')
ax1.set_title('Average Speed and Total Distance by Play Direction in Special Teams (2020)')

# Line plot for total distance on secondary y-axis
ax2 = ax1.twinx()
sns.lineplot(x='play_dir', y='total_distance', data=play_direction_analysis, ax=ax2, color='darkblue', marker='o')
ax2.set_ylabel('Total Distance Covered (yards)')

plt.show()

```


    
![png](output_18_0.png)
    



```python
# The dual-axis plot illustrates the comparison of average speed and total distance covered by players 
# in special teams plays based on the direction of the play (left or right) during the 2020 NFL season. 
# The bar plot represents the average speed, while the line plot shows the total distance covered for each 
# play direction.This visualization helps in understanding if the play direction significantly impacts the 
# players' speed and the distance they cover.
```