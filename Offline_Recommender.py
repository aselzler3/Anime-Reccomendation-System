from __future__ import division
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import requests
import ast

# V transpose as pandas dataframe
V_T_df=pd.read_csv('data/spark_V.csv')
#Indicies of V transpose that correspond to anime id.
id_map=pd.read_csv('data/id_mapping.csv')
# Average rating of each show corresponding to each anime id.
avg_show=pd.read_csv('data/avg_show_R.csv')
# number of people that watched each show corresponding to each anime id
num_views=pd.read_csv('data/num_views.csv')
item_mapping=np.array(id_map['col1'])
# Info about each show corresponding to each anime id. Start date, number of
# episodes, whether its adult or not, title of show, etc.
series_df=pd.read_csv('data/series_data.csv')
series_df['real_genres']=series_df['genres'].apply(lambda x: ast.literal_eval(x))
# G is a list of unique genres
G = []
for x in series_df['real_genres']:
    for y in x:
        if y not in G:
            G.append(y)
# Titles of all shows corresponding to each anime id
titles={}
for i in range(len(series_df)):
    ID = series_df['id'][i]
    title = series_df['title_english'][i]
    titles[ID] = title
# Popularity is number of people that viewed each show corresponding to each anime id.
popularity={}
for x in np.array(num_views):
    popularity[x[0]]=x[1]
V=np.array(V_T_df).T
# Average ratings for each show corresponding to each anime id.
average_ratings={}
for i in range(len(avg_show)):
    average_ratings[avg_show['anime_id'][i]]=avg_show['avg_for_show'][i]

# finds year show aired corresponding to each anime id
def StartYear(ID):
    if len(list(series_df[series_df['id']==ID]['start_date']))>0:
        if type(list(series_df[series_df['id']==ID]['start_date'])[0])==str:
            return int(list(series_df[series_df['id']==ID]['start_date'])[0][:4])
    else:
        return None

# finds whether show corresponding to anime id is adult or not.
def isAdult(ID):
    if len(list(series_df[series_df['id']==ID]['adult'])):
        return list(series_df[series_df['id']==ID]['adult'])[0]
    else:
        return None

# finds if a certain genre is associated with a show corresponding to anime id
def hasGenre(ID, genre):
    if len(list(series_df[series_df['id']==ID]['real_genres']))>0:
        return genre in list(series_df[series_df['id']==ID]['real_genres'])[0]
    else:
        return None


def numEpisodes(ID):
    if len(list(series_df[series_df['id']==ID]['total_episodes']))>0:
        return list(series_df[series_df['id']==ID]['total_episodes'])[0]
    else:
        return None

def Type(ID):
    if len(list(series_df[series_df['id']==ID]['type']))>0:
        return list(series_df[series_df['id']==ID]['type'])[0]
    else:
        return None




def recommendation_for_user():
    username=raw_input("Enter Anilist.co username: ")
    print
    url='https://anilist.co/api/'
    cid='selzla-6acux'
    sec='eGi4fmsY9pV64E1fSTWJJ1'
    params={'grant_type':"client_credentials",'client_id':cid,'client_secret':sec}
    access=requests.post(url+'auth/access_token',data=params)
    access_token=access.json()['access_token']
    user_anime=requests.get(url+'user/'+username+'/animelist?access_token='+access_token)
    completed = user_anime.json()['lists']['completed']


    if 'watching' in user_anime.json()['lists']:
        watching = user_anime.json()['lists']['watching']
    else:
        watching = []

    if 'dropped' in user_anime.json()['lists']:
        dropped = user_anime.json()['lists']['dropped']
    else:
        dropped = []

    if 'on_hold' in user_anime.json()['lists']:
        on_hold = user_anime.json()['lists']['on_hold']
    else:
        on_hold = []

    if 'plan_to_watch' in user_anime.json()['lists']:
        plan_to_watch = user_anime.json()['lists']['plan_to_watch']
    else:
        plan_to_watch = []

    scores = []
    for i in range(len(completed)):
        title=completed[i]['anime']['title_english'].encode("utf-8")
        anime_id=completed[i]['anime']['id']
        score=completed[i]['score_raw']
        scores.append([anime_id, score])
    user=np.array([row for row in scores if row[0] in item_mapping])
    user_vector = np.zeros(V.shape[1])
    for row in user:
        user_vector[list(item_mapping).index(row[0])]=row[1]+74-np.mean(user[:,1])-average_ratings[row[0]]
    new_user=user_vector[np.where(user_vector!=0)]
    new_V=V.T[np.where(user_vector!=0)].T
    s=np.linalg.lstsq(new_V.T,new_user.T)[0].T
    new_R=np.dot(s,V)
    for i in range(len(new_R)):
        ratio = (popularity[item_mapping[i]]-np.sqrt(popularity[item_mapping[i]]))/popularity[item_mapping[i]]
        new_R[i]=(new_R[i]+average_ratings[item_mapping[i]]+np.mean(user[:,1])-74)*ratio
    seen = [row[0] for row in scores]
    others = [watching, dropped, on_hold, plan_to_watch]
    for x in others:
        for i in range(len(x)):
            seen.append(x[i]['anime']['id'])
    rec=np.argsort(-1*new_R)
    rec=[item_mapping[x] for x in rec]

    type_filter=raw_input("Filter by type? For example only show movies (yes/no): ")
    if type_filter=='yes':
        type_filter=True
    else:
        type_filter=False
    print
    if type_filter:
        kind_type=raw_input("What type of anime would you like to view? Possible \
answers are (TV, Movie, Special, TV Short, OVA, ONA, Music) Note: most anime \
series are in type 'TV': ")
        while kind_type not in series_df['type'].unique():
            kind_type=raw_input("Not a valid answer. Possible \
answers are (TV, Movie, Special, TV Short, OVA, ONA, Music) Note: most anime \
series are in type 'TV': ")
    else:
        kind_type='TV'
    print

    genre_filter=raw_input("Filter by genre? For example Action, Drama, Slice of Life (yes/no): ")
    if genre_filter=='yes':
        genre_filter=True
    else:
        genre_filter=False
    print
    if genre_filter:
        kind_genre=raw_input("What genre of anime would you like to view? Possible \
answers are (Action, Adventure, Comedy, Drama, Ecchi, Fantasy, Horror, Mahou Shoujo, \
Mecha, Music, Mystery, Psychological, Romance, \
Sci-Fi, Slice of Life, Sports, Supernatural, Thriller): ")
        while kind_genre not in G:
            kind_genre=raw_input("Not a valid answer. Possible \
answers are (Action, Adventure, Comedy, Drama, Ecchi, Fantasy, Horror, Mahou Shoujo, \
Mecha, Music, Mystery, Psychological, Romance, \
Sci-Fi, Slice of Life, Sports, Supernatural, Thriller: ")
    else:
        kind_genre='Action'
    print
    year_filter=raw_input("Filter by year show was released? (yes/no): ")
    if year_filter=='yes':
        year_filter=True
    else:
        year_filter=False
    print
    if year_filter:
        min_year=input("Enter earliest year of release of shows you would like \
to view. Note: this must be a 4 digit whole number! (eg 2005): ")
        max_year=input("Enter latest year of release of shows you would like \
to view. Note: this must be a 4 digit whole number! (eg 2010): ")
    else:
        min_year=2005
        max_year=2010
    print

    episodes_filter=raw_input("Filter by number of episodes show has? (yes/no): ")
    if episodes_filter=='yes':
        episodes_filter=True
    else:
        episodes_filter=False
    print
    if episodes_filter:
        min_episodes=input("Enter smallest amount of episodes show contains that \
you would like to view. Note: this must be a whole number! (eg 24): ")
        max_episodes=input("Enter largest amount of episodes show contains that \
you would like to view. Note: this must be a whole number! (eg 300): ")
    else:
        min_episodes=1
        max_episodes=1000

    print
    adult_filter=True
    kind_adult=raw_input("Watch adult shows? (yes/no) Warning: choosing 'yes' will suggest \
shows unsuitable for viewers under the age of 18!: ")
    if kind_adult=='yes':
        kind_adult=True
    else:
        kind_adult=False

    print
    popularity_filter=raw_input("Filter by number of people that have completed \
the show? (yes/no) Note: you should choose yes if you would like to view popular \
shows: ")
    if popularity_filter=='yes':
        popularity_filter=True
    else:
        popularity_filter=False
    print
    if popularity_filter:
        min_popularity=input("Enter minimum number of people that have watched the shows\
you would like to view. Note: this must be a whole number! (eg 100) A good number \
to use if you would like to view well known shows is ~5000: ")
    else:
        min_popularity=1
    print

    f_type = (type_filter, kind_type)
    f_genres = (genre_filter, kind_genre)
    f_year = (year_filter, min_year, max_year)
    f_episodes = (episodes_filter, min_episodes, max_episodes)
    f_adult = (adult_filter, kind_adult)
    f_popularity = (popularity_filter, min_popularity)
    count = 0
    for i in range(6000):
        if rec[i] not in seen:
            allowed = True
            if f_popularity[0]:
                if popularity[rec[i]]<f_popularity[1]:
                    allowed = False
            if f_type[0]:
                if Type(rec[i])!=f_type[1]:
                    allowed = False
            if f_genres[0]:
                if not hasGenre(rec[i], f_genres[1]):
                    allowed = False
            if f_year[0]:
                if StartYear(rec[i])<f_year[1] or StartYear(rec[i])>f_year[2]:
                    allowed = False
            if f_episodes[0]:
                if numEpisodes(rec[i])<f_episodes[1] or numEpisodes(rec[i])>f_episodes[2]:
                    allowed = False
            if f_adult[0]:
                if isAdult(rec[i])!=f_adult[1]:
                    allowed = False
            if allowed:
                print rec[i], titles[rec[i]], new_R[list(item_mapping).index(rec[i])]
                count +=1
        if count>15:
            break
    if count==0:
        print "Couldn't find any shows that match your criteria"

def recommendation_for_non_user():
    new_R=np.zeros(V.shape[1])
    for i in range(len(new_R)):
        ratio = (popularity[item_mapping[i]]-np.sqrt(popularity[item_mapping[i]]))/popularity[item_mapping[i]]
        new_R[i]=average_ratings[item_mapping[i]]*ratio
    rec=np.argsort(-1*new_R)
    rec=[item_mapping[x] for x in rec]

    type_filter=raw_input("Filter by type? For example only show movies (yes/no): ")
    if type_filter=='yes':
        type_filter=True
    else:
        type_filter=False
    print
    if type_filter:
        kind_type=raw_input("What type of anime would you like to view? Possible \
answers are (TV, Movie, Special, TV Short, OVA, ONA, Music) Note: most anime \
series are in type 'TV': ")
        while kind_type not in series_df['type'].unique():
            kind_type=raw_input("Not a valid answer. Possible \
answers are (TV, Movie, Special, TV Short, OVA, ONA, Music) Note: most anime \
series are in type 'TV': ")
    else:
        kind_type='TV'
    print

    genre_filter=raw_input("Filter by genre? For example Action, Drama, Slice of Life (yes/no): ")
    if genre_filter=='yes':
        genre_filter=True
    else:
        genre_filter=False
    print
    if genre_filter:
        kind_genre=raw_input("What genre of anime would you like to view? Possible \
answers are (Action, Adventure, Comedy, Drama, Ecchi, Fantasy, Horror, Mahou Shoujo, \
Mecha, Music, Mystery, Psychological, Romance, \
Sci-Fi, Slice of Life, Sports, Supernatural, Thriller): ")
        while kind_genre not in G:
            kind_genre=raw_input("Not a valid answer. Possible \
answers are (Action, Adventure, Comedy, Drama, Ecchi, Fantasy, Horror, Mahou Shoujo, \
Mecha, Music, Mystery, Psychological, Romance, \
Sci-Fi, Slice of Life, Sports, Supernatural, Thriller: ")
    else:
        kind_genre='Action'
    print
    year_filter=raw_input("Filter by year show was released? (yes/no): ")
    if year_filter=='yes':
        year_filter=True
    else:
        year_filter=False
    print
    if year_filter:
        min_year=input("Enter earliest year of release of shows you would like \
to view. Note: this must be a 4 digit whole number! (eg 2005): ")
        max_year=input("Enter latest year of release of shows you would like \
to view. Note: this must be a 4 digit whole number! (eg 2010): ")
    else:
        min_year=2005
        max_year=2010
    print

    episodes_filter=raw_input("Filter by number of episodes show has? (yes/no): ")
    if episodes_filter=='yes':
        episodes_filter=True
    else:
        episodes_filter=False
    print
    if episodes_filter:
        min_episodes=input("Enter smallest amount of episodes show contains that \
you would like to view. Note: this must be a whole number! (eg 24): ")
        max_episodes=input("Enter largest amount of episodes show contains that \
you would like to view. Note: this must be a whole number! (eg 300): ")
    else:
        min_episodes=1
        max_episodes=1000

    print
    adult_filter=True
    kind_adult=raw_input("Watch adult shows? (yes/no) Warning: choosing 'yes' will suggest \
shows unsuitable for viewers under the age of 18!: ")
    if kind_adult=='yes':
        kind_adult=True
    else:
        kind_adult=False

    print
    popularity_filter=raw_input("Filter by number of people that have completed \
the show? (yes/no) Note: you should choose yes if you would like to view popular \
shows: ")
    if popularity_filter=='yes':
        popularity_filter=True
    else:
        popularity_filter=False
    print
    if popularity_filter:
        min_popularity=input("Enter minimum number of people that have watched the shows\
you would like to view. Note: this must be a whole number! (eg 100) A good number \
to use if you would like to view well known shows is ~5000: ")
    else:
        min_popularity=1
    print

    f_type = (type_filter, kind_type)
    f_genres = (genre_filter, kind_genre)
    f_year = (year_filter, min_year, max_year)
    f_episodes = (episodes_filter, min_episodes, max_episodes)
    f_adult = (adult_filter, kind_adult)
    f_popularity = (popularity_filter, min_popularity)
    count = 0
    for i in range(6000):
        allowed = True
        if f_popularity[0]:
            if popularity[rec[i]]<f_popularity[1]:
                allowed = False
        if f_type[0]:
            if Type(rec[i])!=f_type[1]:
                allowed = False
        if f_genres[0]:
            if not hasGenre(rec[i], f_genres[1]):
                allowed = False
        if f_year[0]:
            if StartYear(rec[i])<f_year[1] or StartYear(rec[i])>f_year[2]:
                allowed = False
        if f_episodes[0]:
            if numEpisodes(rec[i])<f_episodes[1] or numEpisodes(rec[i])>f_episodes[2]:
                allowed = False
        if f_adult[0]:
            if isAdult(rec[i])!=f_adult[1]:
                allowed = False
        if allowed:
            print rec[i], titles[rec[i]], new_R[list(item_mapping).index(rec[i])]
            count +=1
        if count>15:
            break
    if count==0:
        print "Couldn't find any shows that match your criteria"



has_account=raw_input("Welcome! This program suggests anime for you to watch. Do you have an Anilist \
account? (yes/no) Note: it is okay if you don't have one: ")
if has_account=='yes':
    recommendation_for_user()
else:
    recommendation_for_non_user()
