from __future__ import division
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import requests
import ast
from flask import render_template, Flask, request
from app import app

def startyear(string):
    try:
        result = int(string[:4])
    except:
        result = 1
    return result

V_T_df=pd.read_csv('data/spark_V.csv')
id_map=pd.read_csv('data/id_mapping.csv')
avg_show=pd.read_csv('data/avg_show_R.csv')
num_views=pd.read_csv('data/num_views.csv')
V_T_df['id'] = id_map['col1']

series_df=pd.read_csv('data/series_data.csv')
series_df['real_genres']=series_df['genres'].apply(lambda x: ast.literal_eval(x))

series_df['in'] = series_df['id'].apply(lambda x: x in np.array(V_T_df['id']))
series_df=series_df[series_df['in']]
series_df=series_df.sort_values('id')

V_T_df['in'] = V_T_df['id'].apply(lambda x: x in np.array(series_df['id']))
V_T_df = V_T_df[V_T_df['in']]

avg_show['in']=avg_show['anime_id'].apply(lambda x: x in np.array(series_df['id']))
avg_show=avg_show[avg_show['in']]

num_views['in']=num_views['anime_id'].apply(lambda x: x in np.array(series_df['id']))
num_views=num_views[num_views['in']]

series_df['average_rating']=np.array(avg_show['avg_for_show'])
series_df['num_views']=np.array(num_views['num_views'])
item_mapping = np.array(series_df['id'])

G = []
for x in series_df['real_genres']:
    for y in x:
        if y not in G:
            G.append(y)

titles={}
for i in range(len(series_df)):
    ID = np.array(series_df['id'])[i]
    title = np.array(series_df['title_english'])[i]
    titles[ID] = title

popularity={}
for i in range(len(series_df)):
    ID = np.array(series_df['id'])[i]
    avr = np.array(series_df['num_views'])[i]
    popularity[ID] = avr

V=np.array(V_T_df[['col'+str(i) for i in range(1,11)]]).T

average_ratings={}
for i in range(len(avg_show)):
    ID = np.array(series_df['id'])[i]
    avr = np.array(series_df['average_rating'])[i]
    average_ratings[ID] = avr

def recommendation_for_user(text, filter_by_type, TYPE, filter_by_genre, GENRE,
filter_by_year, min_year, max_year, filter_by_episodes, min_episodes, max_episodes,
filter_by_popularity, min_popularity, series_df):
    Recommendations = []
    url='https://anilist.co/api/'
    cid='selzla-6acux'
    sec='eGi4fmsY9pV64E1fSTWJJ1'
    params={'grant_type':"client_credentials",'client_id':cid,'client_secret':sec}
    access=requests.post(url+'auth/access_token',data=params)
    access_token=access.json()['access_token']
    user_anime=requests.get(url+'user/'+text+'/animelist?access_token='+access_token)
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
    series_df['predicted_ratings'] = new_R

    f_type = (filter_by_type, TYPE)
    f_genres = (filter_by_genre, GENRE)
    f_year = (filter_by_year, min_year, max_year)
    f_episodes = (filter_by_episodes, min_episodes, max_episodes)
    f_popularity = (filter_by_popularity, min_popularity)

    if f_type[0]:
        series_df = series_df[series_df['type']==f_type[1]]
    if f_genres[0]:
        series_df['has_genre'] = series_df['real_genres'].apply(lambda x: f_genres[1] in x)
        series_df = series_df[series_df['has_genre']]
    if f_year[0]:
        series_df['start_year'] = series_df['start_date'].apply(startyear)
        series_df = series_df[series_df['start_year']>=f_year[1]]
        series_df = series_df[series_df['start_year']<=f_year[2]]
    if f_episodes[0]:
        series_df = series_df[series_df['total_episodes']>=f_episodes[1]]
        series_df = series_df[series_df['total_episodes']<=f_episodes[2]]
    if f_popularity[0]:
        series_df = series_df[series_df['num_views']>=f_popularity[1]]
    series_df = series_df.sort_values('predicted_ratings', ascending=False)
    series_df['not_seen'] = series_df['id'].apply(lambda x: x not in seen)
    series_df = series_df[series_df['not_seen']]
    rec_titles = np.array(series_df['title_english'])
    rec_descriptions = np.array(series_df['description'])
    rec_ids = np.array(series_df['id'])
    for i in range(len(series_df)):
        Recommendations.append({'title':str(rec_titles[i]).decode("utf8"),
'description':str(rec_descriptions[i]).decode("utf8"), 'id':'https://anilist.co/anime/'+str(rec_ids[i]),'picture':'https://cdn.anilist.co/img/dir/anime/reg/'+str(rec_ids[i])+'.jpg'})
        if i>19:
            break


    return Recommendations

def recommendation_for_non_user(filter_by_type, TYPE, filter_by_genre, GENRE,
filter_by_year, min_year, max_year, filter_by_episodes, min_episodes, max_episodes,
filter_by_popularity, min_popularity, series_df):
    Recommendations = []

    new_R=np.zeros(V.shape[1])
    for i in range(len(new_R)):
        ratio = (popularity[item_mapping[i]]-np.sqrt(popularity[item_mapping[i]]))/popularity[item_mapping[i]]
        new_R[i]=average_ratings[item_mapping[i]]*ratio

    series_df['predicted_ratings'] = new_R

    f_type = (filter_by_type, TYPE)
    f_genres = (filter_by_genre, GENRE)
    f_year = (filter_by_year, min_year, max_year)
    f_episodes = (filter_by_episodes, min_episodes, max_episodes)
    f_popularity = (filter_by_popularity, min_popularity)

    if f_type[0]:
        series_df = series_df[series_df['type']==f_type[1]]
    if f_genres[0]:
        series_df['has_genre'] = series_df['real_genres'].apply(lambda x: f_genres[1] in x)
        series_df = series_df[series_df['has_genre']]
    if f_year[0]:
        series_df['start_year'] = series_df['start_date'].apply(startyear)
        series_df = series_df[series_df['start_year']>=f_year[1]]
        series_df = series_df[series_df['start_year']<=f_year[2]]
    if f_episodes[0]:
        series_df = series_df[series_df['total_episodes']>=f_episodes[1]]
        series_df = series_df[series_df['total_episodes']<=f_episodes[2]]
    if f_popularity[0]:
        series_df = series_df[series_df['num_views']>=f_popularity[1]]
    series_df = series_df.sort_values('predicted_ratings', ascending=False)
    rec_titles = np.array(series_df['title_english'])
    rec_descriptions = np.array(series_df['description'])
    rec_ids = np.array(series_df['id'])
    for i in range(len(series_df)):
        Recommendations.append({'title':str(rec_titles[i]).decode("utf8"),
'description':str(rec_descriptions[i]).decode("utf8"), 'id':'https://anilist.co/anime/'+str(rec_ids[i]),'picture':'https://cdn.anilist.co/img/dir/anime/reg/'+str(rec_ids[i])+'.jpg'}) 
        if i>19:
            break

    return Recommendations

@app.route('/')
def submission_page():
    return '''
        <html>
          <head>
            <title></title>
            <title>Welcome to microblog</title>
          </head>
          <body>
            <h1>Welcome!</h1>
            <p>This program suggests animes for you to watch. Pick and choose which
            filters you would like to apply to customize your search. Note: you do not
            need an Anilist.co account to use this service.</p>
          </body>
        </html>
        <form action="/recommendation" method='POST' >
            Anilist.co username: <br>
            <input type="text" name="user_name" /><br>
            <p></p>
            Minimum release year (eg 2005): <br>
            <input type="text" name="min_year" /><br>
            <p></p>
            Maximum release year (eg 2016): <br>
            <input type="text" name="max_year" /><br>
            <p></p>
            <select name="Type">
                <option value="Filter by Type">Filter by Type</option>
                <option value="TV">TV</option>
                <option value="Movie">Movie</option>
                <option value="Special">Special</option>
                <option value="TV Short">TV Short</option>
                <option value="OVA">OVA</option>
                <option value="ONA">ONA</option>
                <option value="Music">Music</option>
            </select><br>
            <p></p>
            <select name="Genre">
                <option value="Filter by Genre">Filter by Genre</option>
                <option value="Action">Action</option>
                <option value="Adventure">Adventure</option>
                <option value="Comedy">Comedy</option>
                <option value="Drama">Drama</option>
                <option value="Ecchi">Ecchi</option>
                <option value="Fantasy">Fantasy</option>
                <option value="Horror">Horror</option>
                <option value="Mahou Shoujo">Mahou Shoujo</option>
                <option value="Mecha">Mecha</option>
                <option value="Music">Music</option>
                <option value="Mystery">Mystery</option>
                <option value="Psychological">Psychological</option>
                <option value="Romance">Romance</option>
                <option value="Sci-Fi">Sci-Fi</option>
                <option value="Slice of Life">Slice of Life</option>
                <option value="Sports">Sports</option>
                <option value="Supernatural">Supernatural</option>
                <option value="Thriller">Thriller</option>
                <option value="Hentai">Hentai</option>
            </select><br>
            <p></p>
            Minimum number of episodes: <br>
            <input type="text" name="min_episodes"/><br>
            <p></p>
            Maximum number of episodes: <br>
            <input type="text" name="max_episodes"/><br>
            <p></p>
            Minimum number of views show has: <br>
            <input type="text" name="min_popularity" /><br>
            <p></p>
            <input type="submit" />
        </form>
        '''

@app.route('/recommendation', methods=["POST"])
def index():

    user = {'nickname':'Miguel'}
    text = str(request.form['user_name'])
    filter_by_type = str(request.form['Type'])!='Filter by Type'
    TYPE = str(request.form['Type'])
    filter_by_genre = str(request.form['Genre'])!='Filter by Genre'
    GENRE = str(request.form['Genre'])
    min_year = str(request.form['min_year'])
    max_year = str(request.form['max_year'])
    try:
        min_year = float(min_year)
    except:
        min_year = 1
    try:
        max_year = float(max_year)
    except:
        max_year = 3000
    if type(min_year)==float or type(max_year)==float:
        filter_by_year=True
    else:
        filter_by_year=False
    min_popularity = str(request.form['min_popularity'])
    try:
        min_popularity = float(min_popularity)
    except:
        min_popularity = 1
    if type(min_popularity)==float:
        filter_by_popularity=True
    else:
        filter_by_popularity=False
    min_episodes = str(request.form['min_episodes'])
    max_episodes = str(request.form['max_episodes'])
    try:
        min_episodes = float(min_episodes)
    except:
        min_episodes = 0
    try:
        max_episodes = float(max_episodes)
    except:
        max_episodes = 30000
    if type(min_episodes)==float or type(max_episodes)==float:
        filter_by_episodes=True
    else:
        filter_by_episodes=False
    if text == '':
        posts = recommendation_for_non_user(filter_by_type, TYPE, filter_by_genre,
         GENRE, filter_by_year, min_year, max_year, filter_by_episodes, min_episodes,
         max_episodes, filter_by_popularity, min_popularity, series_df)
    else:
        posts = recommendation_for_user(text, filter_by_type, TYPE, filter_by_genre,
         GENRE, filter_by_year, min_year, max_year, filter_by_episodes, min_episodes,
         max_episodes, filter_by_popularity, min_popularity, series_df)
    return render_template('index.html', title='Home', user=user, posts=posts)
