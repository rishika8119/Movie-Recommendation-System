import requests
import streamlit as st
import pickle as pk
import pandas as pd
import os.path


import numpy as np
import pandas as pd
import ast
import nltk
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.metrics.pairwise import cosine_similarity


def make_model():
    movies=pd.read_csv("movies.csv")
    movie_meta=pd.read_csv("cast_crew.csv")

    movies=movies.merge(movie_meta,on="title")

    movies.drop(labels="id",axis=1,inplace=True)

    movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]
    movies.dropna(inplace=True)



    def badaldo(o):
        L=[]
        for i in ast.literal_eval(o):
            L.append(i['name'])
        return L

    movies['genres']=movies['genres'].apply(badaldo)
    movies['keywords']=movies['keywords'].apply(badaldo)

    def firse_badaldo(o):
        L=[]
        counter=0
        for i in ast.literal_eval(o):
            if counter<=3:
                L.append(i['name'])
                counter+=1
            else:
                break
        return L

    movies['cast']=movies['cast'].apply(firse_badaldo)

    def look(o):
        L=[]
        for i in ast.literal_eval(o):
            if i['job']=='Director':
                L.append(i['name'])
                break
        return L

    movies['crew']=movies['crew'].apply(look)
    movies.crew=movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])
    movies['overview']=movies.overview.apply(lambda x:x.split())
    movies.genres=movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
    movies.keywords=movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
    movies.cast=movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
    movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
    final_movies=movies[['movie_id','title','tags']]
    final_movies['tags']=final_movies['tags'].apply(lambda x:" ".join(x))
    final_movies['tags']=final_movies['tags'].apply(lambda x:x.lower())
    cv=CV(max_features=5000,stop_words="english")
    vector=cv.fit_transform(final_movies['tags']).toarray()
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    wn=nltk.WordNetLemmatizer()
    print(wn.lemmatize("accepts"))
    print(wn.lemmatize("accepting"))
    # print(wn.lemmatize("accepting"))
    def stem(text):
        l=[]
        for i in text.split():
            l.append(wn.lemmatize(i))
        
        return " ".join(l)

    final_movies['tags']=final_movies['tags'].apply(stem)

    simalirity=cosine_similarity(vector)
    def recommend(movie):
        movie_index=final_movies[final_movies['title']== movie ].index[0]
        distance=simalirity[movie_index]
        answer=sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])

        for i in answer[1:6]:
            print(final_movies.iloc[i[0]].title)

    final_movies['title'].values
    pk.dump(final_movies.to_dict(),open('movies_dict.pkl','wb'))
    pk.dump(simalirity,open('similarity.pkl','wb'))



if os.path.exists("similarity.pkl")==False:
    make_model()
    print("made model")


# recommend("Batman Begins")







def fetch_details(movie_id):
    response=requests.get('https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US'.format(movie_id))
    data=response.json()
    return "https://image.tmdb.org/t/p/w500/"+data['poster_path']

def recommend(movie):
    movie_index=movies[movies['title']== movie ].index[0]
    distance=simalarity[movie_index]
    answer=sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]


    recommended_movies=[]
    recommended_movies_poster=[]
    for i in answer:
        movie_id=movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        #fetch poster from api
        recommended_movies_poster.append(fetch_details(movie_id))
    return recommended_movies,recommended_movies_poster

movies_dict=pk.load(open('movies_dict.pkl','rb'))
movies=pd.DataFrame(movies_dict)

simalarity=pk.load(open('similarity.pkl','rb'))

st.title("Movie Recommendation system")

selected_movie = st.selectbox(
     'Enter Movie',
     movies['title'].values)

if st.button('Recommend'):
    names,poster=recommend(selected_movie)
    col1, col2, col3, col4 , col5= st.columns(5)
    with col1:
        st.text(names[0])
        st.image(poster[0])
    with col2:
        st.text(names[1])
        st.image(poster[1])
    with col3:
        st.text(names[2])
        st.image(poster[2])
    with col4:
        st.text(names[3])
        st.image(poster[3])
    with col5:
        st.text(names[4])
        st.image(poster[4])
        


        # 8265bd1679663a7ea12ac168da84d2e8