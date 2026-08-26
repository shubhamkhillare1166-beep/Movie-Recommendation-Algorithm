import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("loading movie datasets...")
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

credits = credits.drop(columns=['title'])
df = movies.merge(credits, left_on='id', right_on='movie_id')

print(f"data loaded sucessfully..total movies : {len(df)}")

def get_list(messy_text):
    real_list = ast.literal_eval(messy_text)
    names = []
    for item in real_list:
        names.append(item['name'])
    return names
    
def get_director(messy_text):
    real_list = ast.literal_eval(messy_text)
    for item in real_list:
        if item['job'] == 'Director':
            return item['name']
    return ""
        
def remove_spaces(word_list):
    clean_list = []
    for word in word_list:
        clean_list.append(word.replace(" ","").lower())
    return clean_list     
   
df['genres'] = df['genres'].apply(get_list).apply(remove_spaces)
df['keywords'] = df['keywords'].apply(get_list).apply(remove_spaces)
df['cast'] = df['cast'].apply(get_list).apply(lambda x: x[:3]if len(x) >= 3 else x).apply(remove_spaces)
df['director'] = df['crew'].apply(get_director).apply(lambda x: [x.replace(" ","").lower()]if x != "" else [])

def create_soup(row):
    return ' '.join(row['keywords']) + ' ' + ' '.join(row['cast']) + ' ' + ' '.join(row['director']) + ' ' + ' '.join(row['genres'])

df['soup'] = df.apply(create_soup, axis=1)

# Check our work on the very first movie (Avatar)
print("\n🎉 Movie Soup created!")
print(df['soup'].iloc[0])

print("\n converting text soup into numbers(TF-IDF)")
tfidf = TfidfVectorizer(stop_words='english')
tfidf_metrix = tfidf.fit_transform(df['soup'])
print(f"metrix build sucessfully!  Shape: {tfidf_metrix.shape}")  

cosine_sim = cosine_similarity(tfidf_metrix , tfidf_metrix)
print(f"similarity metrix calculated!shape: {cosine_sim.shape}")

indices = pd.Series(df.index , index = df['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    
    title_lower = title.lower()
    matching_titles = df[df['title'].str.lower() == title_lower]['title']
    
    if matching_titles.empty:
        return f"movie {title} not found in dataset. try another title!"
    
    exact_title = matching_titles.iloc[0]
    idx = indices[exact_title]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

print("\n recommend movies for 'The Dark Knight':")
print(get_recommendations('The Dark Knight'))
