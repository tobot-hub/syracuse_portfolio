import json
import pandas as pd

with open("movies.json") as f:
	movies_json = json.load(f)

movie_cols = ["imdbID","Title","Year","Rated",
	"Plot","Metascore","imdbVotes","imdbRating"]

movie_df = pd.DataFrame(columns=movie_cols)

for m in movies_json:
	m_list = []
	for col in movie_cols:
		try:
			m_list.append(movies_json[m][col])
		except:
			m_list.append("")
	df_len = len(movie_df)
	movie_df.loc[df_len] = m_list

movie_df.to_csv("movie.csv", index=False)

genre_df = pd.DataFrame(columns=['imdbID','Genre'])

for m in movies_json:
	for g in movies_json[m]["Genre"].split(','):
		g_list = [movies_json[m]["imdbID"],g.strip()]
		df_len = len(genre_df)
		genre_df.loc[df_len] = g_list

genre_df.to_csv("genre.csv", index=False)

director_df = pd.DataFrame(columns=['imdbID','Director'])

for m in movies_json:
	for d in movies_json[m]["Director"].split(','):
		d_list = [movies_json[m]["imdbID"],d.strip()]
		df_len = len(director_df)
		director_df.loc[df_len] = d_list
director_df.to_csv("director.csv", index=False)

writer_df = pd.DataFrame(columns=['imdbID','Writer'])
for m in movies_json:
	for w in movies_json[m]["Writer"].split(','):
		w_list = [movies_json[m]["imdbID"],w.split('(')[0].strip()]
		df_len = len(writer_df)
		writer_df.loc[df_len] = w_list

writer_df.to_csv("writer.csv", index=False)

actor_df = pd.DataFrame(columns=['imdbID','Actor'])

for m in movies_json:
	for a in movies_json[m]["Actors"].split(','):
		a_list = [movies_json[m]["imdbID"],a.strip()]
		df_len = len(actor_df)
		actor_df.loc[df_len] = a_list

actor_df.to_csv("actor.csv", index=False)

production_df = pd.DataFrame(columns=['imdbID','Production'])
for m in movies_json:
	try:
		for w in movies_json[m]["Production"].split(','):
			w_list = [movies_json[m]["imdbID"],w.split('(')[0].strip()]
			df_len = len(production_df)
			production_df.loc[df_len] = w_list
	except:
		pass

production_df.to_csv("production.csv", index=False)
