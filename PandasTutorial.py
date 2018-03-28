import pandas as pd

reviews = pd.read_csv("C:/Users/aszewczyk/Documents/Scripts/Python/real_world_machine_learning/ign.csv")

reviews.head(4)

reviews.shape
type(reviews)

#indexing
reviews.iloc[0:5,:]
reviews.iloc[:5,2:]

#will show all index
reviews.index

some_reviews = reviews.iloc[20:30,:]
#index form 20 to 30 only!!
some_reviews.index


#indexing using labels
reviews.loc[0:5,["score", "release_year"]]

reviews.loc[:,["score", "release_year"]]
#or use lists of columns
reviews[["score", "release_year"]]
#creates pandas series
reviews["score"]
#type: pandas d.f.
type(reviews.loc[0:5,["score", "release_year"]])
#type pandas series (vecotr-like)
type(reviews["score"])

#column names and types
reviews.dtypes
reviews.columns

#create series and data frames

s1 = pd.Series([1,2,3])
s2 = pd.Series(["Pimpek", "Pisior", "Kociamber"])

pd.DataFrame([s1,s2])

#elegent and with column and row names specified
pd.DataFrame(
    [
        [1,2,3],
        ["Pimpek", "Pisior", "Kociamber"]
    ],
    columns=["jeden", "dwa", "trzy"],
    index=['index_galgana', "imie_galgana"]
)

#or using Pzton dictionarz we can specifzy column name implicitely
pd.DataFrame(
    {
        "jeden": [1,"Pimpek"],
        "dwa": [2, "Pisior"]
    }
)

#each column is a series object, we can call pd methods on them as well
reviews["title"].head()

reviews['score'].mean()

#if we specify axis will compute mean of each row
reviews.mean(axis=1)


reviews["score"].min()
reviews["score"].max()
#count each column separate
reviews.count()
#finds correlation between columns
reviews.corr()
reviews["score"].median()
reviews["score"].std()

#filter on score larger than 7
score_high = reviews[(reviews["score"] > 7) & (reviews["platform"] == "Xbox One")]

score_high = reviews[(reviews["score"] > 7) & (reviews["platform"] == "PC")]

grouped_reviews = reviews.groupby(["platform", "release_year"])["score"].mean()

score_c64_genre = reviews[reviews["platform"] == "Commodore 64/128"].groupby("genre")["score"].mean()

score_c64 =  reviews[reviews["platform"] == "Commodore 64/128"].sort_values("score", ascending=False)[["title", "score"]]