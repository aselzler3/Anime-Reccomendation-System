# Anime-Recommendation-System

This recommendation system suggests anime for people to watch who have a profile on Anilist.co. It is available to use now on www.mynextanime.com.

The data used to make this project was collected from the API documentation on Anilist.co. This data consists of around 100000 users. Each user has a list of shows they have seen along with the rating they give that particular show. Since each user has rated aroung 60 shows on average, there were about 6 million ratings given in total. Since the data used is somewhat confidential, I will not be including the full dataset or the code used to collect the data here.

Because of the large number of ratings in this dataset, I used PySparks Alternating Least Squares matrix factorization model to fit the dataset. I then wrote the second matrix "H" as a csv file (data/spark_V.csv) and sent it to another file as the fitted model to make predictions. This code is available in src/make_spark_V.ipynb.

The csv file "data/spark_V.csv" along with metadata about the shows themselves were then used to make a flask app "app/views.py". This flask app follows a very simple html template included in app/templates. 
