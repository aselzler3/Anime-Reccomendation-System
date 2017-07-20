# Anime-Recommendation-System

This recommendation system suggests anime for people to watch who have a profile on Anilist.co. It is available to use now on www.mynextanime.com.

The data used to make this project was collected from the API documentation on Anilist.co. Since the data used is somewhat confidential, I will not be including the full dataset or the code used to collect the data here.

I used PySparks Alternating Least Squares matrix factorization model to fit the dataset. I then wrote the second matrix "H" as a csv file and sent it to another file as the fitted model to make predictions. This code is available in src/make_spark_V.ipynb.

The csv file "data/spark_V.csv" along with metadata about the shows themselves were then used to make a flask app "app/views.py". This flask app follows a very simple html template included in app/templates. 
