# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 18:51:48 2018

@author: HP
"""

import media
import fresh_tomatoes

toy_story = media.Movie("Toy Story",
                        "A boy and his toys that became to life",
                        "https://upload.wikimedia.org/wikipedia/pt/thumb/d/dc/Movie_poster_toy_story.jpg/250px-Movie_poster_toy_story.jpg",
                        "https://www.youtube.com/watch?v=P9-jf9-c9JM")

avatar = media.Movie("Avatar",
                     "Marines in a alien planet",
                     "https://upload.wikimedia.org/wikipedia/pt/thumb/b/b0/Avatar-Teaser-Poster.jpg/250px-Avatar-Teaser-Poster.jpg",
                     "https://www.youtube.com/watch?v=6ziBFh3V1aM")

despicable_me = media.Movie("Despicable Me",
                     "A vilain that wants to steal the moon",
                     "http://t0.gstatic.com/images?q=tbn:ANd9GcTSrsUAuM5YzZGa9pyYawSj3JtkiMt05Ls10ynFlK4izVgkR7Zy",
                     "https://www.youtube.com/watch?v=2K9bxQhPLRY")


movies =[toy_story, avatar, despicable_me]

fresh_tomatoes.open_movies_page(movies)

print(media.Movie.__doc__)
print(media.Movie.__name__)
print(media.Movie.__module__)