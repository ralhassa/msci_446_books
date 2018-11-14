# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:24:03 2018

@author: hilla
"""

import csv
import requests 
from bs4 import BeautifulSoup
for i in range(907):      # Number of pages plus one 
    url = "https://www.goodreads.com/list/show/429.The_BOOK_was_BETTER_than_the_MOVIE?page=2".format(i)
    url = "http://www.pga.com/golf-courses/search?page={}&searchbox=Course+Name&searchbox_zip=ZIP&distance=50&price_range=0&course_type=both&has_events=0".format(i)
    r = requests.get(url)
    soup = BeautifulSoup(r.content)
