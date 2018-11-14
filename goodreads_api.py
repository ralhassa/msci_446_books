# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:48:10 2018

@author: hilla
"""

import requests
import sys
import xmltodict
import rauth
import constant
from xml.etree import ElementTree

apiKey = "k1Y1icnodolM9lzNbXM8Yg"
apiSecret =  "7dS2oLnoNQEpZpKQgvBLDksOLUVDAIEzIFdZ257U" 
#from goodreads import client
client = GoodreadsClient.Create(apiKey, apiSecret)#gc.authenticate(<access_token>, <access_token_secret>)
book = await client.Books.GetByBookId(bookId: 15979976)
groups = await client.Groups.GetGroups(search: "Arts") 
print(book)