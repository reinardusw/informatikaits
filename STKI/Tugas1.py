import sys
from bs4 import BeautifulSoup as bs4

from bs4 import BeautifulSoup
infile = open("data.xml","r")
contents = infile.read()
soup = BeautifulSoup(contents,'xml')
titles = soup.find_all('title')
for title in titles:
    print(title.get_text())
