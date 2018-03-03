from bs4 import BeautifulSoup
import requests

www = "https://www.forexfactory.com"
page = requests.get(www)
soup = BeautifulSoup(page.content, "lxml")
body = []

b = soup.findAll("tr", {"class": "calendar__row calendar_row calendar__row--new-day newday"})
m = soup.findAll("tr", {"class": "calendar__row calendar_row"})
e = soup.findAll("tr", {"class": "calendar__row calendar_row calendar__row--no-grid nogrid"})

# WORK IN PROGRESS