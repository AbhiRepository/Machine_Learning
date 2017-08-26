import requests
import urllib
import re

urls = 'google yahoo CNN tesla apple'.split()
pat = re.compile(r'<title>+.*</title>+')

for url in urls:
	r = requests.get('http://'+url+'.com')
	pageText = r.text
	titles = re.findall(pat,str(pageText))
	print(titles[0])





