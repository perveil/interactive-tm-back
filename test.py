from sklearn.datasets import fetch_20newsgroups


docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

print(docs)