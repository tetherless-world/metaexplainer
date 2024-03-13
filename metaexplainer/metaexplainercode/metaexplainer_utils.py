from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def find_cosine_similarity(s1, s2):
	'''
	return score for sklearn's cosine similarity
	'''
	comparisons = (s1, s2)
	#print(comparisons)

	tfidf_vectorizer = TfidfVectorizer()
	tfidf_matrix = tfidf_vectorizer.fit_transform(comparisons)

	result_cos = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
	return result_cos[0][1]

def print_list(list_n):
	'''
	Print contents of list on separate lines
	'''
	for val in list_n:
		print(val)