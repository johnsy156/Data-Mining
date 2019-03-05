# Name: Mallavarapu Johnsy Vineela
# Script title: Cosine Similarity calculation on Debate dataset 
# References: 
# https://medium.freecodecamp.org/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3
# https://www.accelebrate.com/blog/using-defaultdict-python/
# https://towardsdatascience.com/tfidf-for-piece-of-text-in-python-43feccaa74f8
# https://nlpforhackers.io/tf-idf/
# http://billchambers.me/tutorials/2014/12/22/cosine-similarity-explained-in-python.html


#All the essential imports for the program are included here
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import math
from collections import defaultdict


# Let us return the term frequency of the terms using the formula calculation		
def get_term_frequency(paratokens):
	for docs in paratokens:
		doc_id = docs['doc_id']
		terms = docs['tfdict']
		# We calculate the term frequency and update the dict to the list
		for term in terms:
			tf_value = 1 + math.log10(terms[term])
			tfscore = {'doc_id': doc_id, 'term': term, 'term_freq': tf_value}
			tfscore_list.append(tfscore)

	return tfscore_list
			
# Let us calculate inverse document frequency
# We need N and document frequency value
def inverse_doc_freq(paratokens):
	#First we need to calculate the document freq of the terms 
	for docs in paratokens_in_doc:
		doc_id = docs['doc_id']
		terms = docs['tfdict']
		# the document frequency of the terms are calculated
		for term in terms:
			doc_freq[term] += 1

	# Based on the document frequency, we calculate the inverse document frequency below
	for df_term in doc_freq:
		inverse_dfreq[df_term] = math.log10(dcount / float(doc_freq[df_term]))

	# We generate a list of dicts with both the tf and idf weights for each term
	for docs in paratokens_in_doc:
		doc_id = docs['doc_id']
		terms = docs['tfdict']
		for term in terms:
			temp4dict = {'docid': doc_id, 'term': term, 'df': doc_freq[term], 'idf': inverse_dfreq[term]}
			idf_weights_list.append(temp4dict)

	return idf_weights_list
			
# We calculate the tf-idf weights for all the terms in this function
def tf_idf_weight(tfscore_list,idf_weights_list):
	for tf_doc in tfscore_list:
		for idf_doc in idf_weights_list:
			if tf_doc['doc_id'] == idf_doc['docid'] and tf_doc['term'] == idf_doc['term']:
				tfidf_dict = {'doc_id': tf_doc['doc_id'], 'term': tf_doc['term'], 'tf_idf': tf_doc['term_freq'] * idf_doc['idf']}
				tf_idf_list.append(tfidf_dict)
	
	m = len(tf_idf_list)

	# To calculate the total sum vector, we sum the squares of the vectors and append the square root of the value to list
	for d_id in range(1,58):
		vec_den = 0
		for doc in tf_idf_list:
			if d_id == doc['doc_id']:
				vec_den += (doc['tf_idf']**2)
		root_vec_den = math.sqrt(vec_den)
		root_vec_list.append(root_vec_den)

	n = len(root_vec_list)
	
	# We update the tf_idf_list with the normalized values for the tf_idf weights 
	for i in range(0,n):
		for j in range(0,m):
			if i+1 == tf_idf_list[j]['doc_id']:
				tf_idf_list[j]['tf_idf'] = tf_idf_list[j]['tf_idf'] / root_vec_list[i]

	return tf_idf_list

# Based on the calculated values, this function provides the output for the provided parameter
# If the token is not found then we return -1
def getidf(token):
	for term in inverse_dfreq:
		if term == token:
			return inverse_dfreq[term]
	return -1

#This function can be used to calculate the query vector for the given string
def getqvec(qstring):
	tdict = {}
	qstring_processed = []
	dcnt = 1

	# Convert the provided query to lower string characters
	qstring = qstring.lower()

	#Tokenize the provided string based on the provided regular expression to allow only alphabets
	tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
	tokens = tokenizer.tokenize(qstring)

	# Check if the token is in the stopwords english file and proceed if not
	for token in tokens:
	 	if token not in stopwords.words('english'):
	 		stemmed_token = stemmer.stem(token)

	 		# Let us sum the occurrences of the term and save it 
	 		# This will help us in calculating the term frequency later
	 		if stemmed_token not in tdict:
	 			tdict[stemmed_token] = 1
	 		else:
	 			tdict[stemmed_token] += 1
	 		temp1dict = {'doc_id': dcnt,'tfdict': tdict}

	# Process the strings and append the dicts to the list
	qstring_processed.append(temp1dict)

	#Compute Term Frequency (TF) of the terms in the query string 
	qstring_tf = []
	tfscore = {}
	tf_value = 0
	for word in qstring_processed[0]['tfdict']:
		tf_value = 1 + math.log10(qstring_processed[0]['tfdict'][word])
		tfscore = {'doc_id': dcnt, 'term': word, 'term_freq': tf_value}
		qstring_tf.append(tfscore)
	
	#Compute IDF for the given query string	
	idf_weights_qstring = []
	qstring_idf = []
	true_value = 0
	for word in qstring_processed[0]['tfdict']:
		idf_dict = {}
		for doc in idf_weights_list:
			if word in doc['term']:
				# Setting the flag value to 1 meaning that the token is present
				true_value = 1  
				idf_dict = {'doc_id': doc['docid'], 'term': word, 'idf': doc['idf']}
				qstring_idf.append(idf_dict)
				# We break after the first occurence of the word and proceed to the next word in string
				break			

		# We set this flag so that we can avoid having to visit all the tokens in the list
		if true_value == 0:
			# As per description, we set the document frequency to 1 for the tokens that are not present
			df = 1.0
			idf_dict2 = {'doc_id': dcnt, 'term': word, 'idf': math.log10(57/df)}
			qstring_idf.append(idf_dict2)  
		
	#Calculating TF-IDF weights of the query
	dict_tfidf = {}
	query_tf_idf = []
	# We take both the tf and idf lists and multiple the tf and idf weights to generate a new list
	for d1 in qstring_tf:
		for d2 in qstring_idf:
			if d1['term'] == d2['term']:
				dict_tfidf = {'doc_id': d2['doc_id'], 'term': d2['term'], 'tfidf': d1['term_freq']*d2['idf']}
				query_tf_idf.append(dict_tfidf)

	listlen = len(query_tf_idf)

	#Calculating normalized TF-IDF query vector values
	true_value = 0
	norm_dict = {}
	normquery_tf_idf = []
	vector_sum = 0

	# We sum the squares of the tfidf values of all tokens and then finally take a square root to get the total vector
	for record in query_tf_idf:
		vector_sum += (record['tfidf']**2)
	vector_sum = math.sqrt(vector_sum)
	final_result = {}

	# This loop is to normalize the tf-idf values and return the output to the function in the desired format
	for i in range(0,listlen):
		query_tf_idf[i]['tfidf'] = query_tf_idf[i]['tfidf']/vector_sum
		final_result.update({query_tf_idf[i]['term']:query_tf_idf[i]['tfidf']})

	return final_result

# We calculate the cosine similarity of the query with the document
def query(qstring):
	vector_query = getqvec(qstring)
	document = ""
	max_cosine_value = 0

	# In this loop, we sum the values of the normalized vectors
	# the value is then used to compare the maximum cosine value
	for docid in range(1,58):
		value = 0
		for vec in vector_query:
			for record in tf_idf_list:
				if vec == record['term'] and docid == record['doc_id']:
					value += (vector_query[vec]*record['tf_idf'])

		# We check if the value is greater than the one stored in max_cosine_value and update accordingly
		if value > max_cosine_value:
			max_cosine_value = value
			# We find the doc_id corresponding to the document that has the max cosine value 
			for docs in paragraphs_in_doc:
				if docid == docs['doc_cnt']:
					document = docs['document'] + "\n"

	# We return the output of the function as per the desired format
	if max_cosine_value == 0:
		document = "No match" + "\n"
		return(document, max_cosine_value)
	else:
		return(document, max_cosine_value)


#Declaring global variables for accessing the calculated lists and dicts throughout the code
paragraphs_in_doc = []
paratokens_in_doc = []
dcount = 0
stemmer = PorterStemmer()
tfscore_list = []
tf_idf_list = []
inverse_dfreq = dict()
doc_freq = defaultdict(int)
idf_weights_list = []
root_vec_list = []


#Read the input file into a variable
filename = './debate.txt'
file = open(filename, "r", encoding='UTF-8')
doc = file.readlines()

#We process every line in the doc variable
for line in doc:
	line = line.lower()
	line = line.rstrip("\n")
	if line != "":      # We do not consider empty lines to be added to the list
	 	dcount += 1 	# The document count is stored in dcount
	 	tempdict = {'doc_cnt': dcount, 'document': line}   #Setting docid and document as key value pairs for ease of access later
	 	paragraphs_in_doc.append(tempdict)	# Append the doc id and the document into a list 

# Now we process every document in the document collection
idv = 0
for docs in paragraphs_in_doc:
	idv = idv + 1
	tdict = {}
	#Tokenize the words in the particular document
	tokenizer = RegexpTokenizer(r'[a-zA-Z]+')		
	tokens = tokenizer.tokenize(docs['document'])
	for token in tokens:
		#Remove stop words from the tokens and then stem it 
		if token not in stopwords.words('english'):
			stemmed_token = stemmer.stem(token)
			# Once the text is processed, we calculate the frequency of the tokens in the document 
			if stemmed_token not in tdict:
				tdict[stemmed_token] = 1
			else:
				tdict[stemmed_token] += 1
			temp1dict = {'doc_id': idv, 'tfdict': tdict}
	paratokens_in_doc.append(temp1dict)

#Calculate TF
tf = get_term_frequency(paratokens_in_doc)
#Calculate IDF
idf = inverse_doc_freq(paratokens_in_doc)
#Normalized weights are available in the tf_idf_list
tf_idf = tf_idf_weight(tf,idf) 


