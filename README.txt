TF-IDF and Cosine Similarity using NLTK

- In this application, we do data preprocessing on the provided file that contains the transcript of the latest Texas Senate race debate between Ted Cruz
and Beto O'Rourke and then calculate the TF-IDF vector for each paragraph. 
- Then, given a query string, we calculate the query vector and compute the cosine similarity between the query vector and the paragraphs in the transcript


Note:
For calculating inverse document frequency, we treat debate.txt as the whole corpus and the paragraphs as documents


How to run the code?
Put statements similar to the following in the testfile.py file and execute it like follows:

python testfile.py

For IDF:  
print("%.4f" % getidf(stemmer.stem("immigration")))


For query vector: 
print(getqvec("The alternative, as cruz has proposed, is to deport 11 million people from this country"))


For cosine similarity:
print("%s%.4f" % query("clinton first amendment kavanagh"))
