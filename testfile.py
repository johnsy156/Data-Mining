from tfidf_calc import *


# print("%.4f" % getidf(stemmer.stem("immigration")))
# print("%.4f" % getidf(stemmer.stem("abortion")))
# print("%.4f" % getidf(stemmer.stem("hispanic")))
# print("%.4f" % getidf(stemmer.stem("the")))
print("%.4f" % getidf(stemmer.stem("tax")))
# print("%.4f" % getidf(stemmer.stem("oil")))
# print("%.4f" % getidf(stemmer.stem("beer")))

# print(getqvec("The alternative, as cruz has proposed, is to deport 11 million people from this country"))
print(getqvec("unlike any other time, it is under attack"))
# print(getqvec("vector entropy"))
# print(getqvec("clinton first amendment kavanagh"))

# print("%s%.4f" % query("The alternative, as cruz has proposed, is to deport 11 million people from this country"))
print("%s%.4f" % query("unlike any other time, it is under attack"))
# print("%s%.4f" % query("vector entropy"))
# print("%s%.4f" % query("clinton first amendment kavanagh"))

