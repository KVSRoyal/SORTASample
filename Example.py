# An example of our custom sorta implementation
from SORTA import SORTA

# Make a SORTA object for matching
sorta = SORTA()

# We will only print the top match
match = sorta.get_matches('congenital nose anomalies')[0]
print('"congenital nose anomalies" matched to "{}", with code HP:{} and estimated accuracy of {}%'.format(*match))

# This one is a compound term that is difficult to match either way. Our SORTA is able to produce fairly good matches
# for this term, while Molgenis' SORTA cannot. While the aim of this program is not necessarily to deal with
# compound input such as this, this is a good example of where our custom SORTA would succeed where Molgenis' SORTA
# cannot.
match = sorta.get_matches('other specified congenital malformations of skull and face bone')[0]
print('"other specified congenital malformations of skull and face bone" matched to "{}", with code HP:{} and estimated'
      ' accuracy of {}%'.format(*match))

match = sorta.get_matches('other specified congenital malformations of skull and face bone')[1]
print('"other specified congenital malformations of skull and face bone" matched to "{}", with code HP:{} and estimated'
      ' accuracy of {}%'.format(*match))
