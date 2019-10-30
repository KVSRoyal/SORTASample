# Setup
1) We need the `nltk` module to run this implementation, so we need to `pip install nltk`.
2) It takes a while to load the necessary files for SORTA to run.
There should be some output in red in the console, and then "Ready for Matching"
will be printed. At this point, our custom SORTA is ready to begin matching.

# Example
These are the outputs and scores produced by Molgenis' SORTA for the following input:
1) "congenital nose anomalies" matches to "Congenital hip dislocation", 79.07%
2) "other specified congenital malformations of skull and face bone" matches to 
"Congenital malformation of the great arteries", 50.70%

In *Example.py* we run our custom SORTA algorithm with the same inputs.
