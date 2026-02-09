# svsal-bertopic

Experiment for analysis of the
[School of Salamanca](https://www.salamanca.school) corpus with BERTopic.

Developed together by @jcarrill718 and @awagner-mainz, the idea was to do
dynamic topic modelling over the corpus. Unfortunately, we got stuck when
evaluating the topics BERTopic suggested.

Since the beginning in 2024, the creation of text resources from the
Salamanca corpus has been substantially reworked, and new opportunities
for working with embeddings have appeared. Beginning in May 2025, the
notebooks have thus been re-organized by @awagner-mainz. Different
tasks -- embeddings creation, upload to a vector Db, dimensionality
reduction/projection, analysis (Topic Modeling) -- have been separated
and emphasis was put on creation of embeddings and uploading them to
a vector Db that has been developed alongside
([EmbAPI](https://github.com/mpilhlt/embapi)).

A 3D projection was computed and the School of Salamanca website has
been enhanced with similarity search/RAG and visualization features
(beta mode). For now, documentation is currently still pending, but
you can get a glimpse on this (German-language)
[Blogpost](https://dhc.hypotheses.org/3374) by @tdiepenb and Tom
Hunze.

Currently (Feb 2026) new embeddings have been created for an expanded
set of texts (new texts have been released in the Salamanca project since
2024...). They will be uploaded to the vector Db next and the spatial
projection(s) for the Salamanca website will be created. Hopefully, it
will be possible also to return to the analysis part soon (TM). If that
is of interest to you, you may want to consider following the repository.

## License
