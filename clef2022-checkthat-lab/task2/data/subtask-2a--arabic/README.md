# Task 2a: Detecting Previously Fact-Checked Claims in Arabic
v1.1 is an updated version of the training data with the following updates:
* tweet-ar-291 was excluded from the training queries as it is a duplicate of tweet-ar-273
* the training qrels were updated to exclude tweet-ar-291 and to add another relevant claim to tweet-ar-273 which is vclaim-0559

v1.2 updated the verified claims to JSON format

v1.3 major updates were done to the qrels in addition to some updates to the queries as follows:
* the number of qrels increased from 649 to 696 as we found other claims relevant to the same queries within the verified claims collection.
* tweet-ar-286 and tweet-are-743 were excluded from the training queries as they are duplicates to tweet-ar-729 and tweet-ar-287 respectively.

v1.4 some updates were done to the qrels as follows:
* 9 new qrels were added and excluded one. We have 704 qrels now.

v1.5 we divided the training data shared in the previous version into training and development sets as follows:
* 512 training queries with 602 qrels.
* 85 development queries with 102 qrels.