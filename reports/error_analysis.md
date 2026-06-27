# Error analysis: what ToSFlag misses

At the recall-first threshold (0.337), the model still misses 43 unfair clauses on the test set. These are the false negatives - the failures that matter, since a missed unfair clause is the costly error.

Lowest-scoring misses (the model was most confident these were fair):

- (score 0.085, `potentially_unfair`) infringes any patent, trademark, trade secret, copyright, rights of privacy or publicity, or other proprietary rights of any party (e.g., music, movies, images, e-books, or games you do not own the rights to);
- (score 0.121, `clearly_unfair`) You will not:
- (score 0.121, `clearly_unfair`) You will not:
- (score 0.148, `potentially_unfair`) Either party may bring a lawsuit solely for injunctive relief to stop unauthorized use or abuse of the Service, or to enforce intellectual property rights (e.g., copyright, trademark, trade secret, or patent rights) without first engaging in our informal dispute resolution process or arbitration.
- (score 0.162, `potentially_unfair`) If that happens, unless applicable law requires otherwise, Zynga is not required to provide refunds, benefits, or other compensation to you in connection with discontinued elements of the Services or for Virtual Items previously earned or purchased.
- (score 0.166, `clearly_unfair`) You can opt-out of non-essential communications here.
- (score 0.171, `potentially_unfair`) The failure of Weebly to exercise or enforce any right or provision of these Terms shall not constitute a waiver of such right or provision.
- (score 0.171, `clearly_unfair`) You are responsible for your interactions with other players.
- (score 0.176, `clearly_unfair`) You alone are responsible for Your Content, and once published, it cannot always be withdrawn.
- (score 0.178, `clearly_unfair`) “Community Rules” means the rules of conduct that govern your interaction with our Services and other players, which can be found here.
- (score 0.204, `potentially_unfair`) Weebly does not endorse or assume responsibility for any Third Party Materials and makes no guarantee regarding the reliability, accuracy, nature, origin, quality, or use of such Third Party Materials.
- (score 0.209, `potentially_unfair`) The restrictions above only apply to the extent permissible under applicable law.
- (score 0.210, `clearly_unfair`) Violations
- (score 0.211, `potentially_unfair`) Any failure on Yelp's part to exercise or enforce any right or provision of the Terms does not constitute a waiver of such right or provision.
- (score 0.225, `potentially_unfair`) If you are a consumer bringing a claim relating to a transaction intended for a personal, household, or family use, any arbitration hearing will occur within the county where you reside.