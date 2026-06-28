# Error analysis: what ToSFlag misses

At the recall-first threshold (0.456), the model still misses 53 unfair clauses on the test set. These are the false negatives - the failures that matter, since a missed unfair clause is the costly error.

Lowest-scoring misses (the model was most confident these were fair):

- (score 0.083, `potentially_unfair`) infringes any patent, trademark, trade secret, copyright, rights of privacy or publicity, or other proprietary rights of any party (e.g., music, movies, images, e-books, or games you do not own the rights to);
- (score 0.129, `clearly_unfair`) You alone are responsible for Your Content, and once published, it cannot always be withdrawn.
- (score 0.131, `clearly_unfair`) You can opt-out of non-essential communications here.
- (score 0.169, `potentially_unfair`) The failure of Weebly to exercise or enforce any right or provision of these Terms shall not constitute a waiver of such right or provision.
- (score 0.175, `clearly_unfair`) For more information, please go to Zynga’s Copyright Page to review our Notification Guidelines.
- (score 0.181, `potentially_unfair`) If that happens, unless applicable law requires otherwise, Zynga is not required to provide refunds, benefits, or other compensation to you in connection with discontinued elements of the Services or for Virtual Items previously earned or purchased.
- (score 0.182, `potentially_unfair`) Either party may bring a lawsuit solely for injunctive relief to stop unauthorized use or abuse of the Service, or to enforce intellectual property rights (e.g., copyright, trademark, trade secret, or patent rights) without first engaging in our informal dispute resolution process or arbitration.
- (score 0.184, `clearly_unfair`) You are responsible for your interactions with other players.
- (score 0.209, `clearly_unfair`) You represent that you have read and understood them.
- (score 0.211, `potentially_unfair`) The Terms, and any rights or obligations hereunder, are not assignable, transferable or sublicensable by you except with Yelp's prior written consent, but may be assigned or transferred by us without restriction.
- (score 0.215, `clearly_unfair`) The parties agree that the arbitrator’s decision or award in one person’s case can only impact the person who brought the claim, not other Zynga players, and cannot be used to decide other disputes with other players.
- (score 0.216, `clearly_unfair`) “Community Rules” means the rules of conduct that govern your interaction with our Services and other players, which can be found here.
- (score 0.223, `potentially_unfair`) The restrictions above only apply to the extent permissible under applicable law.
- (score 0.223, `potentially_unfair`) Any failure on Yelp's part to exercise or enforce any right or provision of the Terms does not constitute a waiver of such right or provision.
- (score 0.225, `potentially_unfair`) Use of the materials by the U.S. Government constitutes acknowledgment of our proprietary rights in them.