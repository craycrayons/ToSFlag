# Lap 2: did legal-bert close the semantic gap?

Lap 1's TF-IDF model missed 53 **real** unfair clauses at its recall-first operating point (non-clause header stubs are excluded as dataset label noise, matching `run.py`'s recall discipline). The Lap 1 error analysis argued these residual misses were SEMANTIC - waiver/responsibility clauses unfair by legal meaning, not vocabulary. If that diagnosis was right, a legal-domain encoder should recover them.

**Result: legal-bert recovers 17 of 53 real missed clauses (32%).**

## Clauses recovered (TF-IDF missed, legal-bert caught)

- (`potentially_unfair`) If you are using our Service for an organization, you are agreeing to these Terms on behalf of that organization.
- (`clearly_unfair`) You acknowledge that we do not pre-screen Content, but that we shall have the right (but not the obligation) to refuse, move or delete any Content that is available via the Service.
- (`potentially_unfair`) In no event shall Weebly be liable to you or any third party for your use or alleged use of any Third Party Materials.
- (`potentially_unfair`) Should we, due to failure on the part of you, the account holder, or the new domain name service provider, be unable to make the domain transfer to your new domain name service provider, we are expressly entitled to have the cancelled domain name deleted by the relevant domain na
- (`potentially_unfair`) Your use of the Email Marketing Service serves as your consent to these terms.
- (`clearly_unfair`) If your account is terminated, we may permanently delete it and any associated data.
- (`potentially_unfair`) If you continue to use the Services after the changes are posted, you are agreeing that the changes apply to your continued use of the Services.
- (`potentially_unfair`) We also are not responsible for information, materials, products, or services provided by other players (for instance, in their profiles) and User Content is not approved by us.
- (`potentially_unfair`) Zynga may revise the pricing for the goods and services it licenses to you through the Services at any time.
- (`clearly_unfair`) For more specific information, please read carefully the Terms below.
- (`potentially_unfair`) Nevertheless, the English version governs your relationship with Yelp, and any inconsistencies among the different versions will be resolved in favor of the English version.
- (`potentially_unfair`) The restrictions above only apply to the extent permissible under applicable law.
- (`clearly_unfair`) You represent that you have read and understood them.
- (`clearly_unfair`) These additional rules and terms apply in addition to these Terms and are important.
- (`clearly_unfair`) They will apply in addition to these Terms.
- (`clearly_unfair`) This applies to all claims under any legal theory, unless the claim fits within the Exceptions to Agreement to Arbitrate identified below.
- (`clearly_unfair`) If the translated version means something different from the English version, then the English meaning will be the one that applies.

## Still missed by both (the residual hard cases)

- (`potentially_unfair`) Our Privacy Notice (available at: https://www.weebly.com/privacy), which is part of these Terms, describes how we collect, protect, and use your Registration Data and certain other information about you.
- (`potentially_unfair`) Weebly does not endorse or assume responsibility for any Third Party Materials and makes no guarantee regarding the reliability, accuracy, nature, origin, quality, or use of such Third Party Materials.
- (`potentially_unfair`) Weebly is under no obligation to notify you of any changes to the SendGrid Terms.
- (`potentially_unfair`) If you are a consumer bringing a claim relating to a transaction intended for a personal, household, or family use, any arbitration hearing will occur within the county where you reside.
- (`clearly_unfair`) Otherwise, any arbitration hearing will occur in San Francisco, California, or another mutually agreeable location.
- (`potentially_unfair`) Either party may bring a lawsuit solely for injunctive relief to stop unauthorized use or abuse of the Service, or to enforce intellectual property rights (e.g., copyright, trademark, trade secret, or patent rights) without first engaging in our informal dispute resolution proces
- (`potentially_unfair`) If that happens, unless applicable law requires otherwise, Zynga is not required to provide refunds, benefits, or other compensation to you in connection with discontinued elements of the Services or for Virtual Items previously earned or purchased.
- (`potentially_unfair`) ALL SALES ARE FINAL: YOU ACKNOWLEDGE THAT ZYNGA IS NOT REQUIRED TO PROVIDE A REFUND FOR ANY REASON, AND THAT YOU WILL NOT RECEIVE MONEY OR OTHER COMPENSATION FOR UNUSED VIRTUAL ITEMS WHEN AN ACCOUNT IS CLOSED, WHETHER SUCH CLOSURE WAS VOLUNTARY OR INVOLUNTARY, OR WHETHER YOU MADE
- (`clearly_unfair`) ●      Phishing:  a site meant to trick users into providing their username and password
- (`clearly_unfair`) These Terms will also apply when you use the Service on a trial basis.
- (`clearly_unfair`) The Site and Service are owned by Weebly.
- (`potentially_unfair`) infringes any patent, trademark, trade secret, copyright, rights of privacy or publicity, or other proprietary rights of any party (e.g., music, movies, images, e-books, or games you do not own the rights to);
- (`potentially_unfair`) is adult in nature, such as any nudity in a sexual context or any Content with adult themes or reveals exposed genitalia;
- (`potentially_unfair`) ●      are manufactured as, or primarily intended to be used as, weapons, including firearms, restricted devices, or ammunition.
- (`potentially_unfair`) Examples of applicable laws include laws relating to spam or unsolicited commercial email (hereinafter “Spam” or “UCE”), privacy, security, obscenity, defamation, intellectual property, pornography, terrorism, homeland security, gambling, child protection, and other applicable la