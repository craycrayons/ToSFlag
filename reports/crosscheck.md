# LexGLUE cross-check

Trained on the community dataset (CodeHima/TOS_Dataset), evaluated on the peer-reviewed LexGLUE `unfair_tos` test split as fully held-out data. The model never saw LexGLUE during training or threshold selection. Cost rule: FN:FP = 5.0.

LexGLUE test set: 1607 clauses, 172 unfair (10.7%), 1435 fair.

## Generalisation

Two operating points. `community_threshold` applies the Lap-1 recall-first cut (0.518) as-is. `lexglue_tuned` re-solves the same cost rule on LexGLUE (0.537); the gap between them is threshold-transfer loss, separate from genuine ranking failure.

(Note: this script re-derives the community threshold in-sample on the pooled cleaned community data, so it reads slightly higher than the 0.456 in `operating_point.csv`, which `run.py` derives on the held-out validation split. Same cost rule, different slice; both are correct for their own report.)

| operating_point     |   threshold |   recall |   precision |    f1 |   flagged |
|:--------------------|------------:|---------:|------------:|------:|----------:|
| community_threshold |       0.518 |    0.895 |       0.497 | 0.639 |     0.193 |
| lexglue_tuned       |       0.537 |    0.884 |       0.514 | 0.65  |     0.184 |

## Recall by expert unfairness category (community threshold)

Caught vs missed within each LexGLUE expert type. A type with high miss count is a kind of unfairness the model is structurally blind to.

| category                |   total |   caught |   missed |   recall |
|:------------------------|--------:|---------:|---------:|---------:|
| Unilateral change       |      38 |       35 |        3 |    0.921 |
| Unilateral termination  |      38 |       33 |        5 |    0.868 |
| Limitation of liability |      38 |       32 |        6 |    0.842 |
| Contract by using       |      23 |       20 |        3 |    0.87  |
| Jurisdiction            |      16 |       16 |        0 |    1     |
| Content removal         |      13 |       13 |        0 |    1     |
| Choice of law           |      13 |       13 |        0 |    1     |
| Arbitration             |       7 |        6 |        1 |    0.857 |

## Experts say UNFAIR, model says FAIR (18)

The model passed these; LexGLUE experts flagged them. If many are clearly unfair, the community set under-labels these patterns.

- (0.188, _Unilateral change_) we will try to notify you of material revisions , for example via a service notification or an email to the email associated with your account . 

- (0.262, _Limitation of liability_) to the extent allowed under law , linkedin and its affiliates ( and those that linkedin works with to provide the services ) ( a ) disclaim all implied warranties and representations ( e.g. warranties of merchantability , fitness for a part
- (0.280, _Unilateral change_) we have the right to limit how you connect and interact on our services . 

- (0.306, _Arbitration_) any dispute or claim relating in any way to your use of any amazon service , or to any products or services sold or distributed by amazon or through amazon.com will be resolved by binding arbitration , rather than in court , except that you
- (0.318, _Limitation of liability_) in the event that duolingo suspends or terminates your use of the service or these terms and conditions or you close your account voluntarily , you understand and agree that you will receive no refund or exchange of any kind , including for
- (0.325, _Unilateral termination_) the licenses granted by amazon terminate if you do not comply with these conditions of use or any service terms . 

- (0.354, _Unilateral termination_) failure to meet these standards may result in ebay charging you additional fees , and/or limiting , restricting , suspending , or downgrading your seller account . 

- (0.366, _Contract by using_) you understand that through your use of the services you consent to the collection and use ( as set forth in the privacy policy ) of this information , including the transfer of this information to the united states , ireland , and/or other
- (0.380, _Contract by using_) any revisions to these terms and conditions will become effective the earlier of ( i ) the end of such 7-day period or ( ii ) the first time you access or use the service after such changes . 

- (0.405, _Limitation of liability_) uber 's services may be used by you to request and schedule transportation , goods or logistics services with third party providers , but you agree that uber has no responsibility or liability to you related to any transportation , goods or
- (0.432, _Limitation of liability_) any material downloaded or otherwise obtained through the use of the service is done at your own discretion and risk and you are solely responsible for any damage to your computer or other device or loss of data resulting from the download 
- (0.455, _Contract by using_) you agree to comply with all of the above when accessing or using our services . 

- (0.464, _Unilateral termination_) it is academia.edu 's policy to terminate , in appropriate circumstances , members or other account holders who repeatedly infringe or are believed to be repeatedly infringing the rights of copyright holders . 

- (0.471, _Limitation of liability_) except to the limited extent it may be required by applicable law , linkedin is not responsible for these other sites and apps -- use these at your own risk . 

- (0.477, _Unilateral termination_) we reserve the right to revoke the free trial and put your account on hold in the event that we determine that you are not eligible . 

- (0.479, _Unilateral termination_) we can each end this contract anytime we want . 

- (0.500, _Limitation of liability_) in no event shall the liability of linkedin and its affiliates ( and those that linkedin works with to provide the services ) exceed , in the aggregate for all claims , an amount that is the lesser of ( a ) five times the most recent monthl
- (0.506, _Unilateral change_) netflix regularly makes changes to the service , including the content library . 


## Model says UNFAIR, experts say FAIR (156)
The model flagged these; LexGLUE experts did not. If many look unfair to you, it may be the community set's broader labelling is defensible; if they look fair, this is genuine over-flagging.

- (0.938) we can not and will not be liable for any loss or damage arising from your failure to comply with the above . 

- (0.919) ebay takes no responsibility and assumes no liability for any content provided by you or any third party . 

- (0.916) if you are using the services on behalf of a business ( rather than for your personal use ) , you and snap group limited agree that to the extent permitted by law , all claims and disputes between us arising out of or relating to these term
- (0.908) some jurisdictions do not allow the exclusion of certain warranties or conditions or the limitation or exclusion of liability for loss or damage caused by willful acts , negligence , breach of contract or breach of implied terms , or incide
- (0.861) this limitation of liability is part of the basis of the bargain between you and linkedin and shall apply to all claims of liability ( e.g. warranty , tort , negligence , contract , law ) and even if linkedin or its affiliates has been told
- (0.845) by using the services , you agree that : 

- (0.840) some jurisdictions do not allow the exclusion or limitation of liability for consequential or incidental damages , so the above limitation may not apply to you . 

- (0.826) regardless of who terminates these terms , both you and snap group limited continue to be bound by sections 3 , 6 , 9 , 10 , and 13-22 of the terms . 

- (0.820) evernote does not assume any responsibility for , or liability on account of , the actions or omissions of such third party applications or service providers . 

- (0.819) there 's a good reason for that : these terms do indeed form a legally binding contract between you and snap group limited . 

- (0.818) we refer to each of these as a `` separate agreement . '' 

- (0.818) please read carefully the following terms and conditions ( `` terms '' ) and our privacy policy , which may be found at academia.edu/privacy and which is incorporated by reference into these terms . 

- (0.811) the courts in some countries may not apply the laws of england and wales to some disputes related to these terms . 

- (0.804) amazon does not assume any responsibility or liability for the actions , product , and content of all these and any other third parties . 

- (0.798) finally , the section headings in these terms of service are for convenience only and have no legal or contractual effect . 

- (0.794) we may offer automatic or manual updates to the amazon software at any time and without notice to you . 

- (0.781) some jurisdictions do not allow the exclusion or limitation of certain damages , so some or all of the exclusions and limitations in this section may not apply to you . 

- (0.780) we refer to this as the `` prohibition of class and representative actions . '' 

- (0.773) nothing in these terms of service ( including the limitation of liability provisions ) is intended to exclude or limit any condition , warranty , right or liability which may not be lawfully excluded or limited . 

- (0.768) arbitration is more informal than a lawsuit in court . 
