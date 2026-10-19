# Appendix A. Design Details

<!-- Draft v1 (2026-10-19). Details moved out of Section 3 v2 at the author's instruction. Values in double braces are filled by
     paper/fill_intro.py; design constants are read from the code (paper/CLAIMS_TO_CODE.md, Section 3 table). -->

## A.1 Economic and Social Directions

For each axis, the model answers 20 policy questions under five pairs of system instructions, five times per question and instruction, with up to 256 new tokens. One instruction in each pair asks for economically left (or socially progressive) positions, and the other asks for economically right (or socially traditional) positions. We drop a pair of answers when either answer is shorter than 15 tokens or begins with a refusal. At each layer, the direction is the difference between the mean activations of the two sets of answers, where each answer's activations are averaged over its tokens (Chen et al., 2025, pp. 3–4). Unlike Chen et al., we filter answers by length and refusal rather than by trait ratings, and we select one layer jointly for the two axes. The economic and social directions are orthogonalized with Löwdin's symmetric procedure.

To select the layer, we add each unit direction to the activation at the last position, scaled by the activation's norm, at ten strengths from 0.05 to 0.60, and generate greedy answers to ten held-out questions. Llama-3.1-8B-Instruct rates each answer from 0 to 100 on the corresponding axis, using the expected value of its next-token distribution over the integers. A layer's score on an axis is its best mean rating across strengths. The selected layer has the smallest sum of its ranks on the two axes: layer {{layer_qwen}} in Qwen, {{layer_llama}} in Llama, and {{layer_olmo}} in OLMo.

## A.2 Legislator Map

The system prompt reads: "You are [name], who serves in the U.S. [chamber] representing [state] in [year]. Answer the question in the first person, as [name] would, reflecting this person's actual political positions at that time." The user turn is one of five held-out economic or five held-out social questions. For each question, we take the activation of the last prompt token at the selected layer and project it onto the orthogonalized directions. A member's economic coordinate is the mean projection over the economic questions minus the same mean without the persona, and the social coordinate is computed in the same way from the social questions. Both coordinates are oriented so that higher values are economically right and socially traditional.

To record recognition, we ask in a separate prompt which party the member belonged to and generate up to eight tokens greedily. A row counts as recognized if the answer contains the member's party. Correlations between the map and the roll-call scales use recognized rows. The comparison of map and voting quadrants uses all rows.

## A.3 Roll-Call Scales

Policy Agendas Project roll-call topics are joined to Voteview roll calls by Congress, chamber, and roll-call number. Economic roll calls are those coded under major topics 1 (macroeconomics), 5 (labor), 6 (education), 7 (environment), and 13 (social welfare). Abortion roll calls are those whose Voteview description contains "abortion" or "unborn." For each Congress, chamber, and issue, we keep roll calls on which the minority side cast at least 10 percent of the votes and require at least three such roll calls and at least three votes from a member. Yeas are coded 1, nays 0, and other responses as missing. After centering each roll call and setting missing values to zero, the first principal component is the member's scale. The scale is oriented to correlate positively with the first DW-NOMINATE dimension when the absolute correlation is at least 0.3, and otherwise so that Southern Democrats score above other Democrats. Party differences are never used to orient a scale.

## A.4 Legislators Selected by Their Votes

Eligible members are Democrats and Republicans from the 93rd Congress on whose party both Qwen and Llama recognize. Within each voting quadrant, we take the ten members whose smaller distance from the two medians, in standard units, is largest, and five members drawn at random from the rest of the quadrant, {{a12_n}} per quadrant in all. Model coordinates are not used in the selection. A member counts as placed if the average of the member's map coordinates over the selected Congresses, each measured from the median of that Congress and chamber, falls in the member's voting quadrant.

## A.5 Stated Profiles

The four profiles use these positions. Economically left: "expanding Social Security, public health insurance and union rights." Economically right: "cutting taxes, reducing government spending and deregulating business." Socially progressive: "abortion rights and same-sex marriage." Socially traditional: "restrictions on abortion and a traditional definition of marriage."

Each profile is given in five ways: (1) in the system prompt of an anonymous House member, with ten states and three years; (2) as an instruction in the user turn, with the same states and years; (3) as the pair of extraction instructions for the two axes, five pairs in two orders; (4) as four example questions answered in line with the profile, with 30 random draws; and (5) as the name of a real member whose votes place them in the profile's quadrant, 30 members per profile, without any stated positions. This gives {{s5_n}} conditions per model.

In each condition, the model answers 16 single-issue questions, eight economic and eight social, with yes or no; the four questions shown as examples in way (4) are left out of that condition. Consistency is the probability assigned to the answer that matches the profile, averaged over these questions. Choice uncertainty is the binary entropy of the yes and no probabilities, in bits. Generation uncertainty is the mean token entropy of a greedy answer of up to 64 tokens. For the quadrant measure, the condition's prompt is projected onto the two directions as in A.2, and we record whether the projection falls in the profile's quadrant, measured from the center of the four profiles given in the same way.

Each outcome is regressed on an indicator for cross-pressured profiles with fixed effects for the way the profile is given. The preregistered minimum effects are +0.05 bits for the two uncertainty measures, −0.05 for consistency, and −0.10 for the quadrant measure. Intervals come from 2,000 bootstrap draws that resample conditions within each combination of profile and way of stating it. A hypothesis is supported if the 95 percent interval excludes zero, the Holm-adjusted p value across the four outcomes is below 0.05, and the estimate passes the minimum effect. As a sensitivity analysis, we also cluster the bootstrap by the base prompt.

## A.6 Choices Between Packages

The personas are 20 anonymous House members, ten Democrats and ten Republicans with different states and years. The same items are also asked without a persona. Six economic and nine social policy passages are combined into {{np_n_pairs}} topic pairs, and each pair yields one party-line item and one crossed item. Each item is asked in both orders of the two packages, and the model answers A or B. The outcome is the binary entropy, in bits, of the model's probabilities for A and B, normalized over the two answers and averaged over the two orders.

The main contrast is crossed minus party-line entropy with persona-by-topic fixed effects, estimated separately for Democratic and Republican personas. Standard errors are clustered by persona, p values come from a wild cluster bootstrap with 9,999 Rademacher draws (suited to the ten clusters per party), and the two party tests are Holm-adjusted. The preregistered minimum effect is +0.05 bits. A crossed choice counts as decided when the probability of either package differs from one half by at least 0.10. For decided choices, we report the share that goes to the package satisfying the persona's economic position. For the no-persona arm, the contrast is averaged over the topic pairs. Because pairs that share a passage are not independent, its 95 percent interval comes from a two-way bootstrap that resamples economic and social passages independently, with 10,000 draws.

These items come from a larger battery whose other conditions add the directions to the model's activations. This paper analyzes only the items asked without such an intervention.

## A.7 Items

The policy questions and passages in A.5 and A.6 were drafted with the help of a large language model and approved by the authors. The full item texts are in the replication materials.
