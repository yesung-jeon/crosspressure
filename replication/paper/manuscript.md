# Abstract

Large language models increasingly simulate members of Congress, and the simulations are validated mainly against aggregate measures such as a liberal–conservative ordering. Yet economic and social issues have divided Congress along different lines, and that ordering cannot represent the members whose economic and abortion votes diverge. Studies of simulated survey respondents suggest that accuracy may depend on whether positions follow the usual combination. We test whether three open models represent these cross-pressured legislators, comparing maps built from the models' internal directions with roll-call votes. The models recover the aligned structure of Congress but miss its cross-pressured members. The maps place 71 to 84 percent of aligned rows in their voting quadrant but only 12 to 17 percent of cross-pressured rows. Stated cross-pressured profiles are followed less faithfully than aligned ones, and in two models crossed combinations are harder even without a persona. For research that simulates political actors, agreement with the aggregate ordering can coexist with a large failure on an identifiable group, chiefly Democrats whose abortion votes leaned toward restriction and Republicans whose votes leaned toward access. For the study of Congress, these simulated legislators miss the members whose two records a single score cannot represent.

# 1. Introduction

Researchers increasingly use large language models as stand-ins for members of Congress. Models have been used to place senators on ideological scales by comparing them in pairs (Wu et al., 2023), to predict roll-call votes in the U.S. House by simulating its members as role-playing agents (Li et al., 2025), and to predict the votes of individual members of the European Parliament from personas built with limited information (Kreutner et al., 2026). Other work looks inside the models: when open models write from the perspective of individual lawmakers, their internal activations linearly predict the lawmakers' DW-NOMINATE scores (Kim et al., 2025). These "silicon legislators" are validated mainly against aggregate measures, such as agreement with a liberal–conservative ordering of members or overall accuracy in predicting votes. Wu et al. (2023) check their ideology scale against existing ideology measures, and Kim et al. (2025) predict the first dimension of DW-NOMINATE, which places members on the liberal–conservative axis. Such benchmarks describe most members well. The history of Congress shows which members it describes poorly.

Economic and social issues have divided Congress along different lines. Social welfare issues have structured party conflict since the New Deal. Cultural issues such as abortion at first divided each party internally, and congressional Republicans became more conservative than Democrats on these issues only in the 1980s and 1990s (Layman & Carsey, 2002). When the Supreme Court decided Roe v. Wade in 1973, the parties' positions on abortion were still unclear, and they became distinct only over the following two decades (Adams, 1997). The standard summary of roll-call voting places every member on one line, so it cannot describe the members caught between these lines on both dimensions. In Poole and Rosenthal's (1991) analysis of votes from 1789 to 1985, a single dimension correctly classifies about 80 percent of individual voting decisions, and a second dimension improves classification by about three percentage points. In the postwar period, economic votes such as the minimum wage lined up on the first dimension, while the final passage of the 1964 Civil Rights Act, which split Southern from Northern members, fell on the second. Later work cautions that the low dimensionality of recent Congresses may reflect party polarization more than the structure of members' preferences (Aldrich et al., 2014). Voting on individual bills is also more complex than the aggregate record suggests, even though first-dimension scores measure aggregate positions well (Roberts et al., 2016). Members whose economic and social votes point in opposite directions, such as Democrats who opposed abortion rights and Republicans who supported them, are therefore the members whose two positions a single liberal–conservative score cannot represent at once. A simulation validated against such a score could look accurate while misplacing exactly these members.

Research on simulated survey respondents gives reason to expect that it does. Argyle et al. (2023) argued that a language model conditioned on respondents' backgrounds reproduces the response distributions of many demographic subgroups, a property they called "algorithmic fidelity." Later studies found failures in relationships between variables, in voters who do not conform to their party, and in combinations of identities. Bisbee et al. (2024) found that synthetic feeling-thermometer scores corresponded closely to survey averages, yet 48 percent of the regression coefficients estimated on the synthetic data differed significantly from those estimated on the survey. Von der Heyde et al. (2026) found that a model predicting German vote choice captured the tendencies of typical voters but missed more complex factors: partisans who voted in line with their party were easy to predict, and voters who did not conform were not. Rennard and Xypolopoulos (2026) found that personas combining two demographic identities tend to answer as if they held only one. The legislator studies contain a related hint. Wu et al. (2023) found that their abortion and gun-control scales predicted interest-group ratings and votes better than ideology alone, so a member's positions on specific issues carry information that a single ordering does not. These mixed findings suggest that whether a silicon legislator is accurate could depend on whether the member's economic and social positions follow the parties' usual combination or cut across it.

Cross-pressured members are a well-defined group, so this possibility can be tested against each member's own voting record. The same configuration is familiar among citizens. Feldman and Johnston (2014) showed that economic and social ideology are both needed to account for domestic policy preferences and that the two have different determinants. Starting from the premise that the beliefs of political elites are captured by a single dimension, Treier and Hillygus (2009) found that many citizens hold liberal positions on one dimension and conservative positions on the other. These cross-pressured citizens tend to call themselves moderates or answer that they do not know. In Congress, the configuration appears in roll-call votes. By the measure we use below, which compares each member's economic and abortion votes with those of other members of the same chamber and Congress, cross-pressured members account for 22 percent of member-Congress rows from the 93rd to the 117th Congress. If a language model represents these members, the economic and social coordinates it assigns them should separate them from Democrats and Republicans whose votes follow the usual combination. If it does not, its agreement with the chamber's aggregate structure would not reveal the failure.

We test this in three open models (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, and OLMo-2-1124-7B-Instruct). From each model we extract an economic and a social direction in its activation space, using answers generated under opposing economic and social instructions. We then prompt the model to speak as each member of every Congress, with no party label in the prompt, and place the member on the two directions. The benchmark is the member's own roll-call record. We build economic and abortion scales from Voteview roll calls (Lewis et al., 2026), selecting the roll calls by the Policy Agendas Project's topic codes (Jones et al., 2025) and Voteview's vote descriptions. The scales assign each member in each Congress to an aligned or a cross-pressured voting quadrant. We then run three analyses. The first compares the model's map with the members' votes. The second states a profile's positions in the prompt and asks whether the model follows them. The third asks the model to choose between policy packages that either follow or cross the parties' usual combinations.

The models recover the aligned structure of Congress but miss its cross-pressured members. First, the map built from each model's own directions follows the roll-call scales: the economic coordinate correlates with the economic vote scale at 0.77 to 0.84, and the social coordinate with the abortion vote scale at 0.61 to 0.69. Second, the map places 71 to 84 percent of aligned member-Congress rows in their voting quadrant, but only 12 to 17 percent of cross-pressured rows (10,444 rows in each model). Among 15 legislators per quadrant selected by their votes from members the models recognize, the map puts 13 to 15 of the aligned members in their own quadrant and 2 to 6 of the cross-pressured members. Third, the failure does not come only from a lack of information. When the prompt states a profile's positions on both dimensions, all three models follow cross-pressured profiles less faithfully than aligned ones. The probability of answering consistently with the stated profile is lower by 0.087 to 0.175, and the share of conditions in which the profile's projection reaches the stated quadrant is lower by 0.127 to 0.473. Fourth, crossed combinations are harder for two of the three models. In choices between two policy packages, each pairing an economic and a social position, legislator personas are more uncertain when the packages cross the parties' usual combinations than when they follow them. The difference in choice entropy is +0.18 and +0.15 bits for Democratic and Republican personas in Qwen and +0.26 and +0.25 in OLMo. The difference appears without any persona as well: +0.17 [+0.04, +0.31] in Qwen and +0.48 [+0.35, +0.61] in OLMo, with 95% intervals that resample the policy passages. Llama shows the opposite sign with personas (−0.14 and −0.15) and no clear difference without them (−0.05 [−0.13, +0.02]).

These results matter for two literatures. For research that uses language models to simulate political actors, they show that agreement with the aggregate ordering of members is compatible with a large failure on an identifiable group of members. The failure becomes visible when each member's own votes serve as the benchmark and cross-pressured members are examined separately. For the study of Congress, they show that the silicon legislators we study miss the members whose economic and social records a single liberal–conservative score cannot represent at once. Because crossed combinations are harder for two of the three models even when no legislator is named, part of the difficulty does not depend on the persona.

The paper proceeds as follows. Section 2 discusses what cross-pressure can mean for a simulated legislator and states the expectations of four accounts. Section 3 describes the models, the two directions, the roll-call benchmark, and the analyses. Section 4 reports the map, Section 5 the stated profiles, and Section 6 the choices between crossed and party-line packages. Section 7 discusses what the results imply for research that uses language models to simulate political actors and for the study of Congress.

# 2. What Cross-Pressure Could Mean for a Simulated Legislator

For a member of Congress, cross-pressure is visible in the voting record. A Democrat who votes with the party on economic legislation and against abortion rights holds two positions that the parties usually keep apart. What we can observe is how a language model handles such a member, and what that looks like depends on the task the model is given.

Our analyses give the models three tasks. In the first, the model receives a member's name, chamber, state, and year and is asked policy questions as that member. It must supply the member's positions itself, and we read the member's economic and social coordinates from the model's activations. In the second, the prompt states the member's position on each dimension, and the model must answer policy questions consistently with that profile. In the third, a legislator persona chooses between two policy packages, each pairing an economic position with a social one. In party-line choices, the packages follow the two parties' usual combinations. In crossed choices, the two packages reverse one half of each usual pairing: economically left and socially traditional against economically right and socially progressive. For a persona that holds its party's usual positions, each crossed package satisfies one of those positions and violates the other. Four accounts of what a crossed combination does to a model's output predict different patterns across the three tasks. We describe each account by the output it predicts. The accounts are not mutually exclusive.

The first account, which we call combination, holds that both positions enter the output and pull against each other. It is the counterpart of the cross-pressured citizens who call themselves moderates (Section 1); Treier and Hillygus (2009) expected that answer because people torn between competing considerations tend to choose the middle category (p. 683). If the account holds, the map should place cross-pressured members near the boundary between quadrants, where the two pulls partly offset each other. Answers to a stated cross-pressured profile should be less consistent with the profile, and the profile's projection should often fall short of its quadrant. Choices between crossed packages should be more uncertain than choices between party-line packages, because no package satisfies both positions.

The second account, dominance, holds that one position governs the output and the other is dropped. Rennard and Xypolopoulos (2026) found this pattern in synthetic survey respondents. When a persona combined two demographic identities, the model's answers were usually explained better by one identity than by the sum of the two. The dominant identity carried a median share of 0.88 of the weight (p. 7). Among Western European voters with left-wing economic and traditional social views, Lefkofridi et al. (2014) found a similar pattern: these voters generally privilege economic congruence, although the choice depends on the issues they care about (p. 65). If dominance holds, the map should place a cross-pressured member in the aligned quadrant that matches the dominant dimension. Answers to a stated profile should follow one dimension and neglect the other. Choices between crossed packages need not be more uncertain, but decided choices should go mostly to the package that satisfies the dominant position.

The third account, default, holds that the output drifts toward a default package: either the position a model takes without any persona, or the usual combination of the persona's party. Language models asked opinion questions without a persona match some demographic groups better than others, and prompting them to answer as a group member improves the match only modestly (Santurkar et al., 2023, pp. 29971–29972, 29975–29976). If the account holds, a cross-pressured member should be placed near the party's aligned quadrant or near the model's default, whatever the member's votes, and answers to a stated profile should drift the same way. For many cross-pressured members this prediction coincides with economic dominance. A Democrat in the economically left and socially traditional quadrant votes on the economically left side, as most Democrats do, so a pull toward the economic position and a pull toward the party's package both move the member into the Democratic aligned quadrant. The map alone cannot separate these two accounts for such members.

The fourth account, direction, holds that a model handles one crossed direction worse than the other. If the account holds, failure should concentrate in one of the two cross-pressured quadrants, so that recall on the map and consistency with stated profiles differ between them. Prior work gives this account the least direct support. We include it because it predicts an asymmetry that the other accounts do not.

Two further explanations do not concern how the two positions combine. The first is missing information. Less may be written about rank-and-file members who cross party lines than about party leaders, so a model may simply not know where they stand. This explanation predicts that the gap between aligned and cross-pressured members should close when the prompt states the member's positions. The second is unfamiliarity. Crossed combinations of positions may be rarer in the text a model learns from, so a model may find them harder whoever holds them. This explanation predicts that crossed packages should be harder even when no legislator is named. It also predicts the stated-profile result, because a stated cross-pressured profile is itself a rarer combination. Table 1 lists what each account and each alternative predicts for the outputs we observe. Several of them predict the same output in some tasks, so the pattern across tasks matters more than any single result. We return to what the results allow us to say about the accounts in Section 7.

**Table 1.** Expected outputs for cross-pressured members under each account. A dash means that the account makes no specific prediction for that output.

| Account | Map (Section 4) | Stated profiles (Section 5) | Packages with a legislator persona (Section 6) | Packages without a persona (Section 6) |
|---|---|---|---|---|
| Combination | placed near the boundary between quadrants | less consistent answers; projection falls short of the stated quadrant; more uncertain choices | crossed choices more uncertain than party-line choices | – |
| Dominance | placed in the aligned quadrant of the dominant dimension | answers follow one dimension | decided crossed choices go mostly to the dominant position; uncertainty need not rise | – |
| Default | placed near the party's aligned quadrant or the model's default | answers drift toward the party's usual package | on party-line items, choices favor the persona's party package; on crossed items, no unique prediction | – |
| Direction | failure concentrated in one cross-pressured quadrant | the two cross-pressured profiles differ | – | – |
| Missing information | cross-pressured members misplaced | gap closes when positions are stated | – | – |
| Unfamiliarity | – | cross-pressured profiles harder | crossed choices harder | crossed choices harder |

# 3. Data and Methods

## Data Collection

**Models.** We study three open instruction-tuned models of similar size from different developers: Qwen2.5-7B-Instruct (Qwen Team, 2025), Llama-3.1-8B-Instruct (Llama Team, 2024), and OLMo-2-1124-7B-Instruct (OLMo Team, 2025). All three run locally in 8-bit floating point (FP8). Every quantity we analyze is read from the models' next-token probabilities or hidden activations.

**Economic and social directions.** To place legislators in a two-dimensional space of the model's own making, we extract an economic and a social direction from each model by adapting the persona-vector procedure of Chen et al. (2025, pp. 3–4). The model answers the same policy questions under opposing system instructions, economically left against economically right and socially progressive against socially traditional, and each direction is the difference between the mean activations of the two sets of answers. Activations are the internal numerical values that a model computes at each of its successive processing stages, called layers. The two directions are then made perpendicular by a symmetric orthogonalization, so that neither axis takes priority. As in Chen et al. (2025, p. 4), the layer is the one at which the directions steer the model's answers most effectively (adding a direction to the activations moves the answers furthest toward the intended side), as rated by Llama-3.1-8B-Instruct for all three models (the judge ranks layers only and scores none of the reported outcomes): layer 20 in Qwen, 14 in Llama, and 15 in OLMo (see Appendix A for the questions, the instructions, and the selection rule). Steering is used only to select the layer. None of the analyses below adds anything to the models' activations.

**Legislator map.** We prompt each model to answer as every member of the House and Senate from the 80th to the 118th Congress, 21,288 member-Congress rows in all. The prompt gives the member's name, chamber, state, and year, but not the member's party. For a set of held-out economic and social questions, we project the model's activations onto the two directions and subtract the projection obtained without the persona, so that the origin of the map is the model's default position. The result is an economic and a social coordinate for every member in every Congress (see Appendix A for the prompt). To know which members the model recognizes, we also ask it for the member's party; it answers correctly for 81 to 92 percent of rows.

**Roll-call benchmark.** The benchmark is each member's own voting record, not an aggregate score. We build an economic and an abortion scale for every Congress and chamber from Voteview roll calls (Lewis et al., 2026). Economic roll calls are those that the Policy Agendas Project codes under macroeconomics, labor, education, environment, or social welfare (Jones et al., 2025), and abortion roll calls are identified from Voteview's vote descriptions. Abortion serves as the social benchmark because its roll calls span the whole period and party positions on it changed during that period (Adams, 1997). Each scale is the first principal component of the members' votes on these roll calls, oriented by its correlation with the first DW-NOMINATE dimension or, where that correlation is weak, by the position of Southern Democrats (see Appendix A). Splitting both scales at their medians within each Congress and chamber assigns every member-Congress row to a voting quadrant. Members who are economically left and socially progressive (EL-SP) or economically right and socially traditional (ER-ST) are aligned with the usual party combinations. Members who are economically left and socially traditional (EL-ST) or economically right and socially progressive (ER-SP) are cross-pressured. Abortion scales exist for 20 Congresses in each chamber between the 93rd (1973) and the 117th (2021). In those Congresses, 10,444 Democratic and Republican rows have both scales, and 22 percent of them are cross-pressured.

## Analysis

We run three analyses, each aimed at one part of the question set out in Section 2.

**Map and votes.** To test whether the map recovers cross-pressured members, we split the model's two coordinates at their medians in the same way and compare each row's map quadrant with its voting quadrant. The main quantity is recall by voting quadrant: the share of rows in a quadrant that the map places in that same quadrant. Row-level recall counts a member once for every Congress served and includes many members close to the medians. We therefore also select 15 members per voting quadrant by their votes, among members whose party Qwen and Llama recognize, most of them far from both medians, and count how many the map places in their own quadrant (see Appendix A for the selection).

**Stated profiles.** To test the missing-information explanation, we state a profile in the prompt instead of leaving the model to infer it. The four profiles correspond to the four quadrants, and each is given in five ways, from a short description of an anonymous House member's positions to the name of a real member whose votes place them in that quadrant (520 conditions per model). We compare cross-pressured with aligned profiles on four outcomes: how consistently the model answers single-issue questions in line with the profile, whether the profile's projection reaches its quadrant, and the uncertainty of the model's choices and of its generated text. The tests, their minimum effects, and the decision rules were fixed before the analysis (see Appendix A).

**Choices between packages.** To test whether the difficulty lies in the combination itself, we ask the model to choose between two policy packages, each pairing an economic with a social position. A party-line item offers the two usual combinations (EL-SP against ER-ST), and a crossed item offers the two reversed combinations (EL-ST against ER-SP), built from the same 54 pairs of economic and social policy passages. The chooser is either one of 20 anonymous House members, half Democrats and half Republicans, or no persona at all. The outcome is the entropy of the model's choice between the two packages, in bits: 0 when the model puts all probability on one package and 1 when it splits evenly. The main contrast is crossed minus party-line entropy. For Democratic and Republican personas, its test, minimum effect, and decision rule were fixed before the analysis; for the no-persona arm, the same contrast is estimated without a registered test. For decided crossed choices, we also describe which position, economic or social, the chosen package satisfies (see Appendix A for the items, the estimation, and the intervals).

# 4. The Map Recovers Aligned Members but Not Cross-Pressured Ones

Each model's map reproduces the main structure of congressional voting. Within each Congress, the economic coordinate correlates with the first DW-NOMINATE dimension at 0.70 to 0.78 on average. Within each Congress and chamber, the economic coordinate correlates with the economic roll-call scale at 0.77 to 0.84, and the social coordinate correlates with the abortion scale at 0.61 to 0.69 (Appendix B, Table B1, gives every value by model). In all three models, the parties are farther apart on the economic coordinate than on the social coordinate. The difference between the Republican and Democratic means, in standard deviations of the coordinate, is 1.25 to 1.40 on the economic coordinate and 0.47 to 0.99 on the social coordinate. This ordering matches the history described in Section 1, in which economic issues divided the parties earlier and more consistently than abortion.

The two coordinates, however, carry less separate information than the two directions suggest. Although the directions are perpendicular, each coordinate also follows the other dimension's roll-call scale. The economic coordinate correlates with the abortion scale, and the social coordinate with the economic scale, at 0.66 to 0.73. In every model, the social coordinate follows the economic scale slightly more closely than the abortion scale (Table B1). The models therefore order most members similarly on the two coordinates.

The test is how the map treats members whose two records diverge. Figure 1d compares each row's map quadrant with its voting quadrant. The map places 71 to 84 percent of aligned rows in their own voting quadrant, but only 12 to 17 percent of cross-pressured rows. Overall, map and votes agree for 60 to 69 percent of rows. Most cross-pressured rows are placed in one of the two aligned quadrants (75 to 81 percent). This share should not be read as a special pull toward aligned positions, because the map places 80 to 88 percent of all rows in aligned quadrants. The map does not separate cross-pressured members from the aligned majority. Because it puts few rows in the cross-pressured quadrants, the low recall there partly reflects how the map's rows are distributed. Two descriptive checks, computed after the main analyses were sealed (Appendix B.5, Table B7), bear on this pattern. A map built from DW-NOMINATE's first and second dimensions, split at the same medians, places 64 percent of cross-pressured rows in their voting quadrant and 54 percent of aligned rows. Its two dimensions are nearly unrelated within a Congress (r = −0.02), so it puts about half of all rows in the cross-pressured quadrants, whereas the models' two coordinates move together (r = 0.73 to 0.88). This comparison mainly reflects how differently the two kinds of map distribute members. Restricted to rows whose party the model names correctly, the models place 77 to 86 percent of aligned rows and 13 to 18 percent of cross-pressured rows in their quadrant, so recognition does not account for the gap. Rows that are economically left and socially traditional are recovered slightly more often than rows that are economically right and socially progressive in every model, by 0.03 to 0.07, but both stay far below the aligned quadrants.

The pattern does not depend on counting members once per Congress or on members close to the medians. Among the 15 legislators per voting quadrant selected by their votes, the map places 13 to 15 of the aligned members in their own quadrant and 2 to 6 of the cross-pressured members (Figure 1e). Figures 1a to 1c show where each model puts these legislators. In all three maps, most aligned members sit in their expected quadrants, and most cross-pressured members sit near the center or in an adjacent aligned quadrant. Which cross-pressured members are missed differs across models. Qwen places the Republicans Barber Conable and William Frenzel, whose abortion votes were among the most progressive in their chamber, among socially traditional Republicans; Llama places both in their own quadrant, and OLMo places one of them there. Qwen places the Democrat Thomas Eagleton in his own quadrant, while Llama and OLMo do not.

**Figure 1.** The map recovers aligned members but not cross-pressured ones. (a)–(c) Maps of Qwen, Llama, and OLMo. Gray points are Democratic and Republican member-Congress rows from the 93rd to the 118th Congress (the voting comparison ends with the 117th) whose party the model recognizes, measured from the median of their Congress and chamber; colored points are the 15 legislators per voting quadrant selected by their votes, at the average of their map positions; the OLMo map omits three of them because OLMo does not recognize their party in the selected rows. Filled points are placed in their own voting quadrant, and open points elsewhere. Labels mark the two members with the most extreme votes in each cross-pressured quadrant, the same four people in every map. (d) Share of member-Congress rows that each model places in their own voting quadrant. (e) Number of the legislators selected by their votes that each model places in their own quadrant. EL = economically left, ER = economically right, SP = socially progressive, ST = socially traditional.

# 5. Stated Cross-Pressured Profiles Are Followed Less Faithfully

If the models misplace cross-pressured members because they know too little about them, stating positions in the prompt should remove most of the difference between cross-pressured and aligned profiles. It does not. In all three models, the answers to single-issue questions follow cross-pressured profiles less consistently than aligned ones. The probability the model gives to the answer that matches the profile averages 0.72 to 0.78 for aligned profiles and 0.60 to 0.66 for cross-pressured profiles, a gap of 0.087 to 0.175 after adjusting for the way the profile is stated. The projection of a cross-pressured profile also reaches its quadrant less often. It does so in 74 to 93 percent of aligned conditions but in 46 to 61 percent of cross-pressured conditions, a gap of 0.127 to 0.473. Both gaps pass the preregistered test in every model (Figure 2a and 2b), and the verdicts do not change when the bootstrap resamples whole base prompts instead of single conditions (Appendix B, Table B3).

The gap in consistency is negative however the profile is stated, although its size varies. It appears in all 15 combinations of model and way of stating the profile, including the way that names a real member without stating any positions. The gap in reaching the quadrant appears in 14 of the 15 combinations (Table B4). The gap in consistency therefore does not depend on whether positions are stated. The gap in reaching the quadrant varies more with the way of stating the profile and is close to zero or reversed in two of the five ways in Qwen (Table B4).

The models are not consistently more uncertain about cross-pressured profiles. The uncertainty of the yes-or-no choices is higher for cross-pressured profiles in Llama (+0.061 [+0.050, +0.072] bits) and OLMo (+0.068 [+0.056, +0.081] bits), and both pass the preregistered test. In Qwen, the difference is close to zero (+0.006 [−0.002, +0.014] bits). The uncertainty of generated text is slightly higher for cross-pressured profiles in all three models, by 0.013 to 0.029 bits per token, but every estimate falls below the preregistered minimum of 0.05 bits, so this hypothesis is not supported in any model (Figure 2c and 2d). The models thus follow stated cross-pressured profiles less faithfully, without a consistent rise in the uncertainty of their answers.

**Figure 2.** Stated cross-pressured profiles are followed less faithfully than aligned ones. Each panel shows the difference between cross-pressured and aligned profiles on one preregistered outcome, adjusted for the way the profile is stated, with its 95 percent bootstrap interval. (a) Probability given to the answer that matches the profile. (b) Share of conditions whose projection reaches the profile's quadrant. (c) Entropy of yes-or-no choices. (d) Mean token entropy of generated text. Dashed lines mark the preregistered minimum effects. Filled markers indicate tests that pass the preregistered rule: an interval that excludes zero, a Holm-adjusted p value below 0.05, and an estimate beyond the minimum effect.

# 6. Crossed Packages Are Harder in Two Models, Even Without a Legislator

The third analysis asks whether part of the difficulty lies in the combination of positions itself. The comparison requires personas that keep to their party's line when the line is available. On party-line items, Qwen and OLMo personas choose their own party's package with probability 0.98 to 1.00. Llama personas do so less reliably, with probabilities of 0.71 for Democrats and 0.60 for Republicans.

In Qwen and OLMo, crossed choices are clearly more uncertain than party-line choices (Figure 3a). Across the two models and parties, the entropy of party-line choices is close to zero, from 0.004 to 0.070 bits, and the entropy of crossed choices ranges from 0.176 to 0.316 bits. The preregistered contrast is +0.18 and +0.15 bits for Democratic and Republican personas in Qwen and +0.26 and +0.25 bits in OLMo, and all four tests pass the preregistered rule. Llama goes in the opposite direction (−0.14 and −0.15 bits). Its party-line choices are already uncertain (0.285 to 0.286 bits), and its crossed choices are less so (0.132 to 0.149 bits). The two models whose personas hold the party line are also the two in which crossed packages are harder.

The same contrast appears without any persona. With no legislator in the prompt, the difference between crossed and party-line entropy is +0.17 [+0.04, +0.31] bits in Qwen and +0.48 [+0.35, +0.61] bits in OLMo, with 95 percent intervals that resample the economic and social passages. In Llama, it is −0.05 [−0.13, +0.02] bits. This contrast has no registered test. Crossed combinations are therefore harder for Qwen and OLMo whether or not a legislator is named. This analysis does not test whether the difference is larger with a persona than without one.

When a persona's crossed choice is decided, it usually goes to the package that keeps the persona's economic position (Figure 3b). The share is 0.84 and 0.90 for Democratic and Republican personas in Qwen and 0.68 and 0.71 in OLMo. In Llama it is 0.92 for Republican personas (53 decided choices) but 0.36 for Democratic personas (66), so both shares rest on few decided choices. The economic position wins in five of the six combinations of model and party. We report these shares descriptively; no test was registered for them. Table B5 in Appendix B gives all values by model.

**Figure 3.** Crossed packages are harder for two of three models, with or without a legislator persona. (a) Difference in choice entropy between crossed and party-line items. Circles are Democratic and Republican personas, with intervals from the wild cluster bootstrap; filled circles pass the preregistered rule, and the dashed line marks the minimum effect. Diamonds are the no-persona arm, with intervals from a two-way bootstrap over the economic and social passages. (b) Share of decided crossed choices that go to the package keeping the persona's economic position; n is the number of decided choices.

# 7. Discussion and Conclusion

We asked whether language models that simulate members of Congress represent the members whose economic and social votes point in different directions. In three open models, they largely do not. Each model's map follows the roll-call record and recovers most aligned members, but it places only 12 to 17 percent of cross-pressured member-Congress rows in their voting quadrant. When the prompt states a cross-pressured profile, the models follow it less faithfully than an aligned one, and in Qwen and OLMo a choice between crossed packages is more uncertain than a choice between party-line packages even without a legislator persona. Llama is the exception in the last analysis: with personas, its crossed choices are less uncertain, and without a persona its interval includes zero.

**Who the simulated Congress misses.** The members at issue are not an abstract category. The counts in this subsection come from a descriptive follow-up computed after the main analyses were sealed (Appendix B.5). In the roll-call record from the 93rd Congress on, 817 of the 912 rows that are economically left and socially traditional belong to Democrats, and 1,084 of the 1,338 rows that are economically right and socially progressive belong to Republicans. Cross-pressure in this period mostly means Democrats whose abortion votes leaned toward restriction and Republicans whose abortion votes leaned toward access. The senators among the legislators we selected by their votes illustrate both groups (Appendix B, Table B2, lists all 60 legislators). The Democrats are Thomas Eagleton of Missouri, Walter Huddleston of Kentucky, Jennings Randolph of West Virginia, John Pastore of Rhode Island, Howell Heflin of Alabama, and John Melcher of Montana. The Republicans are Robert Packwood of Oregon, John Chafee of Rhode Island, and Hiram Fong of Hawaii. Only Randolph is placed in his own quadrant by all three models. Of all 30 cross-pressured legislators selected by their votes, none of the models places 16 correctly (14 are misplaced by all three and two have no OLMo position) and only one is placed correctly by every model, whereas 23 of the 30 aligned legislators are placed correctly by every model.

Where the models put these legislators is informative as well. Packwood, whose votes place him among economically right and socially progressive members, appears in all three maps among economically right and socially traditional members. The same happens to most misplaced cross-pressured Republicans (Qwen 11 of 13; Llama 10 of 12; OLMo 7 of 10), who land in the quadrant that is the usual position of their party. For these members, the map matches the economic record but not the abortion record. Misplaced Democrats are less uniform. Llama and OLMo move most of them into the aligned Democratic quadrant (8 of 10 in Llama and 7 of 13 in OLMo), so that a Democrat like Eagleton appears there as an economically left and socially progressive senator. Qwen places Eagleton correctly but sends other Democrats in both directions, and Huddleston appears in all three maps among economically right and socially traditional members.

Many of the legislators the models miss served early in the sample. Of the 30 cross-pressured legislators selected by their votes, 20 have their first observation in our sample, which begins in 1973, in the 1970s, when the parties' positions on abortion were still forming (Adams, 1997; Layman & Carsey, 2002). The share of cross-pressured rows falls from 33 percent in 1973 to 1979 to 16 percent from 2011 to 2021 (through the 117th Congress, the last with an abortion scale). Over the same decades the maps recover aligned rows increasingly well (Qwen 0.56 to 0.85; Llama 0.79 to 0.89; OLMo 0.66 to 0.90), while the recall of cross-pressured rows stays low (0.07 to 0.22) and is lowest from 2011 to 2021 in every model (Table B6). The maps therefore fit the sorted Congress of recent decades better than the Congress of the 1970s, mainly because they recover aligned members better and those members are more common.

**What the results say about the four accounts.** The results narrow the accounts in Section 2 without selecting one. Missing information is not enough, because stated cross-pressured profiles are still followed less faithfully than aligned ones. Combination receives partial support: crossed packages are more uncertain in two models, and choices about stated cross-pressured profiles are more uncertain in two models, but the uncertainty of generated text rises only slightly, below the preregistered minimum in every model. Dominance fits both the pattern for misplaced Republicans and the economic side of most decided crossed choices. Default fits the first of these as well. As anticipated in Section 2, the map cannot separate the two, because for these members the economic position and the party's package point to the same quadrant. On the map, the two cross-pressured quadrants differ only slightly in recall; for stated profiles, the two differ in some ways of stating the profile and not in others, so the results give direction no consistent support. Unfamiliarity remains possible, since crossed packages are harder for two models without any persona.

**Implications for simulated political actors.** The failure that agreement with the aggregate ordering of members can hide falls on named legislators, such as Packwood, Chafee, Eagleton, and Huddleston, and it is visible when each member's own votes serve as the benchmark and cross-pressured members are examined as a group. This pattern resembles findings on simulated survey respondents. Synthetic data can match averages while missing relationships (Bisbee et al., 2024), and simulated voters who do not conform to their party are predicted poorly (von der Heyde et al., 2026). It differs from the finding of Rennard and Xypolopoulos (2026) that, once sampling noise in the human benchmark is accounted for, counter-stereotypical combinations of identities carry no extra simulation error. Their benchmark is the answer distribution of demographic groups, whereas ours is the record of individual legislators, and with that benchmark the atypical members are the ones the models miss.

**Implications for the study of Congress.** A first dimension correctly classifies most votes (Poole & Rosenthal, 1991), but a single score cannot place a member whose economic and abortion votes diverge on both dimensions at once, and low-dimensional summaries can underemphasize issues that divide the parties internally (Aldrich et al., 2014) and the complexity of individual bill votes (Roberts et al., 2016). The simulated legislators fail with these members even though they produce two coordinates. Their two coordinates move together, so their maps place few members in the cross-pressured quadrants (Section 4). Consider a researcher who simulates how the Senate of the 1970s would vote on an abortion measure. Senators who were cross-pressured in that decade, such as Huddleston (in our sample from 1973 to 1979), Packwood, and Eagleton, are among those the maps most often misplace, averaged over their years in the sample: Huddleston and Packwood in all three models, and Eagleton in two.

**Conclusion.** In the three open models we study, simulated members of Congress reproduce the aligned structure of the chamber and miss its cross-pressured members, and the models follow stated cross-pressured profiles less faithfully than aligned ones. The cross-pressured members they miss are mostly Democrats whose abortion votes leaned toward restriction and Republicans whose votes leaned toward access, a group that was largest before the parties sorted on the issue.

# References

Adams, G. D. (1997). Abortion: Evidence of an issue evolution. *American Journal of Political Science, 41*(3), 718–737. https://doi.org/10.2307/2111673

Aldrich, J. H., Montgomery, J. M., & Sparks, D. B. (2014). Polarization and ideology: Partisan sources of low dimensionality in scaled roll call analyses. *Political Analysis, 22*(4), 435–456. https://doi.org/10.1093/pan/mpt048

Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., & Wingate, D. (2023). Out of one, many: Using language models to simulate human samples. *Political Analysis, 31*(3), 337–351. https://doi.org/10.1017/pan.2023.2

Bisbee, J., Clinton, J. D., Dorff, C., Kenkel, B., & Larson, J. M. (2024). Synthetic replacements for human survey data? The perils of large language models. *Political Analysis, 32*(4), 401–416. https://doi.org/10.1017/pan.2024.5

Chen, R., Arditi, A., Sleight, H., Evans, O., & Lindsey, J. (2025). *Persona vectors: Monitoring and controlling character traits in language models* (arXiv:2507.21509v3). arXiv. https://doi.org/10.48550/arXiv.2507.21509

Feldman, S., & Johnston, C. (2014). Understanding the determinants of political ideology: Implications of structural complexity. *Political Psychology, 35*(3), 337–358. https://doi.org/10.1111/pops.12055

Jones, B. D., Baumgartner, F. R., Theriault, S. M., Epp, D. A., Khandekar, S., & Little, D. (2025). *Policy Agendas Project: Codebook*. https://www.comparativeagendas.net/project/us/datasets

Kim, J., Evans, J., & Schein, A. (2025). Linear representations of political perspective emerge in large language models. In *Proceedings of the Thirteenth International Conference on Learning Representations (ICLR 2025)*. https://doi.org/10.48550/arXiv.2503.02080

Kreutner, M., Lutz, M., & Strohmaier, M. (2026). Persona-driven simulation of voting behavior in the European Parliament with large language models. In *Findings of the Association for Computational Linguistics: EACL 2026* (pp. 490–511). Association for Computational Linguistics. https://doi.org/10.18653/v1/2026.findings-eacl.25

Layman, G. C., & Carsey, T. M. (2002). Party polarization and "conflict extension" in the American electorate. *American Journal of Political Science, 46*(4), 786–802. https://doi.org/10.2307/3088434

Lefkofridi, Z., Wagner, M., & Willmann, J. E. (2014). Left-authoritarians and policy representation in Western Europe: Electoral choice across ideological dimensions. *West European Politics, 37*(1), 65–90. https://doi.org/10.1080/01402382.2013.818354

Lewis, J. B., Poole, K., Rosenthal, H., Boche, A., Rudkin, A., & Sonnet, L. (2026). *Voteview: Congressional roll-call votes database*. https://voteview.com/

Li, H., Gong, R., & Jiang, H. (2025). Political Actor Agent: Simulating legislative system for roll call votes prediction with large language models. *Proceedings of the AAAI Conference on Artificial Intelligence, 39*(1), 388–396. https://doi.org/10.1609/aaai.v39i1.32017

Llama Team, AI @ Meta. (2024). *The Llama 3 herd of models* (arXiv:2407.21783v3). arXiv. https://doi.org/10.48550/arXiv.2407.21783

OLMo Team. (2025). *2 OLMo 2 furious* (arXiv:2501.00656v3). arXiv. https://doi.org/10.48550/arXiv.2501.00656

Poole, K. T., & Rosenthal, H. (1991). Patterns of congressional voting. *American Journal of Political Science, 35*(1), 228–278. https://doi.org/10.2307/2111445

Qwen Team. (2025). *Qwen2.5 technical report* (arXiv:2412.15115v2). arXiv. https://doi.org/10.48550/arXiv.2412.15115

Rennard, V., & Xypolopoulos, C. (2026). *Large language models simulate intersectional synthetic identities with a budget of one to two dimensions* (arXiv:2608.23005v1). arXiv. https://doi.org/10.48550/arXiv.2608.23005

Roberts, J. M., Smith, S. S., & Haptonstahl, S. R. (2016). The dimensionality of congressional voting reconsidered. *American Politics Research, 44*(5), 794–815. https://doi.org/10.1177/1532673X15608940

Santurkar, S., Durmus, E., Ladhak, F., Lee, C., Liang, P., & Hashimoto, T. (2023). Whose opinions do language models reflect? In *Proceedings of the 40th International Conference on Machine Learning* (Proceedings of Machine Learning Research, Vol. 202, pp. 29971–30004). PMLR. https://proceedings.mlr.press/v202/santurkar23a.html

Treier, S., & Hillygus, D. S. (2009). The nature of political ideology in the contemporary electorate. *Public Opinion Quarterly, 73*(4), 679–703. https://doi.org/10.1093/poq/nfp067

von der Heyde, L., Haensch, A.-C., & Wenz, A. (2026). Vox populi, vox AI? Using large language models to estimate German vote choice. *Social Science Computer Review, 44*(3), 549–571. https://doi.org/10.1177/08944393251337014

Wu, P. Y., Nagler, J., Tucker, J. A., & Messing, S. (2023). *Large language models can be used to estimate the latent positions of politicians* (arXiv:2303.12057v4). arXiv. https://doi.org/10.48550/arXiv.2303.12057

# Appendix A. Design Details

## A.1 Economic and Social Directions

For each axis, the model answers 20 policy questions under five pairs of system instructions, five times per question and instruction, with up to 256 new tokens. One instruction in each pair asks for economically left (or socially progressive) positions, and the other asks for economically right (or socially traditional) positions. We drop a pair of answers when either answer is shorter than 15 tokens or begins with a refusal. At each layer, the direction is the difference between the mean activations of the two sets of answers, where each answer's activations are averaged over its tokens (Chen et al., 2025, pp. 3–4). Unlike Chen et al., we filter answers by length and refusal rather than by trait ratings, and we select one layer jointly for the two axes. The economic and social directions are orthogonalized with Löwdin's symmetric procedure.

To select the layer, we add each unit direction to the activation at the last position, scaled by the activation's norm, at ten strengths from 0.05 to 0.60, and generate greedy answers to ten held-out questions. Llama-3.1-8B-Instruct rates each answer from 0 to 100 on the corresponding axis, using the expected value of its next-token distribution over the integers. A layer's score on an axis is its best mean rating across strengths. The selected layer has the smallest sum of its ranks on the two axes: layer 20 in Qwen, 14 in Llama, and 15 in OLMo.

## A.2 Legislator Map

The system prompt reads: "You are [name], who serves in the U.S. [chamber] representing [state] in [year]. Answer the question in the first person, as [name] would, reflecting this person's actual political positions at that time." The user turn is one of five held-out economic or five held-out social questions. For each question, we take the activation of the last prompt token at the selected layer and project it onto the orthogonalized directions. A member's economic coordinate is the mean projection over the economic questions minus the same mean without the persona, and the social coordinate is computed in the same way from the social questions. Both coordinates are oriented so that higher values are economically right and socially traditional.

To record recognition, we ask in a separate prompt which party the member belonged to and generate up to eight tokens greedily. A row counts as recognized if the answer contains the member's party. Correlations between the map and the roll-call scales use recognized rows. The comparison of map and voting quadrants uses all rows.

## A.3 Roll-Call Scales

Policy Agendas Project roll-call topics are joined to Voteview roll calls by Congress, chamber, and roll-call number. Economic roll calls are those coded under major topics 1 (macroeconomics), 5 (labor), 6 (education), 7 (environment), and 13 (social welfare). Abortion roll calls are those whose Voteview description contains "abortion" or "unborn." For each Congress, chamber, and issue, we keep roll calls on which the minority side cast at least 10 percent of the votes and require at least three such roll calls and at least three votes from a member. Yeas are coded 1, nays 0, and other responses as missing. After centering each roll call and setting missing values to zero, the first principal component is the member's scale. The scale is oriented to correlate positively with the first DW-NOMINATE dimension when the absolute correlation is at least 0.3, and otherwise so that Southern Democrats score above other Democrats. Party differences are never used to orient a scale.

## A.4 Legislators Selected by Their Votes

Eligible members are Democrats and Republicans from the 93rd Congress on whose party both Qwen and Llama recognize. Within each voting quadrant, we take the ten members whose smaller distance from the two medians, in standard units, is largest, and five members drawn at random from the rest of the quadrant, 15 per quadrant in all. Model coordinates are not used in the selection. A member counts as placed if the average of the member's map coordinates over the selected Congresses, each measured from the median of that Congress and chamber, falls in the member's voting quadrant.

## A.5 Stated Profiles

The four profiles use these positions. Economically left: "expanding Social Security, public health insurance and union rights." Economically right: "cutting taxes, reducing government spending and deregulating business." Socially progressive: "abortion rights and same-sex marriage." Socially traditional: "restrictions on abortion and a traditional definition of marriage."

Each profile is given in five ways: (1) in the system prompt of an anonymous House member, with ten states and three years; (2) as an instruction in the user turn, with the same states and years; (3) as the pair of extraction instructions for the two axes, five pairs in two orders; (4) as four example questions answered in line with the profile, with 30 random draws; and (5) as the name of a real member whose votes place them in the profile's quadrant, 30 members per profile, without any stated positions. This gives 520 conditions per model.

In each condition, the model answers 16 single-issue questions, eight economic and eight social, with yes or no; the four questions shown as examples in way (4) are left out of that condition. Consistency is the probability assigned to the answer that matches the profile, averaged over these questions. Choice uncertainty is the binary entropy of the yes and no probabilities, in bits. Generation uncertainty is the mean token entropy of a greedy answer of up to 64 tokens. For the quadrant measure, the condition's prompt is projected onto the two directions as in A.2, and we record whether the projection falls in the profile's quadrant, measured from the center of the four profiles given in the same way.

Each outcome is regressed on an indicator for cross-pressured profiles with fixed effects for the way the profile is given. The preregistered minimum effects are +0.05 bits for the two uncertainty measures, −0.05 for consistency, and −0.10 for the quadrant measure. Intervals come from 2,000 bootstrap draws that resample conditions within each combination of profile and way of stating it. A hypothesis is supported if the 95 percent interval excludes zero, the Holm-adjusted p value across the four outcomes is below 0.05, and the estimate passes the minimum effect. As a sensitivity analysis, we also cluster the bootstrap by the base prompt.

## A.6 Choices Between Packages

The personas are 20 anonymous House members, ten Democrats and ten Republicans with different states and years. The same items are also asked without a persona. Six economic and nine social policy passages are combined into 54 topic pairs, and each pair yields one party-line item and one crossed item. Each item is asked in both orders of the two packages, and the model answers A or B. The outcome is the binary entropy, in bits, of the model's probabilities for A and B, normalized over the two answers and averaged over the two orders.

The main contrast is crossed minus party-line entropy with persona-by-topic fixed effects, estimated separately for Democratic and Republican personas. Standard errors are clustered by persona, p values come from a wild cluster bootstrap with 9,999 Rademacher draws (suited to the ten clusters per party), and the two party tests are Holm-adjusted. The preregistered minimum effect is +0.05 bits. A crossed choice counts as decided when the probability of either package differs from one half by at least 0.10. For decided choices, we report the share that goes to the package satisfying the persona's economic position. For the no-persona arm, the contrast is averaged over the topic pairs. Because pairs that share a passage are not independent, its 95 percent interval comes from a two-way bootstrap that resamples economic and social passages independently, with 10,000 draws.

These items come from a larger battery whose other conditions add the directions to the model's activations. This paper analyzes only the items asked without such an intervention.

## A.7 Items

The policy questions and passages in A.5 and A.6 were drafted with the help of a large language model and approved by the authors. The full item texts are in the replication materials.

# Appendix B. Supplementary Results

Tables B1 to B7 report, by model, the quantities that the main text summarizes as ranges. Tables B1 to B5 come from the analyses fixed before the results were computed. Tables B6 and B7 and the legislator columns of Table B2 come from the descriptive follow-ups described in B.5.

## B.1 The Map (Section 4)

**Table B1.** Map and roll-call scales by model.

|  | Qwen | Llama | OLMo |
|---|---|---|---|
| Layer | 20 | 14 | 15 |
| Party recognized (share of rows) | 0.81 | 0.92 | 0.85 |
| r(economic coordinate, DW-NOMINATE dim. 1), mean within Congress | 0.71 | 0.78 | 0.70 |
| r(economic coordinate, economic scale) | 0.77 | 0.84 | 0.79 |
| r(social coordinate, abortion scale) | 0.61 | 0.69 | 0.69 |
| r(social coordinate, economic scale) | 0.66 | 0.73 | 0.73 |
| r(economic coordinate, abortion scale) | 0.67 | 0.73 | 0.69 |
| Party gap, economic coordinate (SD units) | 1.25 | 1.40 | 1.29 |
| Party gap, social coordinate (SD units) | 0.47 | 0.99 | 0.97 |
| Agreement of map and voting quadrants | 0.60 | 0.69 | 0.64 |
| Recall, EL-SP | 0.72 | 0.85 | 0.79 |
| Recall, ER-ST | 0.70 | 0.83 | 0.79 |
| Recall, EL-ST | 0.21 | 0.19 | 0.14 |
| Recall, ER-SP | 0.14 | 0.15 | 0.11 |

The DW-NOMINATE entry is the mean of correlations computed separately in each Congress. The issue-scale correlations pool recognized rows after demeaning within Congress and chamber. Party recognition uses all 21,288 map rows; recall and agreement use the 10,444 Democratic and Republican rows with both voting scales (93rd to 117th Congress) in every model.

## B.2 Legislators Selected by Their Votes (Sections 4 and 7)

**Table B2.** The 60 legislators selected by their votes and where each model places them. "yes" means placed in the voting quadrant; otherwise the cell gives the map quadrant; a dash means the model does not recognize the member's party in the selected rows. "deepest" members are the ten farthest from both medians in their quadrant; "random" members are drawn from the rest.

| Voting quadrant | Legislator | Party | Chamber | State | Years in sample | Selection | Qwen | Llama | OLMo |
|---|---|---|---|---|---|---|---|---|---|
| EL-SP | John Sarbanes | D | House | MD | 2011–2017 | deepest | yes | yes | yes |
| EL-SP | Yvette Clarke | D | House | NY | 2011–2017 | deepest | yes | yes | yes |
| EL-SP | Paul Tonko | D | House | NY | 2011–2017 | deepest | yes | yes | yes |
| EL-SP | Judy Chu | D | House | CA | 2011–2017 | deepest | yes | yes | yes |
| EL-SP | David Cicilline | D | House | RI | 2011–2017 | deepest | yes | yes | yes |
| EL-SP | Mark Desaulnier | D | House | CA | 2015–2017 | deepest | yes | yes | yes |
| EL-SP | Brenda Lawrence | D | House | MI | 2015–2017 | deepest | yes | yes | yes |
| EL-SP | Bonnie Watson Coleman | D | House | NJ | 2015–2017 | deepest | yes | yes | yes |
| EL-SP | Mark Takano | D | House | CA | 2013–2017 | deepest | yes | yes | yes |
| EL-SP | Juan Vargas | D | House | CA | 2013–2017 | deepest | yes | yes | yes |
| EL-SP | Brad Sherman | D | House | CA | 1997–2017 | random | yes | yes | yes |
| EL-SP | Ben Chandler | D | House | KY | 2005–2011 | random | EL-ST | yes | yes |
| EL-SP | Paul Tsongas | D | House/Senate | MA | 1975–1983 | random | yes | yes | ER-SP |
| EL-SP | Louis Stokes | D | House | OH | 1973–1997 | random | yes | yes | yes |
| EL-SP | Lucien Nedzi | D | House | MI | 1973–1979 | random | ER-SP | yes | EL-ST |
| EL-ST | Thomas Eagleton | D | Senate | MO | 1973–1985 | deepest | yes | EL-SP | EL-SP |
| EL-ST | Jennings Randolph | D | Senate | WV | 1975–1983 | deepest | yes | yes | yes |
| EL-ST | John Melcher | D | House/Senate | MT | 1973–1987 | deepest | ER-SP | yes | EL-SP |
| EL-ST | Timothy Roemer | D | House | IN | 1991–2001 | deepest | EL-SP | EL-SP | ER-SP |
| EL-ST | Frank Tejeda | D | House | TX | 1993–1995 | deepest | EL-SP | EL-SP | ER-ST |
| EL-ST | John Pastore | D | Senate | RI | 1973–1975 | deepest | yes | yes | ER-ST |
| EL-ST | Glenn Poshard | D | House | IL | 1991–1997 | deepest | EL-SP | EL-SP | ER-ST |
| EL-ST | James Burke | D | House | MA | 1973–1977 | deepest | ER-SP | yes | EL-SP |
| EL-ST | James Delaney | D | House | NY | 1973–1977 | deepest | ER-SP | EL-SP | EL-SP |
| EL-ST | Ronald Klink | D | House | PA | 1993–1999 | deepest | EL-SP | EL-SP | EL-SP |
| EL-ST | Edward Beard | D | House | RI | 1975–1979 | random | ER-SP | EL-SP | EL-SP |
| EL-ST | Walter Huddleston | D | Senate | KY | 1973–1979 | random | ER-ST | ER-ST | ER-ST |
| EL-ST | Lincoln Davis | D | House | TN | 2003–2005 | random | yes | yes | EL-SP |
| EL-ST | Collin Peterson | D | House | MN | 1991–2017 | random | yes | EL-SP | ER-SP |
| EL-ST | Howell Heflin | D | Senate | AL | 1979–1995 | random | yes | ER-ST | yes |
| ER-SP | Barber Conable Jr. | R | House | NY | 1973–1983 | deepest | ER-ST | yes | ER-ST |
| ER-SP | William Frenzel | R | House | MN | 1973–1987 | deepest | ER-ST | yes | yes |
| ER-SP | David Stockman | R | House | MI | 1977–1979 | deepest | ER-ST | ER-ST | ER-ST |
| ER-SP | Robert Whittaker | R | House | KS | 1979–1987 | deepest | ER-ST | ER-ST | ER-ST |
| ER-SP | Jan Meyers | R | House | KS | 1985–1995 | deepest | ER-ST | ER-ST | – |
| ER-SP | James Broyhill | R | House | NC | 1973–1985 | deepest | ER-ST | ER-ST | ER-ST |
| ER-SP | Herman Schneebeli | R | House | PA | 1973–1975 | deepest | ER-ST | ER-ST | – |
| ER-SP | Carl Pursell | R | House | MI | 1979–1991 | deepest | yes | ER-ST | EL-ST |
| ER-SP | James Kolbe | R | House | AZ | 1985–2005 | deepest | ER-ST | yes | yes |
| ER-SP | Paul Findley | R | House | IL | 1973–1981 | deepest | ER-ST | ER-ST | EL-SP |
| ER-SP | Hiram Fong | R | Senate | HI | 1973–1975 | random | EL-ST | ER-ST | EL-ST |
| ER-SP | Robert Packwood | R | Senate | OR | 1973–1995 | random | ER-ST | ER-ST | ER-ST |
| ER-SP | John Chafee | R | Senate | RI | 1977–1999 | random | yes | EL-SP | ER-ST |
| ER-SP | Mary Bono | R | House | CA | 1997–2011 | random | EL-SP | EL-SP | yes |
| ER-SP | Barry Loudermilk | R | House | GA | 2015–2017 | random | ER-ST | ER-ST | ER-ST |
| ER-ST | Roman Hruska | R | Senate | NE | 1973–1975 | deepest | ER-SP | yes | yes |
| ER-ST | Carl Curtis | R | Senate | NE | 1973–1977 | deepest | yes | yes | yes |
| ER-ST | Bob Corker | R | Senate | TN | 2007–2009 | deepest | yes | yes | yes |
| ER-ST | Jake Garn | R | Senate | UT | 1975–1991 | deepest | yes | yes | yes |
| ER-ST | James McClure | R | Senate | ID | 1977–1989 | deepest | yes | yes | yes |
| ER-ST | Jesse Helms | R | Senate | NC | 1973–1999 | deepest | yes | yes | yes |
| ER-ST | Paul Laxalt | R | Senate | NV | 1975–1985 | deepest | yes | yes | yes |
| ER-ST | Gordon Humphrey | R | Senate | NH | 1979–1989 | deepest | yes | yes | yes |
| ER-ST | Robert Griffin | R | Senate | MI | 1973–1977 | deepest | yes | yes | – |
| ER-ST | Bob Dole | R | Senate | KS | 1973–1995 | deepest | yes | ER-SP | yes |
| ER-ST | Vin Weber | R | House | MN | 1981–1991 | random | ER-SP | yes | ER-SP |
| ER-ST | Clifford Stearns | R | House | FL | 1991–2011 | random | yes | yes | yes |
| ER-ST | Ted Cruz | R | Senate | TX | 2019–2021 | random | yes | yes | yes |
| ER-ST | Mike Coffman | R | House | CO | 2011–2017 | random | yes | yes | yes |
| ER-ST | Ann Wagner | R | House | MO | 2013–2017 | random | yes | yes | yes |

## B.3 Stated Profiles (Section 5)

**Table B3.** Cross-pressured profiles are followed less consistently and reach their quadrant less often in all three models; choice uncertainty rises in two models and generation uncertainty in none beyond the minimum. Preregistered tests: cross-pressured minus aligned profiles with fixed effects for the way the profile is stated. Primary intervals resample conditions within profile and way; the sensitivity intervals resample base prompts.

| Model | Outcome | Aligned mean | Cross-pressured mean | Difference | 95% interval | Base-prompt interval | Minimum | Supported |
|---|---|---|---|---|---|---|---|---|
| Qwen | H3 consistency | 0.72 | 0.63 | −0.087 | [−0.095, −0.078] | [−0.100, −0.073] | −0.05 | yes |
| Qwen | H4 quadrant reached | 0.74 | 0.61 | −0.127 | [−0.192, −0.065] | [−0.189, −0.066] | −0.10 | yes |
| Qwen | H1 choice uncertainty | 0.10 | 0.10 | +0.006 | [−0.002, +0.014] | [−0.002, +0.014] | +0.05 | no |
| Qwen | H2 generation uncertainty | 0.66 | 0.68 | +0.013 | [+0.005, +0.022] | [+0.006, +0.021] | +0.05 | no |
| Llama | H3 consistency | 0.78 | 0.66 | −0.127 | [−0.140, −0.114] | [−0.138, −0.116] | −0.05 | yes |
| Llama | H4 quadrant reached | 0.87 | 0.55 | −0.319 | [−0.373, −0.265] | [−0.367, −0.250] | −0.10 | yes |
| Llama | H1 choice uncertainty | 0.25 | 0.31 | +0.061 | [+0.050, +0.072] | [+0.050, +0.072] | +0.05 | yes |
| Llama | H2 generation uncertainty | 0.67 | 0.69 | +0.014 | [+0.004, +0.023] | [+0.005, +0.023] | +0.05 | no |
| OLMo | H3 consistency | 0.78 | 0.60 | −0.175 | [−0.188, −0.163] | [−0.188, −0.163] | −0.05 | yes |
| OLMo | H4 quadrant reached | 0.93 | 0.46 | −0.473 | [−0.519, −0.423] | [−0.532, −0.426] | −0.10 | yes |
| OLMo | H1 choice uncertainty | 0.21 | 0.28 | +0.068 | [+0.056, +0.081] | [+0.054, +0.082] | +0.05 | yes |
| OLMo | H2 generation uncertainty | 0.72 | 0.74 | +0.029 | [+0.021, +0.037] | [+0.021, +0.037] | +0.05 | no |

**Table B4.** Cross-pressured minus aligned profiles by the way the profile is stated (descriptive means).

| Model | Way of stating the profile | Consistency | Quadrant reached |
|---|---|---|---|
| Qwen | anonymous member, system prompt | −0.154 | −0.017 |
| Qwen | user-turn instruction | −0.138 | −0.333 |
| Qwen | extraction instructions | −0.075 | −0.200 |
| Qwen | four examples | −0.018 | +0.050 |
| Qwen | named real member | −0.041 | −0.183 |
| Llama | anonymous member, system prompt | −0.173 | −0.067 |
| Llama | user-turn instruction | −0.099 | −0.033 |
| Llama | extraction instructions | −0.098 | −0.350 |
| Llama | four examples | −0.072 | −0.533 |
| Llama | named real member | −0.173 | −0.633 |
| OLMo | anonymous member, system prompt | −0.201 | −0.500 |
| OLMo | user-turn instruction | −0.191 | −0.233 |
| OLMo | extraction instructions | −0.171 | −0.150 |
| OLMo | four examples | −0.151 | −0.683 |
| OLMo | named real member | −0.161 | −0.583 |

## B.4 Choices Between Packages (Section 6)

**Table B5.** Crossed choices are more uncertain than party-line choices in Qwen and OLMo and less uncertain in Llama with personas. Entropy of party-line and crossed choices (bits), the crossed minus party-line contrast with its 95 percent interval (wild cluster bootstrap for personas; two-way passage bootstrap for the no-persona arm), the Holm-adjusted p value, the mean probability given to the persona's own package on party-line items, and the share of decided crossed choices that go to the economic position (number of decided choices). The no-persona arm has no registered test.

| Model | Chooser | Party-line entropy | Crossed entropy | Difference | 95% interval | Holm p | Supported | Own package | Economic share (n) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen | Democratic personas | 0.004 | 0.186 | +0.182 | [+0.172, +0.192] | 0.003 | yes | 1.00 | 0.84 (344) |
| Qwen | Republican personas | 0.027 | 0.176 | +0.148 | [+0.141, +0.156] | 0.003 | yes | 0.98 | 0.90 (432) |
| Qwen | no persona | 0.053 | 0.226 | +0.172 | [+0.037, +0.309] | – | – | – | – |
| Llama | Democratic personas | 0.285 | 0.149 | −0.136 | [−0.165, −0.106] | 0.004 | no | 0.71 | 0.36 (66) |
| Llama | Republican personas | 0.286 | 0.132 | −0.153 | [−0.190, −0.116] | 0.004 | no | 0.60 | 0.92 (53) |
| Llama | no persona | 0.144 | 0.092 | −0.051 | [−0.128, +0.025] | – | – | – | – |
| OLMo | Democratic personas | 0.025 | 0.282 | +0.256 | [+0.236, +0.276] | 0.002 | yes | 1.00 | 0.68 (468) |
| OLMo | Republican personas | 0.070 | 0.316 | +0.246 | [+0.225, +0.266] | 0.002 | yes | 0.98 | 0.71 (404) |
| OLMo | no persona | 0.161 | 0.643 | +0.482 | [+0.350, +0.607] | – | – | – | – |

## B.5 Descriptive Follow-Up (Section 7)

The legislator columns of Table B2, Table B6 and Table B7 were computed after the analyses above had been sealed, from the same stored maps and roll-call scales. They are descriptive, and no test was registered for them.

**Table B6.** Recall of aligned rows rises across periods while recall of cross-pressured rows stays low. Democratic and Republican member-Congress rows by period: number of rows, share cross-pressured, and recall of aligned and cross-pressured rows.

| Model | Period | Rows | Cross-pressured | Recall, aligned | Recall, cross-pressured |
|---|---|---|---|---|---|
| Qwen | 1973–79 | 2,011 | 33% | 0.56 | 0.18 |
| Qwen | 1980s | 2,166 | 26% | 0.61 | 0.17 |
| Qwen | 1990s | 2,657 | 17% | 0.73 | 0.22 |
| Qwen | 2000s | 1,679 | 16% | 0.80 | 0.16 |
| Qwen | 2011–21 | 1,931 | 16% | 0.85 | 0.10 |
| Llama | 1973–79 | 2,011 | 33% | 0.79 | 0.15 |
| Llama | 1980s | 2,166 | 26% | 0.79 | 0.20 |
| Llama | 1990s | 2,657 | 17% | 0.85 | 0.18 |
| Llama | 2000s | 1,679 | 16% | 0.87 | 0.17 |
| Llama | 2011–21 | 1,931 | 16% | 0.89 | 0.11 |
| OLMo | 1973–79 | 2,011 | 33% | 0.66 | 0.12 |
| OLMo | 1980s | 2,166 | 26% | 0.70 | 0.13 |
| OLMo | 1990s | 2,657 | 17% | 0.80 | 0.15 |
| OLMo | 2000s | 1,679 | 16% | 0.86 | 0.14 |
| OLMo | 2011–21 | 1,931 | 16% | 0.90 | 0.07 |

**Table B7.** A map built from DW-NOMINATE places most cross-pressured rows in their voting quadrant, partly because its two nearly unrelated dimensions put about half of all rows in the cross-pressured quadrants; the models do not, and restricting them to rows whose party they recognize changes little. Recall by voting quadrant and pooled over aligned and over cross-pressured rows. Within Congress and chamber, DW-NOMINATE's two dimensions correlate at −0.02; the models' two coordinates correlate at 0.73, 0.88, 0.87 (Qwen, Llama, OLMo).

| Map | Recall EL-SP | Recall ER-ST | Recall EL-ST | Recall ER-SP | Aligned (pooled) | Cross-pressured (pooled) | Rows |
|---|---|---|---|---|---|---|---|
| DW-NOMINATE map (dimensions 1 and 2) | 0.55 | 0.53 | 0.73 | 0.58 | 0.54 | 0.64 | 10,444 |
| Qwen map, all rows | 0.72 | 0.70 | 0.21 | 0.14 | 0.71 | 0.17 | 10,444 |
| Qwen map, recognized rows | 0.78 | 0.77 | 0.22 | 0.15 | 0.77 | 0.18 | 8,805 |
| Llama map, all rows | 0.85 | 0.83 | 0.19 | 0.15 | 0.84 | 0.17 | 10,444 |
| Llama map, recognized rows | 0.89 | 0.83 | 0.18 | 0.16 | 0.86 | 0.17 | 9,705 |
| OLMo map, all rows | 0.79 | 0.79 | 0.14 | 0.11 | 0.79 | 0.12 | 10,444 |
| OLMo map, recognized rows | 0.81 | 0.85 | 0.14 | 0.12 | 0.82 | 0.13 | 9,164 |
