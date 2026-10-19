# Citations in the introduction and where the sources support them

Checked against the full text on 2026-10-18 (page numbers are the journal's or proceedings' own; arXiv versions as listed).

| Paragraph | Sentence (short) | Source | Page |
|---|---|---|---|
| 1 | Senators placed on ideological scales by pairwise comparison | Wu et al. (2023), arXiv:2303.12057v4 | p. 1 (abstract), p. 3 (116th Senate, ChatGPT-3.5, name, party, state only) |
| 1 | Roll-call votes predicted with role-playing agents | Li et al. (2025), AAAI 39(1) | read in arXiv:2412.07144v2, p. 1 (the AAAI server was unavailable) |
| 1 | Votes of individual MEPs predicted from personas built with limited information | Kreutner et al. (2026), Findings of EACL | p. 490 |
| 1 | Activations linearly predict lawmakers' DW-NOMINATE scores; DW-NOMINATE as the liberal–conservative axis | Kim et al. (2025), ICLR, arXiv:2503.02080v2 | pp. 1–2 |
| 1 | Ideology scale checked against existing measures | Wu et al. (2023) | p. 1 |
| 2 | Social welfare issues structured party conflict since the New Deal; cultural issues first divided parties internally; Republicans more conservative on them in the 1980s and 1990s | Layman & Carsey (2002), AJPS 46(4) | p. 786 |
| 2 | Party positions on abortion unclear at Roe, distinct over the following two decades | Adams (1997), AJPS 41(3) | p. 718 (abstract), p. 721 |
| 2 | One dimension about 80 percent, second about 3 percent (1789–1985); postwar economic votes on the first, civil rights on the second | Poole & Rosenthal (1991), AJPS 35(1) | p. 232 |
| 2 | Polarization can produce misleading evidence of low dimensionality | Aldrich et al. (2014), PA 22(4) | p. 435; pp. 449, 451 |
| 2 | Bill-level voting more complex; first-dimension scores accurate in aggregate | Roberts et al. (2016), APR 44(5) | p. 794; p. 811 |
| 3 | Algorithmic fidelity | Argyle et al. (2023), PA 31(3) | p. 337 |
| 3 | Averages close; 48 percent of coefficients differ | Bisbee et al. (2024), PA 32(4) | p. 401; p. 402 |
| 3 | Typical voters captured, complex factors missed; partisans easy, non-conforming voters hard | von der Heyde et al. (2025), SSCR 44(3) | p. 549; p. 562 |
| 3 | Two-identity personas answer as if they held one | Rennard & Xypolopoulos (2026), arXiv:2608.23005v1 | abstract; Table 1, p. 15 |
| 3 | Abortion and gun-control scales predict ratings and votes better than ideology alone | Wu et al. (2023) | p. 1 |
| 4 | Economic and social ideology both needed; different determinants | Feldman & Johnston (2014), PP 35(3) | p. 337 |
| 4 | Elite beliefs captured by one dimension; cross-pressured citizens call themselves moderate or "don't know" | Treier & Hillygus (2009), POQ 73(4) | p. 679; p. 680; p. 690 |
| 5 | Roll-call data; topic codes | Lewis et al. (2026) Voteview; Jones et al. (2025) Policy Agendas Project | data citations requested by the providers |

## Section 2 (draft v1, 2026-10-19; paper/sections/02_accounts.template.md)

| Paragraph | Sentence (short) | Source | Page |
|---|---|---|---|
| 1 | Cross-pressured citizens call themselves moderate or answer "don't know" on the standard scale | Treier & Hillygus (2009), POQ 73(4) | p. 679 |
| 3 | People torn between competing considerations choose the middle category or skip the question (their expectation, citing Alvarez & Brehm 1995) | Treier & Hillygus (2009) | p. 683 |
| 4 | Two-identity personas explained better by one identity than by the sum | Rennard & Xypolopoulos (2026), arXiv:2608.23005v1 | abstract, p. 1 |
| 4 | Median dominant share 0.88 | Rennard & Xypolopoulos (2026) | p. 7 |
| 4 | No counter-stereotypical penalty once cell size is accounted for; failure uniform across typicality | Rennard & Xypolopoulos (2026) | Figure 7, p. 13; discussion, p. 14 (now cited in Section 7) |
| 4 | Left-authoritarian voters rarely find a matching party; generally privilege economic congruence; depends on issue concerns | Lefkofridi, Wagner & Willmann (2014), WEP 37(1) | p. 65 (abstract) |
| 5 | Without context, LM opinions match some groups better than others; prompting toward a group helps modestly and resolves none of the representativeness problems | Santurkar et al. (2023), ICML, PMLR 202 | pp. 29971–29972 (abstract, contributions); pp. 29975–29976 (Sec. 4.1, Fig. 3); detailed prompting results p. 29977 |

## Section 3 (draft v1, 2026-10-19; paper/sections/03_design.template.md)

| Subsection | Sentence (short) | Source | Page |
|---|---|---|---|
| 3.1 | Qwen2.5-7B-Instruct among the released instruction-tuned sizes | Qwen Team (2025), arXiv:2412.15115v2 | p. 1 (abstract) |
| 3.1 | Llama 3 / 3.1 8B model | Llama Team (2024), arXiv:2407.21783v3 | pp. 1–2 |
| 3.1 | OLMo-2-1124-7B-Instruct among the OLMo 2 Instruct releases | OLMo Team (2025), arXiv:2501.00656v3 | p. 1 |
| 3.2 | Contrastive system-prompt pairs (five), extraction questions, judge score 0–100, difference in mean activations averaged over response tokens | Chen et al. (2025), arXiv:2507.21509v3 | pp. 3–4 (Sec. 2.1–2.2) |
| 3.2 | Layer chosen by steering effectiveness | Chen et al. (2025) | p. 4 (and App. B.4) |
| 3.4 | PAP major topics 1 Macroeconomics, 5 Labor, 6 Education, 7 Environment, 13 Social Welfare | Comparative Agendas Project master codebook page (fetched 2026-10-19) | topic list |
| 3.4 | Roll calls, votes | Lewis et al. (2026), Voteview | data citation |

Differences from Chen et al. stated in the text or to be stated if a reviewer asks: answer pairs are filtered by length and refusal, not by judge score; the judge is Llama-3.1-8B-Instruct, not GPT-4.1-mini. Preprints cited (SSCR discourages preprint citations): the three model reports and Chen et al.; check for published versions before submission.


Corrections after the Codex audit (2026-10-19, TRIAGE A9-A11, A19-A21): Poole and Rosenthal classification gain in percentage points and the 1964 Civil Rights Act example (pp. 232, 239); Treier and Hillygus elite unidimensionality as their premise (p. 680); von der Heyde et al. issue year 2026; Lefkofridi et al. 'generally privilege' (p. 65); Chen et al. procedure adapted (pp. 4, 30); Rennard and Xypolopoulos typicality result after noise correction (pp. 13-14).
