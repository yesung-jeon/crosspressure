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
