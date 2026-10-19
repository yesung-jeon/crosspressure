#!/usr/bin/env python
"""S4c step 1 codes for the 30 people not coded in S4b (Claude, same agent, 2026-10-14 20:00-20:10), written before any map
position of these people was read. Same rules as s4b_codes_write.py."""
import pandas as pd
import dcommon as dc

C = [  # row, econ, econ_quote, social, social_quote
 (1, "unclear", "", "P", "He opposed the overturning of Roe v. Wade"),
 (2, "L", "voting in favor of many of President Lyndon B. Johnson's 'Great Society' initiatives, such as ... Medicare", "unclear", ""),
 (3, "R", "opposes an income tax increase, opposes federal spending", "T", "generally supports anti-abortion legislation"),
 (4, "R", "Club for Growth named Loudermilk a 'defender of economic freedom'; supports ... a flat tax system", "T", "Loudermilk is anti-abortion and believes that life starts at conception"),
 (5, "mixed", "supported ... SCHIP law and the Recovery Act; member of the Blue Dog Coalition, a fiscally conservative Democratic caucus", "unclear", ""),
 (6, "L", "member of Painters' Union Local 195; supported the United Steelworkers's unionization effort", "T", "Beard supported the anti-abortion movement"),
 (7, "unclear", "", "mixed", "supported the Equal Rights Amendment ... but was critical of abortion rights; endorsed voluntary school prayers"),
 (8, "R", "co-sponsored the Labor Management Reporting and Disclosure Act ... which regulates the internal affairs of labor unions (weak)", "unclear", ""),
 (9, "R", "passionate supporter of trucking deregulation; floor management credited with killing Clinton's 1993 health care bill", "P", "introduced the Senate's first abortion legalization bill"),
 (10, "R", "voted for the American Health Care Act, which would have repealed Obamacare", "mixed", "ties to the anti-abortion Susan B. Anthony group; voted for the Respect for Marriage Act"),
 (11, "R", "signer of the Taxpayer Protection Pledge; voted against the Emergency Economic Stabilization Act", "T", "Planned Parenthood investigation started in response to ... the anti-abortion organization Americans United for Life"),
 (12, "L", "early support from the Central Labor Council; tried to restrict local governments' ability to shed pension programs", "P", "He opposed the 2022 overturning of Roe v. Wade"),
 (13, "unclear", "", "unclear", ""),
 (14, "unclear", "", "P", "100% rating from NARAL Pro-Choice America"),
 (15, "R", "voted against the federal budget all 12 years ... because the proposed budget ran a deficit", "T", "pro-life; harshly criticized ... appointment of openly gay geneticist"),
 (16, "unclear", "", "P", "signatory to an amicus curiae brief ... in support of same-sex marriage (2013)"),
 (17, "unclear", "", "unclear", ""),
 (18, "mixed", "called a strong fiscal conservative ... closer to the left wing of his party on certain economic issues", "T", "strongly opposes legal abortion; supported a Constitutional Amendment that would ban ... same-sex marriages"),
 (19, "mixed", "helped defeat the Clinton health care plan; driving force behind a large tax increase (1982)", "unclear", ""),
 (20, "mixed", "a moderate stance on taxes and government assistance to the needy; legislation to expand Medicaid coverage", "P", "He was pro-choice on abortion ... opposed ... school prayer"),
 (21, "L", "most often earned him a 100% rating from the AFL-CIO", "P", "Sherman is pro-choice ... supports same-sex marriage"),
 (22, "R", "a bill to repeal the Patient Protection and Affordable Care Act; endorsed by the Club for Growth", "T", "described as ... a social conservative"),
 (23, "unclear", "", "unclear", ""),
 (24, "L", "increasing the minimum wage, the Paid Family Leave Act (state assembly)", "P", "arrested at an abortion rights rally outside the Supreme Court"),
 (25, "R", "Laxalt governed Nevada as a fiscal conservative (before the Senate)", "unclear", ""),
 (26, "L", "On other economic issues, he was more allied with his party's populist wing (occasionally voted with Republicans on taxes)", "T", "strongly opposed abortion; supported school prayer in public schools"),
 (27, "unclear", "", "unclear", ""),
 (28, "unclear", "", "mixed", "opposes abortion; opposition to the Federal Marriage Amendment"),
 (29, "R", "a prominent advocate of free enterprise and favored cutting the budget", "T", "opposed ... feminism, gay rights ... access to abortions"),
 (30, "R", "a reputation as a social liberal and fiscal conservative; pro-business economic policies", "P", "a reputation as a social liberal"),
]
df = pd.DataFrame(C, columns=["row", "econ_code", "econ_quote", "social_code", "social_quote"])
sheet = pd.read_csv(dc.OUT / "s4c" / "coding_sheet.csv")
assert sorted(df.row) == sorted(sheet.row)
df["source"] = [f"out/s4c/wiki/row{r:02d}.txt" for r in df.row]
dc.atomic_csv(df, dc.OUT / "s4c" / "coding_codes.csv")
print(df.econ_code.value_counts().to_dict(), df.social_code.value_counts().to_dict())
