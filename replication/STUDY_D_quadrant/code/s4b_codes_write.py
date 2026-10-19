#!/usr/bin/env python
"""S4b step 1 codes (Claude, same agent, 2026-10-14 19:50-20:00), written before any map position of these people was read.
Same rules as S4; an axis-specific label ('conservative stances on social issues', 'fiscal progressive') sets that axis code,
a general label ('moderate', 'very conservative on nearly all issues') does not."""
import pandas as pd
import dcommon as dc

C = [  # row, econ, econ_quote, social, social_quote
 (1, "L", "a moderate Democrat with strong labor ties", "T", "his conservative stances on social issues"),
 (2, "unclear", "(only general label: one of the most vocal conservatives)", "unclear", ""),
 (3, "unclear", "(liberal on environmental issues: not the economic axis)", "T", "was pro-life and supported prayer in public schools"),
 (4, "unclear", "", "unclear", ""),
 (5, "R", "fiscal conservatism, which included opposition to social programs such as the New Deal and the Great Society", "unclear", ""),
 (6, "unclear", "", "unclear", ""),
 (7, "R", "'Father of Reaganomics'; committed to the doctrine of supply-side economics (OMB, right after his House years)", "unclear", "(2013 amicus brief for same-sex marriage: 33 years after his House years)"),
 (8, "unclear", "(401(k) tax provisions; AEI event on controlling Social Security costs: direction not stated)", "unclear", ""),
 (9, "unclear", "", "unclear", ""),
 (10, "unclear", "", "unclear", ""),
 (11, "L", "considered a social conservative and fiscal progressive populist", "T", "opposed to abortion, gay marriage ... largely on religious grounds"),
 (12, "mixed", "principal sponsor of AmeriCorps; expand Head Start ... vote in 1993 against the Clinton economic plan", "T", "his anti-abortion stance; voting ... conservatively on social issues"),
 (13, "L", "consistently opposed legislation to reduce government spending and cut taxes", "P", "high ratings from NARAL Pro-Choice America and Planned Parenthood"),
 (14, "R", "Garn-St Germain Act ... partially deregulated the savings and loan industry", "T", "strongly pro-life"),
 (15, "R", "board of the Committee for a Responsible Federal Budget (weak)", "P", "pro-abortion rights Republican; voted against the Federal Marriage Amendment"),
 (16, "unclear", "", "unclear", ""),
 (17, "L", "voted against a bill to reduce spending on non-security items", "P", "ran on supporting marriage equality; opposed restricting federal funding of abortions"),
 (18, "L", "Inspired by the legacy of the Great Society (weak)", "unclear", "(only general label: conservative Democrat)"),
 (19, "unclear", "", "mixed", "derided ERA protesters ... later went on to support the Equal Rights Amendment"),
 (20, "L", "focused efforts to fight poverty (as mayor; weak)", "P", "co-sponsored a bill to repeal the Defense of Marriage Act"),
 (21, "L", "voted for ... the Medicare program; called for a uniform nationwide unemployment insurance tax", "unclear", ""),
 (22, "unclear", "(only general label: moderate Republican)", "unclear", ""),
 (23, "unclear", "(only general label: very conservative stances on nearly all issues)", "unclear", ""),
 (24, "L", "his support of the economic stimulus bill and health care bills", "P", "rebuked for participating in a rally supporting Planned Parenthood"),
 (25, "R", "advocate of President Bush's plan for Social Security privatization (after Congress; consistent with budget role)", "unclear", ""),
 (26, "unclear", "(ADA 25% / ACU 69%: general ratings)", "T", "views tracked ... socially conservative Catholic voters (weak)"),
 (27, "R", "Economic: 69% Conservative; interest in replacing the progressive income tax with a flat tax", "T", "opposes abortion except ...; Corker opposes same-sex marriage"),
 (28, "unclear", "", "unclear", ""),
 (29, "unclear", "(expert on the Social Security system: direction not stated)", "unclear", ""),
 (30, "unclear", "(only general label: reputation as a liberal)", "T", "involved in the attempt to pass the Human Life Amendment ... strongly opposed abortion"),
]
df = pd.DataFrame(C, columns=["row", "econ_code", "econ_quote", "social_code", "social_quote"])
sheet = pd.read_csv(dc.OUT / "s4b" / "coding_sheet.csv")
assert sorted(df.row) == sorted(sheet.row)
df["source"] = [f"out/s4b/wiki/row{r:02d}.txt" for r in df.row]
dc.atomic_csv(df, dc.OUT / "s4b" / "coding_codes.csv")
print(df.econ_code.value_counts().to_dict(), df.social_code.value_counts().to_dict())
