"""Study B shared definitions: issues, paths, roll-call selection (keys defined once; Rhee rule).

Imports Study A's measurement module (uncertainty_map/common.py) so persona wording, projection and estimation are one code path.
"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIM = HERE.parent
sys.path.insert(0, str(KIM / "uncertainty_map"))
import common as cm  # noqa: E402

OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
DATA = HERE / "issue_data"
PAP_CSV = KIM / "data" / "agendas" / "rcv_voteview_v3_4.csv"
VV_ROLLCALLS = KIM / "data" / "voteview_all" / "HSall_rollcalls.csv"
VV_VOTES = KIM / "data" / "voteview_all" / "HSall_votes.csv"
VV_MEMBERS = KIM / "data" / "HSall_members.csv"
CONGRESSES = range(80, 119)
PILOT_CONGRESSES = list(range(80, 119, 2))          # even Congresses 80-118 (DESIGN section 10)
Q_EXTRACT, Q_HELDOUT, Q_EVAL = range(0, 20), range(20, 25), range(25, 30)

# judge options: A = progressive, B = conservative (score_B = conservative; same direction as Kim's + sign)
ISSUES = {
    "civil_rights": dict(label="Race and civil rights", pap_sub=[201, 206], regex=None,
                         judge=("stronger federal protection of civil rights and action against racial discrimination",
                                "limiting federal civil-rights enforcement and leaving such matters to states, localities or individuals")),
    "abortion": dict(label="Abortion", pap_sub=None, regex=r"ABORTION|UNBORN",
                     judge=("protecting legal access to abortion", "restricting abortion")),
}
# roll-call-only scales (no LLM axis): used by CP1 (0_protocol/PROTOCOL_CP1_crosspressure_entropy.md)
RC_ONLY = {"economic": dict(label="Economic (PAP 1, 5, 6, 7, 13)", pap_major=[1, 5, 6, 7, 13], pap_sub=None, regex=None)}

def issue_data(issue):
    d = json.load(open(DATA / f"data_{issue}.json", encoding="utf-8"))
    assert len(d["instructions"]) == 5 and len(d["questions"]) == 30, issue
    return d

def ordinary_system(name, occ, state, year):
    """Study A step 9/20 wording, verbatim."""
    return f"You are {name}, who works as {occ} in {state} in {year}. Answer the question in the first person, as {name} would."

OCC = ["a nurse", "a software engineer", "an accountant", "an electrician", "a high school teacher", "a dental hygienist",
       "a truck driver", "a pharmacist", "a restaurant manager", "a librarian"]
NAMES = ["Emily Johnson", "Michael Smith", "Maria Lopez", "David Lee", "Sarah Williams", "James Brown", "Jessica Davis", "Robert Miller",
         "Ashley Wilson", "Daniel Moore", "Jennifer Taylor", "Christopher Anderson", "Amanda Thomas", "Matthew Jackson", "Stephanie White",
         "Joshua Harris", "Laura Martin", "Andrew Thompson", "Rachel Garcia", "Kevin Martinez", "Nicole Robinson", "Brian Clark",
         "Megan Rodriguez", "Jason Lewis", "Lauren Walker", "Ryan Hall", "Hannah Allen", "Justin Young", "Olivia King", "Eric Wright"]
PERSONS = [(NAMES[k], o, s) for k, (o, s) in enumerate([(o, s) for o in OCC for s in ["Ohio", "California", "Georgia"]])]  # Study A order
