# Shorts transitions v1 (25 Nov 2026, owner OK for NEW Shorts): Nitish Rajput's short cut-based transitions, picked by a Gemini
# side-by-side judge against his real 60-fps frames (E:/mb_builds/shorts_v1/research/TRANSITIONS_PICKED.md).
# Only jobs with "tr": "shorts_v1" use this (make_short.py build writes it for new builds); every other job renders as before.
# Every effect lives INSIDE the head of the incoming shot or the tail of the outgoing shot, so shot lengths, the concat join,
# captions, labels, face circles and SFX times do not move. No overlap, no dissolve, no black stop (Shorts rule).
#
#   kind        Nitish type                   frames @30 (his @60)   where (Shorts)
#   whipsmear   T1 whip smear-in               A 1 f + B 2 f (2-3)     new voiced line / place change; direction alternates
#   crashout    T2 blur crash-out (punch-in)   B 6 f (12-20)           THROUGH / ZOOM / PUNCH requests, "look closer"
#   white       T4 white frame insert          3 f white (6)           FLASH requests, the hook reveal; never twice in 20 s
#   slice       T7 slice glitch                A 4 f + B 4 f (8-11)    GLITCH requests only (tech / system / scam beats)
import re

FPS = 30
KINDS = ("whipsmear", "crashout", "white", "slice")
MAX_PER_MIN = 6.0      # FactTechz / Osm pace: hard cuts dominate (>= 80 % of cuts stay hard cuts in a 34-shot Short)
MIN_GAP = 4.0          # s between two effects
WHITE_GAP = 20.0       # a white frame at most once in 20 s


def kind_of(word):
    """sheet transition word (the way INTO a shot) -> Shorts kind or None (= hard cut)"""
    v = (word or "").upper()
    if "FLASH" in v: return "white"
    if "WHIP" in v: return "whipsmear"
    if "GLITCH" in v or "VHS" in v: return "slice"
    if re.search(r"THROUGH|ZOOM|PUNCH", v): return "crashout"
    return None   # CUT, MATCH, LOOP, MORPH, XRAY, RISE, DISSOLVE, FADE, DIP, WIPE ... stay hard cuts in a Short


def plan(shots, D, log=print):
    """in place: s['tr'] = kind of the cut INTO shot s (None = hard cut), s['tr_dir'] = +1/-1 for whips, s['tr_out'] = kind of the
    cut OUT of it. Priority: sheet requests (FLASH / WHIP / GLITCH / THROUGH), then the first shot of each new voiced line
    (whip smear, every 3rd one a crash-out), inside the budget."""
    budget = max(2, int(round(MAX_PER_MIN * D / 60)))
    cand = []
    for i in range(1, len(shots)):
        s = shots[i]; k = kind_of(s.get("tr_word")) or ("white" if s.get("hit") else None)
        if k: cand.append((0, s["t0"], i, k))
        elif s.get("whoosh"): cand.append((1, s["t0"], i, None))   # first shot of a new voiced line
    kept, n_line, last_white = [], 0, -99.0
    for pr, t, i, k in sorted(cand):
        if len(kept) >= budget: break
        if t < 0.8 or t > D - 1.2: continue                        # not on the very first / last second
        if any(abs(t - shots[j]["t0"]) < MIN_GAP for j, _ in kept): continue
        if k is None:
            n_line += 1; k = "crashout" if n_line % 3 == 0 else "whipsmear"
        if k == "white":
            if t - last_white < WHITE_GAP: k = "whipsmear"
            else: last_white = t
        kept.append((i, k))
    for s in shots: s["tr"] = None; s["tr_out"] = None
    d = 1
    for i, k in sorted(kept):
        shots[i]["tr"] = k
        if k == "whipsmear": shots[i]["tr_dir"] = d; d = -d       # alternate left / right like his montage
        shots[i - 1]["tr_out"] = k
    log("shorts_tr v1: effects", len(kept), "of budget", budget, "|", [(round(shots[i]["t0"], 1), k) for i, k in sorted(kept)])
    return shots


def zoom_mult(kind):
    """extra factor for the scale expression (crash-out: B starts at 270 % and eases out in 0.2 s)"""
    return "*(1+1.7*pow(max(0,1-t/0.2),2.2))" if kind == "crashout" else ""   # 270 % -> 100 % in 0.2 s (Gemini-checked, 25 Nov)


def head(kind, W=1080, H=1920, direction=1):
    """filters for the FIRST frames of the incoming shot (after the zoom crop, before the grade)"""
    if kind == "whipsmear":   # B enters moving: +22 % of width then +5 %, heavy horizontal blur collapsing (2 f @30)
        P = int(round(0.22 * W)) // 2 * 2; p2 = int(round(0.05 * W)) // 2 * 2
        if direction > 0:     # content comes from the left side moving right
            sh = (f"pad=w=iw+{P}:h=ih:x={P}:y=0,fillborders=left={P}:mode=smear,"
                  f"crop={W}:{H}:x='if(eq(n,0),0,if(eq(n,1),{P - p2},{P}))':y=0")
        else:
            sh = (f"pad=w=iw+{P}:h=ih:x=0:y=0,fillborders=right={P}:mode=smear,"
                  f"crop={W}:{H}:x='if(eq(n,0),{P},if(eq(n,1),{p2},0))':y=0")
        return sh + ",dblur=angle=0:radius=70:enable='eq(n,0)',dblur=angle=0:radius=22:enable='eq(n,1)'"
    if kind == "crashout":    # gaussian blur ~17 px at 1080 w, falling with the zoom (stepped, 9 frames)
        return ("gblur=sigma=16:enable='lt(n,2)',gblur=sigma=8:enable='between(n,2,3)',"
                "gblur=sigma=3:enable='eq(n,4)'")
    if kind == "white":       # 6 f @60 = 3 f of solid white, hard on and hard off (no fade)
        return "drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='lt(n,3)'"
    if kind == "slice":       # bands shifted with RGB fringe, 4 f, the first 2 heavy
        return slice_fx("lt(n,4)", W, H)
    return ""


def tail(kind, L, W=1080, H=1920):
    """filters for the LAST frames of the outgoing shot of length L s"""
    if kind == "whipsmear":
        return f"dblur=angle=0:radius=30:enable='gte(t,{L - 1.0 / FPS:.4f})'"
    if kind == "slice":
        return slice_fx(f"gte(t,{L - 4.0 / FPS:.4f})", W, H)
    return ""


def slice_fx(e, W=1080, H=1920):
    bh = H // 14
    return (f"rgbashift=rh=14:bh=-14:enable='{e}',"
            f"swaprect=w={W - 80}:h={bh}:x1=0:y1='mod(n*577,{H - bh})':x2=80:y2='mod(n*331+400,{H - bh})':enable='{e}',"
            f"swaprect=w={W - 160}:h={bh // 2}:x1=160:y1='mod(n*911+200,{H - bh})':x2=0:y2='mod(n*613+900,{H - bh})':enable='{e}',"
            f"drawgrid=w=iw:h=4:t=2:c=black@0.35:enable='{e}'")


def sfx(shots):
    """(time, wav name) list in the old renderer's synthetic SFX style: whoosh peaks ~0.08 s after the cut (GT: 0.05-0.1 s),
    a hit on the white frame, a glitch crackle on the slice"""
    ev = []
    for s in shots:
        k = s.get("tr")
        if k in ("whipsmear", "crashout"): ev.append((s["t0"] - 0.3, "whoosh"))
        elif k == "white": ev += [(s["t0"] - 0.3, "whoosh"), (s["t0"], "hit")]
        elif k == "slice": ev.append((s["t0"] - 4.0 / FPS, "glitch"))
    return ev


GLITCH_WAV = ("anoisesrc=c=white:d=0.28:a=0.7", "highpass=f=1200,acrusher=bits=5:mode=log:aa=0.4,"
              "afade=t=in:d=0.01,afade=t=out:st=0.18:d=0.1,volume=0.6")
