"""Structure diagrams for the report, with label-fit checking.

Every label is measured against the box that holds it before the SVG is
emitted, so a wording change that no longer fits fails here instead of
silently overlapping in the published page.

    python -m Centralized_Local_Planner.local_path_replanning.report.diagrams        # writes /tmp/diagrams.json
"""

W_SANS, W_MONO = 0.55, 0.60          # average advance per em
PAD = 26                             # minimum clear space each side of a label


def w(text, size, mono=True):
    return len(text) * size * (W_MONO if mono else W_SANS)


class Fig:
    def __init__(self, vw, vh, label):
        self.vw, self.vh, self.label = vw, vh, label
        self.parts = []

    def box(self, x, y, bw, bh, stroke="currentColor", op=".55", sw=1.4, fill="none", rx=3):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" opacity="{op}"/>')
        return (x, y, bw, bh)

    def text(self, cx, y, s, kind="sub", fill="currentColor", box=None):
        size, mono, cls = {"title": (13, False, "d-title"),
                           "sub": (11.5, True, "d-sub"),
                           "arrow": (11, True, "d-arrow")}[kind]
        if box is not None:
            need = w(s, size, mono) + PAD
            assert need <= box[2], (
                f"'{s}' needs {need:.0f} units, box is {box[2]}")
        self.parts.append(
            f'<text class="{cls}" x="{cx}" y="{y}" fill="{fill}" text-anchor="middle">{s}</text>')

    def arrow(self, x1, x2, y, stroke="currentColor", marker="ah", sw=1.4):
        self.parts.append(
            f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{stroke}" '
            f'stroke-width="{sw}" marker-end="url(#{marker})"/>')

    def render(self, caption):
        defs = ('<defs>'
                '<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                'markerHeight="7" orient="auto-start-reverse">'
                '<path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker>'
                '<marker id="ahs" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                'markerHeight="7" orient="auto-start-reverse">'
                '<path d="M0,0 L10,5 L0,10 z" fill="var(--signal)"/></marker>'
                '<marker id="ahl" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                'markerHeight="7" orient="auto-start-reverse">'
                '<path d="M0,0 L10,5 L0,10 z" fill="var(--learn)"/></marker>'
                '</defs>')
        body = "\n    ".join(self.parts)
        return (f'<figure>\n  <svg viewBox="0 0 {self.vw} {self.vh}" role="img" '
                f'aria-label="{self.label}">\n    {defs}\n    {body}\n  </svg>\n'
                f'  <figcaption>{caption}</figcaption>\n</figure>')


VW, SPINE = 1290, 152
INK3 = "var(--ink-3)"
SIG, LRN = "var(--signal)", "var(--learn)"


# ---------------------------------------------------------------- stop-and-go
def stop_and_go():
    f = Fig(VW, 264, "Stop-and-go: every active AMR is shielded every frame and has only "
                     "two candidates, drive at full speed or hold position.")
    b1 = f.box(12, 104, 250, 96)
    b2 = f.box(352, 66, 270, 172)
    b3 = f.box(712, 96, 260, 112)
    b4 = f.box(1062, 104, 216, 96)
    for x1, x2 in ((262, 346), (622, 706), (972, 1056)):
        f.arrow(x1, x2, SPINE)
    f.text(304, 140, "state", "arrow", INK3)
    f.text(664, 140, "2", "arrow", INK3)
    f.text(1014, 140, "first safe", "arrow", INK3)

    f.text(137, 140, "every active AMR", "title", box=b1)
    f.text(137, 165, "position, speed,", box=b1)
    f.text(137, 185, "5 s worker tube", box=b1)

    f.text(487, 56, "candidate set", "title")
    c1 = f.box(374, 92, 226, 46, SIG, "1", 1.6)
    c2 = f.box(374, 160, 226, 46, SIG, "1", 1.6)
    f.text(487, 121, "GO &#183; full speed", "title", SIG, box=c1)
    f.text(487, 189, "STOP &#183; hold", "title", SIG, box=c2)

    f.text(842, 130, "space&#8211;time shield", "title", box=b3)
    f.text(842, 156, "worker no-go tube", fill=INK3, box=b3)
    f.text(842, 180, "peer reservations", fill=INK3, box=b3)

    f.text(1170, 140, "TRACK", "title", box=b4)
    f.text(1170, 165, "or STOP", fill=INK3, box=b4)
    f.text(1170, 185, "and wait", fill=INK3, box=b4)
    return f.render(
        "<b>Two candidates, nothing in between.</b> The shield runs on every active AMR "
        "every frame; the only question it ever answers is whether full speed is admissible, "
        "and when it is not, the AMR holds position.")


# ------------------------------------------------------------------- speed adjusting
def speed_adjusting():
    f = Fig(VW, 300, "Speed adjusting: AMRs are ordered by time to collision, a four-step speed "
                     "ladder is tried fastest first, and the committed footprint is reserved "
                     "so the next AMR plans around it.")
    b1 = f.box(12, 108, 250, 96)
    b2 = f.box(352, 52, 270, 208)
    b3 = f.box(712, 96, 260, 132)
    b4 = f.box(1062, 108, 216, 96)
    for x1, x2 in ((262, 346), (622, 706), (972, 1056)):
        f.arrow(x1, x2, SPINE)
    f.text(304, 142, "state", "arrow", INK3)
    f.text(664, 142, "4", "arrow", INK3)
    f.text(1014, 142, "first safe", "arrow", INK3)

    f.text(137, 144, "every active AMR", "title", box=b1)
    f.text(137, 169, "sorted by TTC,", box=b1)
    f.text(137, 189, "most urgent first", box=b1)

    f.text(487, 42, "speed ladder", "title")
    for i, lab in enumerate(("1.00 &#215; v", "0.67 &#215; v", "0.33 &#215; v", "0 &#183; stop")):
        y = 68 + i * 48
        c = f.box(392, y, 190, 38, SIG, "1", 1.6)
        f.text(487, y + 25, lab, "title", SIG, box=c)
    f.text(487, 276, "try the fastest first", "arrow", SIG)

    f.text(842, 130, "5 s rail rollout", "title", box=b3)
    f.text(842, 156, "worker hard lobe", fill=INK3, box=b3)
    f.text(842, 180, "peer reservations", fill=INK3, box=b3)
    f.text(842, 204, "static map", fill=INK3, box=b3)

    f.text(1170, 144, "commit speed", "title", box=b4)
    f.text(1170, 172, "reserve", fill=INK3, box=b4)
    f.text(1170, 190, "footprint", fill=INK3, box=b4)

    f.parts.append('<path d="M1170 204 L1170 262 L842 262 L842 232" fill="none" '
                   'stroke="currentColor" stroke-width="1.3" stroke-dasharray="5 4" '
                   'marker-end="url(#ah)" opacity=".75"/>')
    f.text(1006, 280, "the reserved space&#8211;time blocks the next AMR", "arrow", INK3)
    return f.render(
        "<b>A speed ladder and a reservation loop.</b> Each AMR commits to the fastest speed "
        "whose five-second rail rollout stays clear, then reserves that footprint so the next "
        "AMR in the TTC order plans around it. Stopping is the bottom rung, not a separate mode.")


# ------------------------------------------------------------------------ learning-based
def learning_based():
    f = Fig(VW, 336, "Learning-based: a conflict cluster is encoded per AMR, an attention "
                     "policy proposes a forward, lateral and speed action, and that proposal "
                     "competes with seven backups inside the same shield, which commits the "
                     "cheapest safe one.")
    b1 = f.box(12, 116, 210, 100)
    pol = f.box(272, 74, 216, 184, LRN, "1", 1.6, "var(--learn-soft)")
    cand = f.box(538, 62, 236, 208)
    sh = f.box(824, 56, 226, 220)
    b5 = f.box(1100, 116, 178, 100)

    for x1, x2 in ((222, 266), (774, 818), (1050, 1094)):
        f.arrow(x1, x2, SPINE)
    f.arrow(488, 532, SPINE, LRN, "ahl", 1.6)
    f.text(244, 142, "in", "arrow", INK3)
    f.text(510, 142, "action", "arrow", LRN)
    f.text(796, 142, "8", "arrow", INK3)
    f.text(1072, 132, "cheapest", "arrow", INK3)
    f.text(1072, 148, "safe", "arrow", INK3)

    f.text(117, 152, "conflict cluster", "title", box=b1)
    f.text(117, 177, "N &#8804; 4 AMRs,", box=b1)
    f.text(117, 197, "68-d observation", box=b1)

    f.text(380, 62, "attention policy", "title", LRN)
    for i, lab in enumerate(("shared encoder", "self-attention", "actor + critic")):
        y = 92 + i * 52
        r = f.box(292, y, 176, 38, LRN, ".8", 1.2)
        f.text(380, y + 25, lab, fill=LRN, box=r)

    f.text(656, 50, "candidate set", "title")
    c1 = f.box(558, 84, 196, 42, LRN, "1", 1.6)
    f.text(656, 111, "learned proposal", "title", LRN, box=c1)
    c2 = f.box(558, 140, 196, 42, SIG, "1", 1.6)
    f.text(656, 167, "&#177;1 m lateral", "title", SIG, box=c2)
    f.text(656, 212, "slow &#183; stop", fill=INK3, box=cand)
    f.text(656, 234, "shorter &#183; detour", fill=INK3, box=cand)

    f.text(937, 90, "shield &#183; 4 checks", "title", box=sh)
    for i, lab in enumerate(("worker no-go tube", "peer reservations",
                             "static obstacles", "kinematic limits")):
        f.text(937, 122 + i * 26, lab, fill=INK3, box=sh)
    f.parts.append('<line x1="850" y1="242" x2="1024" y2="242" stroke="currentColor" '
                   'stroke-width="1" opacity=".3"/>')
    f.text(937, 264, "rank by cost J", fill=INK3, box=sh)

    f.text(1189, 152, "command", "title", box=b5)
    f.text(1189, 177, "waypoints,", box=b5)
    f.text(1189, 197, "target speed", box=b5)

    f.text(656, 312, "the learned action is one candidate of eight &#8212; "
                     "the shield still has the last word", "arrow", INK3)
    return f.render(
        "<b>The policy proposes, the shield commits.</b> The learned action competes with the "
        "same backup set the deterministic planner uses, inside the same four checks, and the "
        "cheapest safe candidate wins. What the policy adds over the rail planner is the "
        "lateral shift, and a better first guess.")


if __name__ == "__main__":
    import json, pathlib
    out = {"stop_and_go": stop_and_go(), "speed_adjusting": speed_adjusting(),
           "learning_based": learning_based()}
    pathlib.Path("/tmp/diagrams.json").write_text(json.dumps(out))
    print("all labels fit; 3 diagrams written")
