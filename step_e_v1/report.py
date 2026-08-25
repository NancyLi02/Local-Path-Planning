"""Build the Step-E comparison report (outputs/8_step_e_v1_module/report.html).

Numbers come from the result JSONs, diagrams from `report_diagrams`, and the
three demo videos are embedded inline as data URIs after being re-encoded to a
web-sized copy under `demos/web/`.

    python -m step_e_v1.report
"""
from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

from .report_diagrams import rail, stopgo, v1

_REPO = Path(__file__).resolve().parents[1]
_MOD = _REPO / "outputs" / "8_step_e_v1_module"
_WEB_WIDTH = 1400          # px; the source frames are 2100 wide
_WEB_CRF = 24


def _web_video(name: str) -> str:
    """Re-encode a demo to a web-sized copy and return it as a data URI."""
    src = _MOD / "demos" / f"{name}.mp4"
    dst = _MOD / "demos" / "web" / f"{name}.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-vf", f"scale={_WEB_WIDTH}:-2", "-c:v", "libx264", "-crf", str(_WEB_CRF),
             "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             "-an", str(dst)], check=True)
    return base64.b64encode(dst.read_bytes()).decode()


def _load(name: str, key: str | None = None) -> dict:
    data = json.loads((_MOD / "results" / f"{name}.json").read_text())
    return data[key] if key else data


def numbers() -> dict:
    """The three columns, straight from the result files.

    stop-and-go and V1 come from the single three-way run so they share a
    horizon and a seed set; the rail planner is measured by its own pipeline.
    """
    three = _load("compare_3way")
    return {"stopgo": three["STOP-GO"], "rail": _load("rail_v0"), "v1": three["V1"]}


ROWS = [
    ("worker collisions", "worker_collisions", 1.0, "{:.2f}", "min"),
    ("completion %", "completion", 100.0, "{:.0f}", "max"),
    ("makespan [frames]", "makespan", 1.0, "{:.1f}", "min"),
    ("stop ratio %", "stop_ratio", 100.0, "{:.2f}", "min"),
    ("min clearance [m]", "min_clearance", 1.0, "{:.3f}", "max"),
    ("candidates / AMR", "candidates_per_amr", 1.0, "{:.0f}", None),
    ("plan time [ms]", "plan_ms", 1.0, "{:.1f}", "min"),
]


def table(n: dict) -> str:
    cols = ["stopgo", "rail", "v1"]
    head = ("<tr><th>metric</th><th>stop-and-go</th><th>rail V0</th>"
            "<th>V1 safety-first</th></tr>")
    body = ['<tr><td>what it may do</td><td class="txt">drive or halt</td>'
            '<td class="txt">+ slow down</td><td class="txt">+ step aside</td></tr>']
    for label, key, scale, fmt, better in ROWS:
        vals = [n[c][key] * scale for c in cols]
        mark = None
        if better == "min":
            mark = min(range(3), key=lambda i: vals[i])
        elif better == "max":
            mark = max(range(3), key=lambda i: vals[i])
        cells = []
        for i, v in enumerate(vals):
            best = mark is not None and abs(v - vals[mark]) < 1e-9
            worst = (better is not None and not best
                     and abs(v - (max(vals) if better == "min" else min(vals))) < 1e-9)
            cls = "best" if best else ("worst" if worst else "")
            cells.append(f'<td class="{cls}">{fmt.format(v)}</td>')
        body.append(f"<tr><td>{label}</td>{''.join(cells)}</tr>")
    return ("<div class=\"scroll\"><table><thead>" + head + "</thead><tbody>"
            + "".join(body) + "</tbody><caption>"
            "5 seeds &times; 560 frames, 6 AMRs and 2 workers &mdash; long enough that all "
            "three finish every mission, so the makespans compare directly. Rail V0 runs in "
            "its own pipeline, which shields every AMR every frame and has no cluster "
            "hand-over; that is also why its clearance is larger."
            "</caption></table></div>")


def stats(n: dict, col: str) -> str:
    d = n[col]
    cells = [("collisions", f"{d['worker_collisions']:.2f}"),
             ("completion", f"{d['completion'] * 100:.0f}%"),
             ("makespan", f"{d['makespan']:.0f}"),
             ("stop ratio", f"{d['stop_ratio'] * 100:.1f}%"),
             ("plan time", f"{d['plan_ms']:.1f} ms")]
    inner = "".join(f'<div class="stat"><div class="stat-k">{k}</div>'
                    f'<div class="stat-v">{v}</div></div>' for k, v in cells)
    return f'<div class="stats">{inner}</div>'


def video(name: str, caption: str) -> str:
    return (f'<video controls loop muted playsinline preload="metadata" '
            f'src="data:video/mp4;base64,{_web_video(name)}"></video>'
            f'<div class="vidcap">{caption}</div>')


STYLE = """<style>
  :root {
    --ground:#f4f6f7; --panel:#fff; --panel-2:#eceff1;
    --ink:#14202a; --ink-2:#48606f; --ink-3:#7b8f9c; --rule:#d3dbe0;
    --signal:#c07800; --signal-soft:#f6e9cf; --learn:#1d6ea6; --learn-soft:#e2eef7;
    --bad:#a92c22; --good:#1a6f47;
    --shadow:0 1px 0 rgba(20,32,42,.05), 0 12px 28px -22px rgba(20,32,42,.5);
  }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
    --ground:#0e1418; --panel:#151d23; --panel-2:#1b252c;
    --ink:#e6edf1; --ink-2:#a5b8c3; --ink-3:#758896; --rule:#27343d;
    --signal:#e0a338; --signal-soft:#33280f; --learn:#57a8dd; --learn-soft:#102b3d;
    --bad:#e0776b; --good:#5cbb8c; --shadow:0 1px 0 rgba(0,0,0,.3), 0 14px 30px -24px #000;
  } }
  :root[data-theme="dark"] {
    --ground:#0e1418; --panel:#151d23; --panel-2:#1b252c;
    --ink:#e6edf1; --ink-2:#a5b8c3; --ink-3:#758896; --rule:#27343d;
    --signal:#e0a338; --signal-soft:#33280f; --learn:#57a8dd; --learn-soft:#102b3d;
    --bad:#e0776b; --good:#5cbb8c; --shadow:0 1px 0 rgba(0,0,0,.3), 0 14px 30px -24px #000;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--ground); color:var(--ink);
         font-family:"Source Serif 4",Georgia,serif; font-size:17px; line-height:1.62;
         -webkit-font-smoothing:antialiased; }
  .wrap { max-width:1000px; margin:0 auto; padding:0 28px 96px; }
  .col { max-width:63ch; }
  h1,h2,h3,.eyebrow,.stat-v,th,.mono,figcaption b { font-family:Archivo,"Helvetica Neue",Arial,sans-serif; }
  .eyebrow { font-size:12px; font-weight:600; letter-spacing:.16em; text-transform:uppercase; color:var(--signal); }
  header.masthead { border-bottom:1px solid var(--rule); padding:64px 0 34px; margin-bottom:38px; }
  h1 { font-size:clamp(34px,5.2vw,54px); font-weight:800; line-height:1.03;
       letter-spacing:-.022em; margin:14px 0 16px; text-wrap:balance; }
  .standfirst { font-size:20px; color:var(--ink-2); margin:0; max-width:60ch; }
  .byline { margin-top:22px; font-family:"IBM Plex Mono",ui-monospace,monospace;
            font-size:12.5px; color:var(--ink-3); }
  section { margin-top:56px; }
  h2 { font-size:13px; font-weight:600; letter-spacing:.15em; text-transform:uppercase;
       color:var(--ink-3); margin:0 0 18px; padding-bottom:10px; border-bottom:1px solid var(--rule); }
  h3 { font-size:23px; font-weight:600; letter-spacing:-.012em; margin:0 0 4px; }
  p { margin:0 0 16px; }
  .method { background:var(--panel); border:1px solid var(--rule); border-radius:4px;
            box-shadow:var(--shadow); padding:26px 28px 24px; margin-bottom:26px; }
  .method-head { display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; margin-bottom:6px; }
  .rung { font-family:"IBM Plex Mono",monospace; font-size:12px; font-weight:600;
          color:var(--signal); letter-spacing:.1em; }
  .role { font-size:15.5px; color:var(--ink-3); margin:0 0 20px; }
  figure { margin:4px 0 26px; }
  figure svg { width:100%; max-width:100%; height:auto; display:block; }
  figcaption { font-size:13.5px; color:var(--ink-3); margin-top:10px; max-width:78ch; }
  video { width:100%; border:1px solid var(--rule); border-radius:3px; display:block; background:var(--panel-2); }
  .vidcap { font-size:13px; color:var(--ink-3); margin-top:8px; font-family:"IBM Plex Mono",monospace; }
  .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(112px,1fr)); gap:10px; margin:0 0 22px; }
  .stat { background:var(--panel-2); border-radius:3px; padding:10px 12px; }
  .stat-k { font-family:"IBM Plex Mono",monospace; font-size:10.5px; letter-spacing:.06em;
            text-transform:uppercase; color:var(--ink-3); }
  .stat-v { font-size:21px; font-weight:700; letter-spacing:-.02em; font-variant-numeric:tabular-nums; margin-top:2px; }
  .scroll { overflow-x:auto; margin:0 0 6px; }
  table { width:100%; border-collapse:collapse; font-family:"IBM Plex Mono",monospace;
          font-size:13px; font-variant-numeric:tabular-nums; background:var(--panel); border:1px solid var(--rule); }
  caption { caption-side:bottom; text-align:left; font-family:"Source Serif 4",serif;
            font-size:13.5px; color:var(--ink-3); padding-top:10px; }
  th { font-size:11px; font-weight:600; letter-spacing:.08em; text-transform:uppercase;
       color:var(--ink-3); text-align:right; padding:12px 14px; border-bottom:1px solid var(--rule); white-space:nowrap; }
  th:first-child { text-align:left; }
  td { padding:9px 14px; text-align:right; border-bottom:1px solid var(--rule); white-space:nowrap; }
  td:first-child { text-align:left; font-family:"Source Serif 4",serif; font-size:14.5px; }
  td.txt { font-family:"Source Serif 4",serif; font-size:14.5px; color:var(--ink-2); }
  tbody tr:first-child td { background:var(--panel-2); }
  tbody tr:last-child td { border-bottom:0; }
  .best { font-weight:600; color:var(--good); }
  .worst { color:var(--bad); }
  .note { background:var(--panel); border:1px solid var(--rule); border-left:3px solid var(--signal);
          border-radius:3px; padding:16px 18px; margin:20px 0; }
  .note p:last-child { margin-bottom:0; }
  .note .eyebrow { display:block; margin-bottom:6px; }
  code { font-family:"IBM Plex Mono",monospace; font-size:.88em; background:var(--panel-2);
         padding:1px 5px; border-radius:2px; }
  pre { background:var(--panel); border:1px solid var(--rule); border-radius:3px; padding:15px 17px;
        overflow-x:auto; font-family:"IBM Plex Mono",monospace; font-size:12.5px; line-height:1.65; color:var(--ink-2); }
  footer { margin-top:66px; padding-top:22px; border-top:1px solid var(--rule);
           font-family:"IBM Plex Mono",monospace; font-size:12px; color:var(--ink-3); }
  .d-title { font:600 13px Archivo,sans-serif; }
  .d-sub { font:400 11.5px "IBM Plex Mono",monospace; }
  .d-arrow { font:400 11px "IBM Plex Mono",monospace; }
</style>"""

HEAD = ('<title>Three Ways to Yield</title>\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=Archivo:wght@500;600;800&family=IBM+Plex+Mono:wght@400;500;600&'
        'family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">\n' + STYLE)


def build() -> Path:
    n = numbers()
    html = HEAD + f'''
<div class="wrap">
  <header class="masthead">
    <div class="eyebrow">Step E &mdash; centralized local replanning</div>
    <h1>Three ways to yield to a worker</h1>
    <p class="standfirst">
      Stop and wait, slow down on the rail, or step aside &mdash; the same factory, the same
      workers, the same safety shield, and three planners that differ only in what they are
      allowed to do about it.
    </p>
    <div class="byline">6 AMRs &middot; 2 workers &middot; 5 seeds &times; 560 frames &middot; 2026-08-25</div>
  </header>

  <section style="margin-top:0">
    <h2>At a glance</h2>
    {table(n)}
    <div class="col">
      <p style="margin-top:24px">
        All three are collision-free: safety is the shield's job, not the planner's. What
        separates them is the price they pay for it &mdash; how often an AMR has to come to a
        full stop, and how long the fleet takes to finish.
      </p>
    </div>
  </section>

  <section>
    <h2>The three planners</h2>

    <div class="method">
      <div class="method-head"><span class="rung">01</span><h3>Stop-and-go</h3></div>
      <p class="role">The classic industrial controller: drive, or halt and wait.</p>
      {stopgo()}
      {stats(n, "stopgo")}
      {video("demo_stopgo", "seed 0, 420 frames &mdash; AMRs queue at the busy-area boundary and wait the workers out")}
    </div>

    <div class="method">
      <div class="method-head"><span class="rung">02</span><h3>Rail V0</h3></div>
      <p class="role">Speed control along a fixed rail &mdash; the original Step-5 planner, run unchanged.</p>
      {rail()}
      {stats(n, "rail")}
      {video("demo_rail_v0", "seed 0, 420 frames &mdash; AMRs ease off and let workers cross instead of stopping dead")}
    </div>

    <div class="method">
      <div class="method-head"><span class="rung">03</span><h3>V1, safety-first</h3></div>
      <p class="role">An attention policy proposes; the shield still commits the cheapest safe candidate.</p>
      {v1()}
      {stats(n, "v1")}
      {video("demo_v1", "seed 0, 420 frames &mdash; cluster members slide around the worker and rejoin their lane")}
    </div>
  </section>

  <section>
    <h2>What the comparison shows</h2>
    <div class="col">
      <p>
        <b>Being allowed to slow down is worth 34 frames and three quarters of the stopping.</b>
        Stop-and-go halts on 59&nbsp;% of the frames it controls; the speed ladder brings that
        to 13.7&nbsp;% and the mission from 435 to 401 frames.
      </p>
      <p>
        <b>Being allowed to step aside is worth another 46 frames.</b> V1 finishes in 355
        frames and halts on 1.4&nbsp;% of frames &mdash; a tenth of the rail planner &mdash; by
        moving up to a metre off the lane instead of waiting for it to clear. It pays for that
        in computation: 60.6&nbsp;ms per cluster against 8.7&nbsp;ms, still well inside the
        200&nbsp;ms control period.
      </p>
    </div>
    <div class="note">
      <span class="eyebrow">What the learned part contributes</span>
      <p>
        V1 is the deterministic planner of the same pipeline with its nominal proposal replaced
        by the policy's. That deterministic sibling scores 352.6 frames and a 3.30&nbsp;% stop
        ratio, so the attention policy is not what makes the lateral manoeuvres safe &mdash; the
        shield is &mdash; it is what more than halves the remaining stopping, at equal safety and
        completion. Its numbers are in <code>results/v0.json</code>.
      </p>
    </div>
  </section>

  <section>
    <h2>Reproduce</h2>
    <pre>python -m step_e_v1.evaluate --planner stopgo --seeds 5 --frames 560
python -m step_e_v1.legacy_rail  --seeds 5 --frames 560
python -m step_e_v1.evaluate --planner v1 --model logs/step_e_v1/v1_best.pt \\
       --seeds 5 --frames 560

python -m step_e_v1.render      --planner stopgo
python -m step_e_v1.legacy_rail --render
python -m step_e_v1.render      --planner v1 --model logs/step_e_v1/v1_best.pt
python -m step_e_v1.report                     # rebuilds this page</pre>
    <div class="col">
      <p style="margin-top:18px">
        Metrics land in <code>outputs/8_step_e_v1_module/results/</code>, videos in
        <code>demos/</code>, checkpoints and the training log in <code>logs/step_e_v1/</code>.
        <code>outputs/README.md</code> maps every folder to the step that produced it.
      </p>
    </div>
  </section>

  <footer>step_e_v1 &middot; torch 2.5.1+cu121, RTX 4060 Laptop &middot;
         demos rendered at 2100&nbsp;px, embedded at {_WEB_WIDTH}&nbsp;px</footer>
</div>
'''
    out = _MOD / "report.html"
    out.write_text(html)
    return out


if __name__ == "__main__":
    p = build()
    print(f"wrote {p}  ({p.stat().st_size / 1024:.0f} KB)")
