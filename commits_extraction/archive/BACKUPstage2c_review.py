#!/usr/bin/env python3
"""
VeriDelta — Stage 2c: validate the design/verification labels WITH THE DIFF
stage2c_review.py

The LLM and the human annotator see exactly the same evidence for each commit:
subject, message, the list of changed HDL files, and the git diff of those files
(capped, see --diff-cap).  Neither sees the path-rule label.

  sample   Draw a stratified pilot sample (default 25 rule-confident + 25
           needs-review commits, split across repos), extract each commit's diff
           from the local clones in --sources-dir, and write
             review_sample.csv   full table (rule labels, diff file names)
             diffs/*.diff        the exact diff text shown to both raters
             review_pack.html    BLIND review page for you (open in a browser)
             python3 stage2c_review.py sample

  llm      Label the same commits with Claude Haiku (needs ANTHROPIC_API_KEY).
           Fills llm_label / llm_confidence / llm_reason in review_sample.csv.
           --scope review does every needs_review commit instead.
             python3 stage2c_review.py llm

  report   After you export your labels from the HTML page (human_labels.csv):
             python3 stage2c_review.py report --annotations human_labels.csv
           Accuracy with 95% Wilson intervals, Cohen's kappa, confusion matrices
           for rules-vs-human and LLM-vs-human, accuracy by confidence level.

Labels:  design = changes RTL/design-side sources (incl. RTL generators and
         register specs);  verification = changes testbench / UVM / assertions /
         coverage / DV infrastructure;  both = clearly both;  neither = no
         functional or structural content (license headers, whitespace,
         comment-only edits).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict

MODEL = "claude-haiku-4-5-20251001"
LABELS = ["design", "verification", "both", "neither"]
SHORT = {"d": "design", "v": "verification", "b": "both", "n": "neither"}
SAMPLE_FIELDS = ["repo", "hash", "subject", "body", "stratum", "rule_label", "n_design",
                 "n_verification", "n_unknown", "files", "diff_file", "human_label",
                 "llm_label", "llm_confidence", "llm_reason", "llm_votes", "llm_agreement"]


# ── helpers ───────────────────────────────────────────────────────────────────

def read_csv(path):
    csv.field_size_limit(1 << 30)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_files_by_commit(stage2_dir):
    by = defaultdict(list)
    for r in read_csv(os.path.join(stage2_dir, "selector_files.csv")):
        by[(r["repo"], r["hash"])].append(r)
    return by


def files_text(rows, limit=15):
    mark = {"DESIGN": "D", "VERIFICATION": "V", "UNKNOWN": "?"}
    out = [f"{mark[r['side']]} {r['status']} {r['path']}" for r in rows[:limit]]
    if len(rows) > limit:
        out.append(f"… +{len(rows) - limit} more")
    return "\n".join(out)


# ── diff extraction (identical evidence for the LLM and the human) ────────────

PER_FILE_CAP = 120


def commit_diff(sources_dir, repo, h, selector_paths, cap):
    """
    Diff of the commit restricted to its selector files, capped at `cap` lines in
    total and PER_FILE_CAP per file.  Files that do not fit are listed by name.
    """
    p = subprocess.run(["git", "-C", os.path.join(sources_dir, repo), "-c", "core.quotepath=false",
                        "show", "--format=", "--no-color", "-M", "--unified=3", "--no-ext-diff", h],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        return f"(git show failed: {p.stderr.strip()[:200]})"
    blocks, cur = [], None
    for line in p.stdout.splitlines():
        if line.startswith("diff --git "):
            cur = [line]
            blocks.append(cur)
        elif cur is not None:
            cur.append(line)
    wanted = set(selector_paths)
    out, used, skipped = [], 0, []
    for b in blocks:
        m = re.match(r"diff --git a/(.*) b/(.*)$", b[0])
        path = m.group(2) if m else ""
        if path not in wanted:
            continue
        if used >= cap:
            skipped.append(path)
            continue
        room = min(PER_FILE_CAP, cap - used)
        shown = b[:room]
        out.extend(shown)
        used += len(shown)
        if len(b) > room:
            out.append(f"… ({len(b) - room} more diff lines in this file not shown)")
    if skipped:
        out.append("… files not shown (diff cap reached): " + ", ".join(skipped[:30]))
    n_other = len(blocks) - len([b for b in blocks if re.match(r"diff --git a/.* b/(.*)$", b[0]) and
                                 re.match(r"diff --git a/.* b/(.*)$", b[0]).group(1) in wanted])
    if n_other:
        out.append(f"({n_other} non-HDL files also changed in this commit; not shown)")
    return "\n".join(out) if out else "(no diff text for the selector files)"


def diff_path(stage2_dir, repo, h):
    return os.path.join(stage2_dir, "diffs", f"{repo}__{h[:12]}.diff")


def ensure_diff(a, repo, h, by_files):
    path = diff_path(a.stage2_dir, repo, h)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        paths = [r["path"] for r in by_files[(repo, h)]]
        with open(path, "w", encoding="utf-8") as f:
            f.write(commit_diff(a.sources_dir, repo, h, paths, a.diff_cap))
    with open(path, encoding="utf-8") as f:
        return f.read()


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def kappa(pairs):
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[l] * cb[l] for l in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


# ── sample ────────────────────────────────────────────────────────────────────

def cmd_sample(a):
    selected = read_csv(os.path.join(a.stage2_dir, "commits_selected.csv"))
    bodies = {(r["repo"], r["hash"]): r["body"]
              for r in read_csv(os.path.join(a.stage1_dir, "commits.csv"))}
    by_files = load_files_by_commit(a.stage2_dir)
    repos = sorted({r["repo"] for r in selected})
    rng = random.Random(a.seed)
    out = []
    for stratum, n_total in (("confident", a.n_confident), ("review", a.n_review)):
        pool = [r for r in selected if (r["needs_review"] == "1") == (stratum == "review")]
        cand = {repo: [r for r in pool if r["repo"] == repo] for repo in repos}
        quota = {repo: n_total // len(repos) + (1 if i < n_total % len(repos) else 0)
                 for i, repo in enumerate(repos)}
        take = {repo: min(len(cand[repo]), quota[repo]) for repo in repos}
        short = n_total - sum(take.values())                        # repo too small → give its slots away
        for repo in repos:
            extra = min(short, len(cand[repo]) - take[repo])
            take[repo] += extra
            short -= extra
        for repo in repos:
            for r in rng.sample(cand[repo], take[repo]):
                k = (r["repo"], r["hash"])
                out.append({
                    "repo": r["repo"], "hash": r["hash"], "subject": r["subject"],
                    "body": bodies.get(k, "")[:600], "stratum": stratum,
                    "rule_label": r["label"], "n_design": r["n_design"],
                    "n_verification": r["n_verification"], "n_unknown": r["n_unknown"],
                    "files": files_text(by_files[k]), "human_label": "",
                    "llm_label": "", "llm_confidence": "", "llm_reason": "",
                })
    rng.shuffle(out)                                   # do not annotate stratum by stratum
    if not os.path.isdir(os.path.join(a.sources_dir, repos[0])):
        print(f"{a.sources_dir}/{repos[0]} not found — pass --sources-dir pointing at the Stage 1 clones")
        return 1
    for r in out:
        r["diff_file"] = os.path.basename(diff_path(a.stage2_dir, r["repo"], r["hash"]))
        r["_diff"] = ensure_diff(a, r["repo"], r["hash"], by_files)
    path = os.path.join(a.stage2_dir, "review_sample.csv")
    write_csv(path, out, SAMPLE_FIELDS)
    hpath = os.path.join(a.stage2_dir, "review_pack.html")
    with open(hpath, "w", encoding="utf-8") as f:
        f.write(build_html(out))
    print(f"wrote {len(out)} rows → {path}  (full table, rule labels, diff file names)")
    print(f"wrote {len(out)} diffs → {os.path.join(a.stage2_dir, 'diffs')}")
    print(f"wrote BLIND review page → {hpath}   (open it in a browser; label, then Export CSV)")
    print("Then:  python3 stage2c_review.py report --annotations ~/Downloads/human_labels.csv")
    return 0


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>VeriDelta commit review</title>
<style>
body{font:14px/1.45 system-ui,sans-serif;margin:0;background:#f6f7f9;color:#1b1f24}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #d0d7de;padding:8px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
main{max-width:1100px;margin:0 auto;padding:12px 16px 120px}
h2{margin:6px 0;font-size:17px} .meta{color:#57606a;font-size:12px}
pre{background:#fff;border:1px solid #d0d7de;border-radius:6px;padding:8px;overflow:auto;max-height:55vh;font:12px/1.35 ui-monospace,Menlo,monospace}
pre.msg{max-height:14vh;white-space:pre-wrap} pre.files{max-height:16vh}
.add{background:#e6ffec}.del{background:#ffebe9}.hunk{background:#ddf4ff;color:#0550ae}.hdr{color:#57606a;font-weight:600}
button{font:inherit;padding:6px 12px;border:1px solid #d0d7de;border-radius:6px;background:#fff;cursor:pointer}
button.on{background:#0969da;color:#fff;border-color:#0969da} .bar{position:fixed;bottom:0;left:0;right:0;background:#fff;border-top:1px solid #d0d7de;padding:10px 16px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
textarea{width:260px;height:34px;font:inherit} .done{color:#1a7f37;font-weight:600}
</style></head><body>
<header><b>VeriDelta commit review</b><span id="prog"></span>
<button onclick="go(-1)">&larr; prev</button><button onclick="go(1)">next &rarr;</button>
<button onclick="nextTodo()">next unlabeled</button><button onclick="exportCsv()">Export CSV</button>
<span class="meta">keys: d v b n = label &nbsp; 1 2 3 = confidence (low/med/high) &nbsp; &larr;/&rarr; = navigate</span></header>
<main><h2 id="subj"></h2><div class="meta" id="meta"></div>
<h4>Message</h4><pre class="msg" id="body"></pre>
<h4>Changed HDL files</h4><pre class="files" id="files"></pre>
<h4>Diff (HDL files only, capped)</h4><pre id="diff"></pre></main>
<div class="bar"><span>Label:</span>
<button data-l="design">design (d)</button><button data-l="verification">verification (v)</button>
<button data-l="both">both (b)</button><button data-l="neither">neither (n)</button>
<span>Confidence:</span><button data-c="1">low (1)</button><button data-c="2">med (2)</button><button data-c="3">high (3)</button>
<textarea id="note" placeholder="note (optional)"></textarea><span id="state"></span></div>
<script>
const DATA=__DATA__;
const KEY="veridelta_review_v1_"+DATA.map(d=>d.hash.slice(0,6)).join("").slice(0,60);
let ans={};try{ans=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let i=0;
const $=id=>document.getElementById(id);
function save(){try{localStorage.setItem(KEY,JSON.stringify(ans))}catch(e){}}
function render(){
 const d=DATA[i],a=ans[d.hash]||{};
 $("subj").textContent=d.subject;$("meta").textContent=d.repo+"  "+d.hash.slice(0,12)+"  ("+(i+1)+"/"+DATA.length+")";
 $("body").textContent=d.body||"(no message body)";$("files").textContent=d.files;
 const pre=$("diff");pre.textContent="";
 for(const ln of d.diff.split("\n")){const s=document.createElement("div");s.textContent=ln||" ";
  s.className=ln.startsWith("+")&&!ln.startsWith("+++")?"add":ln.startsWith("-")&&!ln.startsWith("---")?"del":ln.startsWith("@@")?"hunk":ln.startsWith("diff ")?"hdr":"";pre.appendChild(s)}
 pre.scrollTop=0;
 document.querySelectorAll("[data-l]").forEach(b=>b.classList.toggle("on",a.label===b.dataset.l));
 document.querySelectorAll("[data-c]").forEach(b=>b.classList.toggle("on",a.conf===b.dataset.c));
 $("note").value=a.note||"";
 const n=Object.values(ans).filter(x=>x.label).length;
 $("prog").textContent=n+"/"+DATA.length+" labeled";$("state").innerHTML=a.label?'<span class="done">saved</span>':"";
}
function set(k,v){const d=DATA[i];ans[d.hash]=Object.assign(ans[d.hash]||{},{[k]:v});save();render()}
function go(n){i=Math.max(0,Math.min(DATA.length-1,i+n));render()}
function nextTodo(){for(let k=1;k<=DATA.length;k++){const j=(i+k)%DATA.length;if(!(ans[DATA[j].hash]||{}).label){i=j;render();return}}}
document.querySelectorAll("[data-l]").forEach(b=>b.onclick=()=>set("label",b.dataset.l));
document.querySelectorAll("[data-c]").forEach(b=>b.onclick=()=>set("conf",b.dataset.c));
$("note").oninput=e=>{const d=DATA[i];ans[d.hash]=Object.assign(ans[d.hash]||{},{note:e.target.value});save()};
document.addEventListener("keydown",e=>{if(e.target.tagName==="TEXTAREA")return;
 const m={d:"design",v:"verification",b:"both",n:"neither"};
 if(m[e.key])set("label",m[e.key]);else if("123".includes(e.key)&&e.key)set("conf",e.key);
 else if(e.key==="ArrowRight")go(1);else if(e.key==="ArrowLeft")go(-1)});
function q(s){return '"'+String(s==null?"":s).replace(/"/g,'""')+'"'}
function exportCsv(){const rows=["repo,hash,human_label,human_confidence,note"];
 for(const d of DATA){const a=ans[d.hash]||{};rows.push([d.repo,d.hash,a.label||"",a.conf||"",a.note||""].map(q).join(","))}
 const blob=new Blob([rows.join("\n")],{type:"text/csv"});const l=document.createElement("a");
 l.href=URL.createObjectURL(blob);l.download="human_labels.csv";l.click()}
render();
</script></body></html>"""


def build_html(rows):
    data = [{"repo": r["repo"], "hash": r["hash"], "subject": r["subject"], "body": r["body"],
             "files": "\n".join(l[2:] for l in r["files"].splitlines()), "diff": r["_diff"]}
            for r in rows]
    return HTML.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))


# ── llm ───────────────────────────────────────────────────────────────────────

PROMPT = """You label commits in the hardware repository "{repo}".

Decide what kind of change this commit makes to the HDL (SystemVerilog/Verilog) sources.
Judge by the CONTENT OF THE CHANGED LINES, not by the folder name and not by the reason for the commit.
- design: the changed lines are RTL / design code: modules, packages, parameters, defines, register RTL, and generator templates (.tpl) that generate RTL.
- verification: the changed lines are testbench, UVM environment / sequences / scoreboards, DPI or other testbench libraries, assertions (including assertion macros or assert statements added inside RTL files), bind files, coverage, formal properties, and templates that generate DV code. Autogenerated DV files count as verification.
- both: the commit meaningfully changes both kinds of lines (for example an RTL change plus a testbench change; a vendored update that adds both RTL and testbench files).
- neither: no functional or structural change: whitespace or re-alignment only, comment-only edits, license/copyright headers.
A template takes the side of the code it generates. Vendored third-party updates are judged by content like any other commit.

Commit subject: {subject}
Commit message body: {body}
Changed HDL files (status path):
{files}

Diff of those files (truncated; judge from the diff, not only from the file names):
{diff}

Confidence scale (be strict; "high" must be earned):
- high: the diff clearly fits exactly one label.
- medium: it fits one label, but another is plausible (mixed file kinds, unclear role of a file).
- low: two labels are about equally plausible, or the diff is truncated or uninformative.

Respond with ONLY a JSON object:
{{"label": "design|verification|both|neither", "confidence": "high|medium|low", "reason": "<max 20 words>"}}"""


CONF_WORDS = {"high": 0.9, "medium": 0.6, "low": 0.3}


def conf_value(x):
    """'high'/'medium'/'low' -> 0.9/0.6/0.3; a bare number is kept (older cached results)."""
    if isinstance(x, str) and x.strip().lower() in CONF_WORDS:
        return CONF_WORDS[x.strip().lower()]
    return float(x)


_CACHE_LOCK = threading.Lock()
_TEMPERATURE_OK = True          # flips to False the first time the SDK/model rejects it


def llm_label(client, row, cache, cache_path, run=0, temperature=None):
    global _TEMPERATURE_OK
    paths = "\n".join(l[2:] for l in row["files"].splitlines())      # drop the D/V/? side marker
    prompt = PROMPT.format(repo=row["repo"], subject=row["subject"],
                           body=(row.get("body") or "(none)")[:600], files=paths,
                           diff=row.get("_diff") or "(not available)")
    tag = (f"|run{run}" if run else "") + (f"|t{temperature}" if temperature is not None else "")
    key = hashlib.sha256((MODEL + tag + prompt).encode()).hexdigest()
    if key in cache:
        return cache[key]
    kwargs = dict(model=MODEL, max_tokens=200, messages=[{"role": "user", "content": prompt}])
    if temperature is not None and _TEMPERATURE_OK:
        kwargs["temperature"] = temperature
    for attempt in range(6):
        try:
            msg = client.messages.create(**kwargs)
            break
        except TypeError as e:
            # newer SDKs reject `temperature`; drop it and retry (the on-disk cache
            # keeps results reproducible even without temperature 0)
            if "temperature" in str(e) and "temperature" in kwargs:
                kwargs.pop("temperature")
                _TEMPERATURE_OK = False
                print("  note: this SDK/model rejects `temperature`; using default sampling", flush=True)
                continue
            raise
        except Exception as e:                                      # retry transient errors only
            code = getattr(e, "status_code", None)
            if attempt == 5 or code not in (408, 429, 500, 502, 503, 504, 529):
                raise
            time.sleep(2 ** attempt)
    text = msg.content[0].text
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group()) if m else {}
        lab = str(d.get("label", "")).lower()
        res = {"label": lab if lab in LABELS else "", "confidence": conf_value(d.get("confidence", 0)),
               "reason": str(d.get("reason", ""))[:200]}
    except (ValueError, TypeError):
        res = {"label": "", "confidence": 0.0, "reason": "unparseable: " + text[:80]}
    with _CACHE_LOCK:
        cache[key] = res
        with open(cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"k": key, **res}) + "\n")
    return res


def cmd_llm(a):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set")
        return 1
    import anthropic                                                # lazy
    client = anthropic.Anthropic()
    cache_path = os.path.join(a.stage2_dir, "llm_cache.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path, encoding="utf-8"):
            d = json.loads(line)
            cache[d.pop("k")] = d

    by_files = load_files_by_commit(a.stage2_dir)
    if a.scope == "sample":
        path = os.path.join(a.stage2_dir, "review_sample.csv")
        rows = read_csv(path)
        out_path = path
        fields = SAMPLE_FIELDS
    else:                                                           # every needs_review commit
        selected = [r for r in read_csv(os.path.join(a.stage2_dir, "commits_selected.csv"))
                    if r["needs_review"] == "1"]
        bodies = {(r["repo"], r["hash"]): r["body"]
                  for r in read_csv(os.path.join(a.stage1_dir, "commits.csv"))}
        rows = [{"repo": r["repo"], "hash": r["hash"], "subject": r["subject"],
                 "body": bodies.get((r["repo"], r["hash"]), ""), "stratum": "review",
                 "rule_label": r["label"], "n_design": r["n_design"],
                 "n_verification": r["n_verification"], "n_unknown": r["n_unknown"],
                 "files": files_text(by_files[(r["repo"], r["hash"])], 40),
                 "human_label": "", "llm_label": "", "llm_confidence": "", "llm_reason": ""}
                for r in selected]
        out_path = os.path.join(a.stage2_dir, "llm_review_all.csv")
        fields = SAMPLE_FIELDS

    def work(r):
        r["_diff"] = ensure_diff(a, r["repo"], r["hash"], by_files)
        results = [llm_label(client, r, cache, cache_path, run=k, temperature=a.temperature)
                   for k in range(a.runs)]
        votes = [x["label"] for x in results]
        top, n_top = Counter(v for v in votes if v).most_common(1)[0] if any(votes) else ("", 0)
        tie = sum(1 for v, c in Counter(v for v in votes if v).items() if c == n_top) > 1
        best = [x for x in results if x["label"] == top and not tie]
        r["llm_label"] = "" if tie else top
        r["llm_confidence"] = round(sum(x["confidence"] for x in best) / len(best), 3) if best else ""
        r["llm_reason"] = best[0]["reason"] if best else "runs disagree (tie)"
        r["llm_votes"] = "|".join(votes)
        r["llm_agreement"] = round(n_top / a.runs, 3) if top else 0
        return r

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:      # results come back in input order
        for i, _ in enumerate(ex.map(work, rows), 1):
            if i % 5 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}", flush=True)
    write_csv(out_path, rows, fields)
    print(f"labelled {len(rows)} commits → {out_path}   "
          f"unparseable: {sum(1 for r in rows if not r['llm_label'])}")
    return 0


# ── report ────────────────────────────────────────────────────────────────────

def norm(x):
    x = (x or "").strip().lower()
    return SHORT.get(x, x)


def confusion(pairs, rows_label, cols_label):
    cnt = Counter(pairs)
    head = "human\\pred".ljust(14) + "".join(f"{c:>14}" for c in cols_label)
    lines = [head]
    for r in rows_label:
        lines.append(r.ljust(14) + "".join(f"{cnt[(r, c)]:>14}" for c in cols_label))
    return "\n".join(lines)


def cmd_report(a):
    path = a.file or os.path.join(a.stage2_dir, "review_sample.csv")
    rows = read_csv(path)
    if a.annotations:
        human = {(r["repo"], r["hash"]): r for r in read_csv(a.annotations)}
        for r in rows:
            h = human.get((r["repo"], r["hash"]))
            r["human_label"] = h["human_label"] if h else r["human_label"]
            r["human_confidence"] = h.get("human_confidence", "") if h else ""
    rows = [r for r in rows if norm(r["human_label"])]
    bad = [r for r in rows if norm(r["human_label"]) not in LABELS]
    if bad:
        print(f"{len(bad)} rows have an invalid human_label, e.g. {bad[0]['human_label']!r}")
        return 1
    print(f"{len(rows)} annotated commits\n")
    for who, col in (("RULES", "rule_label"), ("LLM", "llm_label")):
        print("=" * 70 + f"\n{who} vs human")
        for stratum in ("confident", "review", "all"):
            sub = [r for r in rows if stratum == "all" or r["stratum"] == stratum]
            pred = [(norm(r["human_label"]), norm(r[col])) for r in sub if norm(r[col]) in LABELS]
            n_unc = len(sub) - len(pred)
            if not sub:
                continue
            k = sum(h == p for h, p in pred)
            lo, hi = wilson(k, len(pred))
            kap = kappa(pred)
            print(f"  [{stratum:<9}] n={len(sub):>3}  predicted={len(pred):>3} (abstained/unknown {n_unc})  "
                  f"accuracy={k}/{len(pred)}={k / max(1, len(pred)):.2f}  95% CI [{lo:.2f}, {hi:.2f}]  kappa={kap:.2f}")
        pairs = [(norm(r["human_label"]), norm(r[col])) for r in rows if norm(r[col]) in LABELS]
        if pairs:
            print("\n" + confusion(pairs, LABELS, LABELS) + "\n")
    # accuracy as a function of confidence
    def acc_line(name, sub):
        k = sum(norm(r["human_label"]) == norm(r["llm_label"]) for r in sub)
        return f"  {name:<28} n={len(sub):>3}  LLM correct {k}/{len(sub)}" if sub else None
    scored = [r for r in rows if norm(r["llm_label"]) in LABELS]
    print("LLM accuracy by its own confidence:")
    for name, sub in (("high (>= 0.85)", [r for r in scored if float(r["llm_confidence"] or 0) >= 0.85]),
                      ("medium (0.45 - 0.85)", [r for r in scored if 0.45 <= float(r["llm_confidence"] or 0) < 0.85]),
                      ("low (< 0.45)", [r for r in scored if float(r["llm_confidence"] or 0) < 0.45])):
        line = acc_line(name, sub)
        if line:
            print(line)
    if any(r.get("llm_agreement") not in (None, "", "0") for r in rows):
        print("\nLLM accuracy by agreement between its repeated runs:")
        for name, lo, hi in (("all runs agree (1.0)", 1.0, 1.01), ("majority only (<1.0)", 0.0, 1.0)):
            line = acc_line(name, [r for r in scored if lo <= float(r.get("llm_agreement") or 0) < hi])
            if line:
                print(line)
    if any(r.get("human_confidence") for r in rows):
        print("\nLLM accuracy by YOUR confidence (1 low, 2 med, 3 high):")
        for c in "123":
            line = acc_line(f"human confidence {c}", [r for r in scored if r.get("human_confidence") == c])
            if line:
                print(line)
    print()
    wrong = [r for r in rows if norm(r["llm_label"]) in LABELS and norm(r["llm_label"]) != norm(r["human_label"])]
    if wrong:
        print("LLM disagreements (first 10): hash subject | human → llm")
        for r in wrong[:10]:
            print(f"  {r['hash'][:10]} {r['subject'][:50]} | {norm(r['human_label'])} → {norm(r['llm_label'])}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="VeriDelta Stage 2c: validate labels")
    ap.add_argument("--stage1-dir", default="stage1_out")
    ap.add_argument("--stage2-dir", default="stage2_out")
    ap.add_argument("--sources-dir", default="sources", help="Stage 1 clones (for git show)")
    ap.add_argument("--diff-cap", type=int, default=300, help="max diff lines per commit shown to LLM and human")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n-confident", type=int, default=25)
    s.add_argument("--n-review", type=int, default=25)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(fn=cmd_sample)
    l = sub.add_parser("llm")
    l.add_argument("--scope", choices=["sample", "review"], default="sample")
    l.add_argument("--runs", type=int, default=1,
                   help="label every commit N times (independent samples); majority vote + agreement")
    l.add_argument("--workers", type=int, default=4, help="parallel API calls (default 4)")
    l.add_argument("--temperature", type=float, default=None,
                   help="sent only if given; dropped automatically if the SDK/model rejects it")
    l.set_defaults(fn=cmd_llm)
    r = sub.add_parser("report")
    r.add_argument("--file", default=None, help="sample CSV with rule/LLM labels (default: review_sample.csv)")
    r.add_argument("--annotations", default=None, help="human_labels.csv exported from review_pack.html")
    r.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())