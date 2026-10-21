"""Step 4 (any Python, standard library only): write the results of one model as report.html (tables and charts) and results.csv.

    python lib/step4_report.py --result-dir results/abcd123
"""
import argparse
import csv
import html
import json
from pathlib import Path

ORDER = ["fp32", "m1", "m2", "m3"]
NAMES = {"fp32": "FP32 baseline", "m1": "M1 all INT8", "m2": "M2 backbone INT8", "m3": "M3 backbone + neck INT8"}
COLORS = {"fp32": "#6b7280", "m1": "#1f4e9e", "m2": "#1b7f5c", "m3": "#c0551b"}


def short(r):
    g = {"per-channel": "per-channel", "per-tensor": "per-tensor"}.get(r["granularity"], "")
    return NAMES[r["design"]] + (f", {g}" if g else "")


def fmt(v, f):
    return "-" if v is None else f.format(v)


def bar_chart(title, rows, unit, f, note=""):
    """rows: [(label, value, color)] -> an inline SVG with horizontal bars."""
    rows = [r for r in rows if r[1] is not None]
    if not rows:
        return ""
    top = max(v for _, v, _ in rows) or 1
    h = 28 * len(rows) + 46
    out = [f'<figure><svg viewBox="0 0 760 {h}" width="100%" role="img" aria-label="{html.escape(title)}"><text x="0" y="18" font-size="15" font-weight="700" fill="currentColor">{html.escape(title)}</text>']
    for i, (label, v, c) in enumerate(rows):
        y = 34 + i * 28
        w = max(2, v / top * 400)
        out.append(f'<text x="0" y="{y + 14}" font-size="12.5" fill="currentColor">{html.escape(label)}</text>'
                   f'<rect x="250" y="{y}" width="{w:.1f}" height="19" rx="2" fill="{c}"/>'
                   f'<text x="{250 + w + 6:.1f}" y="{y + 14}" font-size="12.5" fill="currentColor">{f.format(v)} {unit}</text>')
    out.append("</svg>" + (f"<figcaption>{html.escape(note)}</figcaption>" if note else "") + "</figure>")
    return "".join(out)


def main(result_dir):
    rd = Path(result_dir)
    b = json.loads((rd / "benchmark.json").read_text())
    info = json.loads((rd / "run_info.json").read_text()) if (rd / "run_info.json").exists() else {}
    rows = sorted(b["results"], key=lambda r: (ORDER.index(r["design"]), r["granularity"]))
    has_acc = any(r.get("mAP50-95") is not None for r in rows)
    has_speed = any(r.get("ms_median") is not None for r in rows)

    with open(rd / "results.csv", "w", newline="", encoding="utf8") as fh:
        cols = ["file", "design", "granularity", "size_mb", "compression", "mAP50-95", "mAP50", "retained_pct", "precision", "recall", "F1", "ms_median", "ms_p95", "speedup", "peak_mb", "error"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    th = ["Model", "Size (MB)", "Smaller by"]
    if has_acc:
        th += ["mAP50-95", "Kept", "mAP50", "Precision", "Recall", "F1"]
    if has_speed:
        th += ["ms per image (median)", "p95 ms", "Faster by", "Peak memory added (MB)"]
    body = []
    for r in rows:
        tds = [short(r), fmt(r.get("size_mb"), "{:.2f}"), fmt(r.get("compression"), "{:.2f}x")]
        if has_acc:
            tds += [fmt(r.get("mAP50-95"), "{:.4f}"), fmt(r.get("retained_pct"), "{:.1f}%"), fmt(r.get("mAP50"), "{:.4f}"), fmt(r.get("precision"), "{:.3f}"), fmt(r.get("recall"), "{:.3f}"), fmt(r.get("F1"), "{:.3f}")]
        if has_speed:
            tds += [fmt(r.get("ms_median"), "{:.1f}"), fmt(r.get("ms_p95"), "{:.1f}"), fmt(r.get("speedup"), "{:.2f}x"), fmt(r.get("peak_mb"), "{:.0f}")]
        cls = ' class="base"' if r["design"] == "fp32" else ""
        body.append(f"<tr{cls}>" + "".join(f"<td>{t}</td>" for t in tds) + "</tr>")
    table = "<div class='tablewrap'><table><thead><tr>" + "".join(f"<th>{t}</th>" for t in th) + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"

    col = lambda r: COLORS[r["design"]]
    charts = ""
    if has_acc:
        charts += bar_chart("Accuracy (mAP@0.5:0.95, higher is better)", [(short(r), r.get("mAP50-95"), col(r)) for r in rows], "", "{:.4f}")
    if has_speed:
        charts += bar_chart("Time per image (milliseconds, lower is better)", [(short(r), r.get("ms_median"), col(r)) for r in rows], "ms", "{:.1f}")
        charts += bar_chart("Memory added while running (MB, lower is better)", [(short(r), r.get("peak_mb"), col(r)) for r in rows], "MB", "{:.0f}")
    charts += bar_chart("File size (MB, lower is better)", [(short(r), r.get("size_mb"), col(r)) for r in rows], "MB", "{:.2f}")

    fp = next((r for r in rows if r["design"] == "fp32"), {})
    best = None
    if has_acc and has_speed:
        cand = [r for r in rows if r["design"] != "fp32" and r.get("retained_pct") and r.get("speedup")]
        ok = [r for r in cand if r["retained_pct"] >= 95] or cand
        best = max(ok, key=lambda r: r["speedup"]) if ok else None
    summary = []
    if fp.get("size_mb"):
        summary.append(f"The FP32 baseline is {fp['size_mb']:.2f} MB" + (f" and scores mAP@0.5:0.95 {fp['mAP50-95']:.4f}" if fp.get("mAP50-95") is not None else "") +
                       (f"; it takes {fp['ms_median']:.1f} ms per image on this computer" if fp.get("ms_median") else "") + ".")
    if best:
        summary.append(f"The fastest INT8 variant that keeps at least 95% of the accuracy (or the closest to it) is <b>{short(best)}</b>: {best['retained_pct']:.1f}% of the baseline accuracy, "
                       f"{best['speedup']:.2f}x faster, {best['compression']:.2f}x smaller.")
    warnings = "".join(f"<li>{html.escape(w)}</li>" for w in info.get("warnings", []) + [f"{r['file']}: {r['error']}" for r in rows if r.get("error")])
    setup = [("Model file", info.get("pt", b["name"] + ".pt")), ("Classes", b["nc"]), ("Input size", f"{b['imgsz']} x {b['imgsz']}"),
             ("Test pictures (accuracy)", b["n_test_pictures"] or "none given: accuracy not measured"), ("Calibration pictures", b.get("n_calibration")),
             ("Computer", f"{b['physical_cores']} physical / {b['logical_cores']} logical cores; {b['threads']} threads pinned to CPUs {b['cpus']}"),
             ("TFLite runtime", b["runtime"]), ("Speed protocol", f"batch 1, 20 warm-up + 100 timed calls per round, {b['rounds']} interleaved rounds, median of the rounds")]
    setup_html = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>" for k, v in setup)

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(b['name'])}: TFLite variants</title>
<style>:root{{color-scheme:light dark}}body{{margin:0;font:16px/1.6 Georgia,serif;background:#fff;color:#1a1a1a}}@media(prefers-color-scheme:dark){{body{{background:#16181c;color:#e4e6ea}}th,td{{border-color:#444!important}}}}
.page{{max-width:980px;margin:0 auto;padding:40px 22px 80px}}h1,h2,table,figcaption{{font-family:-apple-system,"Segoe UI",Arial,sans-serif}}h1{{font-size:28px}}h2{{font-size:19px;margin-top:38px}}
.tablewrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13.5px}}th,td{{padding:6px 10px 6px 0;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}
tr.base td{{font-weight:700}}table.setup td{{text-align:left;white-space:normal}}figure{{margin:22px 0}}figcaption{{font-size:13px;color:#666}}.warn{{background:#fff4e5;color:#1a1a1a;border-left:4px solid #e08a00;padding:8px 14px;margin:18px 0}}</style></head><body><main class="page">
<h1>{html.escape(b['name'])}: TFLite variants against the FP32 baseline</h1>
<p>{' '.join(summary)}</p>
{'<div class="warn"><b>Please note</b><ul>' + warnings + '</ul></div>' if warnings else ''}
<h2>Results</h2>{table}
<p style="font-size:14px">"Smaller by", "Kept" (share of the baseline accuracy) and "Faster by" are all against the FP32 baseline in the same table. Per-channel means one INT8 scale for every output channel of each layer (more accurate); per-tensor means one scale per layer.</p>
{charts}
<h2>What the variants are</h2>
<ul><li><b>FP32 baseline</b>: the model you provided, converted to TFLite without quantization.</li>
<li><b>M1 all INT8</b>: every layer, including the detection head, in 8-bit integers. Smallest and usually fastest, least accurate. Box coordinates are scaled to 0-1 inside the model so that they survive 8-bit rounding.</li>
<li><b>M2 backbone INT8</b>: only the feature-extraction part (backbone) is INT8; the neck and the head stay float. Most accurate INT8 variant, largest file.</li>
<li><b>M3 backbone + neck INT8</b>: backbone and neck INT8, the detection head stays float. Usually the best balance.</li>
<li>The weights of the detection head are never changed in any variant.</li></ul>
<h2>Setup</h2><div class="tablewrap"><table class="setup"><tbody>{setup_html}</tbody></table></div>
<h2>How to read these numbers</h2>
<ul><li>mAP@0.5:0.95 is the standard detector score (0 to 1, higher is better) on your test pictures; "Kept" is the INT8 score as a share of the FP32 score. A loss of a few percent is typical for 8-bit models.</li>
<li>Time and memory were measured on this computer's CPU. An NPU or another CPU will rank the variants differently, so use these as a guide and measure on the target.</li>
<li>Precision / recall / F1 are at confidence 0.25 and overlap 0.5; mAP uses confidence 0.001 as in the standard validation.</li>
<li>TFLite's own converter only offers min-max calibration of the value ranges; a few unusual calibration pictures can widen a range and cost accuracy, so use pictures that look like the real data.</li></ul>
</main></body></html>"""
    (rd / "report.html").write_text(page, encoding="utf8")
    return rd / "report.html"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    print("wrote", main(ap.parse_args().result_dir))
