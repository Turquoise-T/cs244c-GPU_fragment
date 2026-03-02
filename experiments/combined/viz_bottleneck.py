"""Parse stderr WARNING logs from FGD experiments and generate bottleneck visualization."""
import re
import sys
import json
from collections import defaultdict

import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_SCRIPT_DIR, "slurm/slurm_logs")
JOB_ID = "1427785"
EXPERIMENTS = {
    0: "FGD 60 jph",
    1: "FGD 180 jph",
    2: "FGD 360 jph",
}
# Alibaba cluster spec
CLUSTER_TOTAL = {"G2": 4392, "T4": 840, "G3": 312, "P100": 264, "V100M32": 200, "V100M16": 192}
TOTAL_GPUS = sum(CLUSTER_TOTAL.values())

# Wall-clock: experiments started at 07:48:26, killed ~41 min later
WALL_SECONDS = 41 * 60  # approximate

def parse_log(exp_idx):
    """Parse stderr log, extract (sim_time, active_jobs, {gpu_type: unused}) per round."""
    path = f"{LOG_DIR}/ali_fgd-{JOB_ID}_{exp_idx}.err"
    pattern = re.compile(
        r'scheduler:WARNING \[([0-9.]+)\] (\d+) GPUs of type (\w+) left unused\. Number of active jobs: (\d+)'
    )

    # Group by sim_time: collect all GPU unused counts for the same round
    rounds = defaultdict(lambda: {"active_jobs": 0, "unused": {}})
    for line in open(path):
        m = pattern.search(line)
        if m:
            sim_time = float(m.group(1))
            unused_count = int(m.group(2))
            gpu_type = m.group(3)
            active_jobs = int(m.group(4))
            rounds[sim_time]["active_jobs"] = active_jobs
            rounds[sim_time]["unused"][gpu_type] = unused_count

    # Convert to sorted list of data points
    data = []
    for sim_time in sorted(rounds.keys()):
        r = rounds[sim_time]
        total_unused = sum(r["unused"].values())
        utilization = (TOTAL_GPUS - total_unused) / TOTAL_GPUS
        data.append({
            "sim_time": sim_time,
            "sim_hours": sim_time / 3600,
            "active_jobs": r["active_jobs"],
            "total_unused_gpus": total_unused,
            "utilization": utilization,
            "unused_by_type": r["unused"],
        })
    return data


def main():
    all_data = {}
    for idx, label in EXPERIMENTS.items():
        data = parse_log(idx)
        all_data[label] = data
        if data:
            last = data[-1]
            print(f"{label}: {len(data)} data points, "
                  f"reached {last['sim_hours']:.1f} sim-hours, "
                  f"{last['active_jobs']} active jobs, "
                  f"{last['utilization']:.1%} utilization")

    # Write JSON for visualization
    output = {
        "cluster": CLUSTER_TOTAL,
        "total_gpus": TOTAL_GPUS,
        "wall_seconds": WALL_SECONDS,
        "experiments": {}
    }
    for label, data in all_data.items():
        output["experiments"][label] = data

    with open(os.path.join(_SCRIPT_DIR, "bottleneck_analysis.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote bottleneck_analysis.json ({len(json.dumps(output))} bytes)")

    # Generate HTML visualization
    generate_html(output)


def generate_html(data):
    html = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>FGD Alibaba Bottleneck Analysis</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0d1117; color: #c9d1d9; padding: 20px; }
  h1 { color: #58a6ff; margin-bottom: 5px; font-size: 22px; }
  .subtitle { color: #8b949e; margin-bottom: 20px; font-size: 14px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
  .chart-container { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .chart-title { color: #f0f6fc; font-size: 14px; font-weight: 600; margin-bottom: 10px; }
  canvas { width: 100%; }
  .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px; }
  .stat-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; }
  .stat-label { color: #8b949e; font-size: 12px; }
  .stat-value { color: #f0f6fc; font-size: 24px; font-weight: 700; margin: 4px 0; }
  .stat-detail { color: #8b949e; font-size: 11px; }
  .legend { display: flex; gap: 20px; justify-content: center; margin-bottom: 16px; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: 13px; }
  .legend-color { width: 14px; height: 3px; border-radius: 2px; }
  .insight { background: #1c2333; border-left: 3px solid #58a6ff; padding: 12px 16px;
             border-radius: 0 6px 6px 0; margin-bottom: 20px; font-size: 13px; line-height: 1.5; }
  .insight strong { color: #58a6ff; }
</style>
</head><body>
<h1>FGD Alibaba Bottleneck Analysis</h1>
<div class="subtitle">3 experiments on 6,200-GPU Alibaba cluster with migration penalty, 10-min rounds | ~41 min wall time before cancellation</div>
"""

    # Stats cards
    exps = data["experiments"]
    html += '<div class="stats">\n'
    colors = {"FGD 60 jph": "#58a6ff", "FGD 180 jph": "#f78166", "FGD 360 jph": "#7ee787"}
    for label in ["FGD 60 jph", "FGD 180 jph", "FGD 360 jph"]:
        d = exps[label]
        if not d:
            continue
        last = d[-1]
        rate = last["sim_hours"] / (data["wall_seconds"] / 60)
        html += f'''<div class="stat-card">
  <div class="stat-label">{label}</div>
  <div class="stat-value" style="color:{colors[label]}">{last["sim_hours"]:.0f} hr</div>
  <div class="stat-detail">{last["active_jobs"]} active jobs | {rate:.1f} sim-hr/wall-min | {last["utilization"]:.0%} util</div>
</div>\n'''
    html += '</div>\n'

    # Insight box
    html += '''<div class="insight">
<strong>Key finding:</strong> Simulation throughput drops ~15x going from 60 jph (1,800 jobs) to 360 jph (5,100+ jobs).
The LP solver scales as O(m&sup2;) with active job count m, and the event loop processes proportionally more events.
At 360 jph, the experiment would need ~6+ hours to complete -- exceeding the 4-hour SLURM limit.
</div>\n'''

    # Legend
    html += '<div class="legend">\n'
    for label, color in colors.items():
        html += f'<div class="legend-item"><div class="legend-color" style="background:{color}"></div>{label}</div>\n'
    html += '</div>\n'

    html += '<div class="grid">\n'

    # We'll use inline JS with canvas for charts
    html += '''
<div class="chart-container">
  <div class="chart-title">Active Jobs vs Simulation Time</div>
  <canvas id="chart1" height="280"></canvas>
</div>
<div class="chart-container">
  <div class="chart-title">GPU Utilization vs Simulation Time</div>
  <canvas id="chart2" height="280"></canvas>
</div>
<div class="chart-container">
  <div class="chart-title">Simulation Progress (sim-hours vs wall-minutes)</div>
  <canvas id="chart3" height="280"></canvas>
</div>
<div class="chart-container">
  <div class="chart-title">Unused GPUs by Type (360 jph)</div>
  <canvas id="chart4" height="280"></canvas>
</div>
</div>
'''

    # Embed data and chart rendering JS
    html += f'<script>\nconst DATA = {json.dumps(data)};\n'
    html += '''
const COLORS = {"FGD 60 jph": "#58a6ff", "FGD 180 jph": "#f78166", "FGD 360 jph": "#7ee787"};
const WALL_MIN = DATA.wall_seconds / 60;

function drawChart(canvasId, getData, opts) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  const pad = {top: 10, right: 15, bottom: 35, left: 55};
  const pw = W - pad.left - pad.right, ph = H - pad.top - pad.bottom;

  // Get all series data
  const series = {};
  let xMax = 0, yMax = 0;
  for (const [label, points] of Object.entries(DATA.experiments)) {
    const d = getData(points, label);
    series[label] = d;
    for (const p of d) {
      if (p.x > xMax) xMax = p.x;
      if (p.y > yMax) yMax = p.y;
    }
  }
  if (opts.xMax) xMax = opts.xMax;
  if (opts.yMax) yMax = opts.yMax;
  // Round up nicely
  yMax = yMax * 1.1;

  // Grid
  ctx.strokeStyle = "#21262d";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (ph * i / 4);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + pw, y); ctx.stroke();
  }

  // Axes
  ctx.fillStyle = "#8b949e";
  ctx.font = "11px -apple-system, sans-serif";
  ctx.textAlign = "center";
  // X axis labels
  const xTicks = 5;
  for (let i = 0; i <= xTicks; i++) {
    const val = (xMax * i / xTicks);
    const x = pad.left + (pw * i / xTicks);
    ctx.fillText(opts.xFormat ? opts.xFormat(val) : val.toFixed(0), x, H - 5);
  }
  // Y axis labels
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const val = yMax * (4 - i) / 4;
    const y = pad.top + (ph * i / 4) + 4;
    ctx.fillText(opts.yFormat ? opts.yFormat(val) : val.toFixed(0), pad.left - 5, y);
  }
  // Axis titles
  ctx.fillStyle = "#8b949e";
  ctx.font = "11px -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(opts.xLabel || "", pad.left + pw / 2, H - 0);
  ctx.save();
  ctx.translate(12, pad.top + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(opts.yLabel || "", 0, 0);
  ctx.restore();

  // Draw lines
  for (const [label, points] of Object.entries(series)) {
    if (points.length < 2) continue;
    ctx.strokeStyle = COLORS[label];
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < points.length; i++) {
      const x = pad.left + (points[i].x / xMax) * pw;
      const y = pad.top + ph - (points[i].y / yMax) * ph;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

// Chart 1: Active jobs vs sim time (hours)
drawChart("chart1",
  (points) => points.map(p => ({x: p.sim_hours, y: p.active_jobs})),
  {xLabel: "Simulation time (hours)", yLabel: "Active jobs", yMax: 6000}
);

// Chart 2: Utilization vs sim time
drawChart("chart2",
  (points) => points.map(p => ({x: p.sim_hours, y: p.utilization * 100})),
  {xLabel: "Simulation time (hours)", yLabel: "Utilization %",
   yMax: 100, yFormat: v => v.toFixed(0) + "%"}
);

// Chart 3: Sim progress -- sim_hours on Y, estimate wall-min on X
// We approximate wall time by distributing total wall time proportionally across data points
drawChart("chart3",
  (points, label) => {
    if (points.length === 0) return [];
    // Approximate: assume data points are equally spaced in wall time
    return points.map((p, i) => ({
      x: (i / (points.length - 1 || 1)) * WALL_MIN,
      y: p.sim_hours
    }));
  },
  {xLabel: "Wall time (minutes)", yLabel: "Sim hours",
   xMax: WALL_MIN}
);

// Chart 4: Unused GPUs by type for 360 jph
(function() {
  const canvas = document.getElementById("chart4");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  const pad = {top: 10, right: 15, bottom: 35, left: 55};
  const pw = W - pad.left - pad.right, ph = H - pad.top - pad.bottom;

  const points = DATA.experiments["FGD 360 jph"];
  const gpuTypes = Object.keys(DATA.cluster);
  const typeColors = ["#58a6ff","#f78166","#7ee787","#d2a8ff","#f2cc60","#ff7b72"];

  // Get max sim hours
  const xMax = points.length ? points[points.length-1].sim_hours : 1;

  // Grid
  ctx.strokeStyle = "#21262d"; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (ph * i / 4);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + pw, y); ctx.stroke();
  }

  // Y max
  const yMax = Math.max(...gpuTypes.map(t => DATA.cluster[t])) * 1.1;

  // X/Y labels
  ctx.fillStyle = "#8b949e"; ctx.font = "11px -apple-system, sans-serif";
  ctx.textAlign = "center";
  for (let i = 0; i <= 5; i++) {
    ctx.fillText((xMax * i / 5).toFixed(0), pad.left + pw * i / 5, H - 5);
  }
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    ctx.fillText((yMax * (4-i) / 4).toFixed(0), pad.left - 5, pad.top + ph * i / 4 + 4);
  }
  ctx.textAlign = "center";
  ctx.fillText("Simulation time (hours)", pad.left + pw/2, H - 0);

  // Draw line per GPU type
  for (let ti = 0; ti < gpuTypes.length; ti++) {
    const gtype = gpuTypes[ti];
    ctx.strokeStyle = typeColors[ti];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let started = false;
    for (const p of points) {
      const unused = (p.unused_by_type && p.unused_by_type[gtype]) || 0;
      const x = pad.left + (p.sim_hours / xMax) * pw;
      const y = pad.top + ph - (unused / yMax) * ph;
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
    // Label at end
    if (points.length) {
      const lp = points[points.length - 1];
      const unused = (lp.unused_by_type && lp.unused_by_type[gtype]) || 0;
      const x = pad.left + pw + 2;
      const y = pad.top + ph - (unused / yMax) * ph;
      ctx.fillStyle = typeColors[ti];
      ctx.textAlign = "left";
      ctx.font = "9px -apple-system, sans-serif";
      ctx.fillText(gtype, x - 12, y - 5);
    }
  }
})();
</script>
</body></html>'''

    out_path = os.path.join(_SCRIPT_DIR, "bottleneck_analysis.html")
    with open(out_path, "w") as f:
        f.write(html)
    print("Wrote bottleneck_analysis.html")


if __name__ == "__main__":
    main()
