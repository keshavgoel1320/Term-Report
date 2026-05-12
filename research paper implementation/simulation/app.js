/* =====================================================================
   app.js — Pipeline Evaluation Simulation Engine
   Cinematic step-by-step reveal: query → results → scores → metrics
   ===================================================================== */

// ── State ──
let DATA = null;
let PRODUCTS = null;
let QUERIES = null;
let currentQueryIdx = -1;
let isAnimating = false;    // true while a query is animating
let autoPlayOn = false;
let autoPlaySpeed = 3000;   // ms pause between queries
const pipelineNames = ['basic_hybrid', 'enriched_hybrid', 'rewrite_enriched'];
const pipelineLabels = { basic_hybrid: 'Basic Hybrid', enriched_hybrid: 'Enriched Hybrid', rewrite_enriched: 'Rewrite + Enriched' };

// cumulative accumulators
const cumulativeMetrics = {
  basic_hybrid:    { ndcg5: [], ndcg10: [], mrr: [], judge: [] },
  enriched_hybrid: { ndcg5: [], ndcg10: [], mrr: [], judge: [] },
  rewrite_enriched:{ ndcg5: [], ndcg10: [], mrr: [], judge: [] }
};

// ── Helper: sleep ──
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ── Data Loading ──
async function loadData() {
  try {
    const [resResp, prodResp, qResp] = await Promise.all([
      fetch('../results.json'), fetch('../data/products.json'), fetch('../data/queries.json')
    ]);
    DATA = await resResp.json();
    PRODUCTS = await prodResp.json();
    QUERIES = await qResp.json();
    document.getElementById('loading').style.display = 'none';
    document.getElementById('app-main').style.display = 'block';
    initUI();
  } catch (e) {
    document.getElementById('loading').innerHTML =
      `<div class="empty-state"><div class="icon">!</div><p>Failed to load data. Run via serve.py<br><br><code>${e.message}</code></p></div>`;
  }
}

// ── Metric Calculations (matching Python exactly) ──
function computeNDCG(scores, k) {
  if (!scores.length) return 0;
  const atK = scores.slice(0, k);
  const dcg = atK.reduce((s, v, i) => s + (Math.pow(2, v) - 1) / Math.log2(i + 2), 0);
  const ideal = [...scores].sort((a, b) => b - a).slice(0, k);
  const idcg = ideal.reduce((s, v, i) => s + (Math.pow(2, v) - 1) / Math.log2(i + 2), 0);
  return idcg > 0 ? dcg / idcg : 0;
}
function computeMRR(scores, threshold = 3) {
  for (let i = 0; i < scores.length; i++) { if (scores[i] >= threshold) return 1 / (i + 1); }
  return 0;
}
function computeMeanJudge(scores) {
  return scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
}
function roundTo(n, d) { return Math.round(n * Math.pow(10, d)) / Math.pow(10, d); }

// ── Init UI ──
function initUI() {
  setupTabs();
  setupControls();
  renderEmptyState();
  initProductDB();
}

// ── Tabs ──
function setupTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab).classList.add('active');
    });
  });
}

// ── Controls ──
function setupControls() {
  document.getElementById('btn-next').addEventListener('click', () => { if (!isAnimating) advanceQuery(); });
  document.getElementById('btn-autoplay').addEventListener('click', toggleAutoPlay);
  document.getElementById('btn-total').addEventListener('click', showTotalResult);
  document.getElementById('btn-reset').addEventListener('click', resetSimulation);
  document.getElementById('speed-slider').addEventListener('input', e => {
    autoPlaySpeed = parseInt(e.target.value);
    document.getElementById('speed-val').textContent = (autoPlaySpeed / 1000).toFixed(1) + 's';
  });
}

function setButtonStates() {
  const done = currentQueryIdx + 1;
  const total = QUERIES.length;
  document.getElementById('btn-next').disabled = isAnimating || done >= total;
  document.getElementById('progress-text').textContent = `${done} / ${total}`;
  document.getElementById('progress-fill').style.width = `${(done / total) * 100}%`;
}

// ═══════════════════════════════════════════════════════════════════════
//  CINEMATIC QUERY PROCESSING — the heart of the animation
// ═══════════════════════════════════════════════════════════════════════

async function advanceQuery() {
  currentQueryIdx++;
  if (currentQueryIdx >= QUERIES.length) {
    if (autoPlayOn) toggleAutoPlay();
    showFinalResults();
    return;
  }
  isAnimating = true;
  setButtonStates();

  const idx = currentQueryIdx;
  const query = QUERIES[idx];
  const queryResults = {};
  for (const p of pipelineNames) {
    queryResults[p] = DATA.pipelines[p].per_query[idx];
  }

  // ── Phase 1: Show the query (scroll to it) ──
  clearPreviousResults();
  await sleep(200);
  renderQueryBanner(query, queryResults.rewrite_enriched);
  document.getElementById('query-banner').scrollIntoView({ behavior: 'smooth', block: 'start' });
  await sleep(800);

  // ── Phase 2: Reveal results one by one, across all 3 pipelines simultaneously ──
  const maxResults = 10;
  for (let rank = 0; rank < maxResults; rank++) {
    for (const p of pipelineNames) {
      const qr = queryResults[p];
      if (rank < qr.results.length) {
        appendResultItem(p, qr.results[rank], rank, false); // no judge score yet
      }
    }
    await sleep(180); // pause between each rank
  }
  await sleep(400);

  // ── Phase 3: Reveal judge scores (badges light up) ──
  for (let rank = 0; rank < maxResults; rank++) {
    for (const p of pipelineNames) {
      const qr = queryResults[p];
      if (rank < qr.judge_scores.length) {
        revealJudgeScore(p, rank, qr.judge_scores[rank].score);
      }
    }
    await sleep(120);
  }
  await sleep(500);

  // ── Phase 4: Calculate and show per-query metrics ──
  computeAndRenderMetrics(queryResults);
  await sleep(600);

  // ── Phase 5: Update cumulative table ──
  renderCumulativeMetrics();
  setButtonStates();

  isAnimating = false;
  setButtonStates();

  // If autoplay, schedule next
  if (autoPlayOn) {
    await sleep(autoPlaySpeed);
    if (autoPlayOn && !isAnimating) advanceQuery();
  }
}

// ── Clear previous results (keep structure) ──
function clearPreviousResults() {
  for (const p of pipelineNames) {
    document.getElementById(`results-${p}`).innerHTML = '';
  }
  document.getElementById('per-query-metrics').style.display = 'none';
}

// ── Render Query Banner ──
function renderQueryBanner(query, rewriteResult) {
  const banner = document.getElementById('query-banner');
  const meta = rewriteResult.metadata || {};
  let html = `
    <div class="query-id">${query.query_id} — QUERY</div>
    <div class="query-text">"${query.text}"</div>
  `;
  if (meta.rewritten_query && meta.rewritten_query !== query.text) {
    html += `
      <div class="rewrite-section">
        <div class="rewrite-label">&#10022; LLM Rewritten Query (Pipeline 3)</div>
        <div class="rewrite-query">"${meta.rewritten_query}"</div>
        ${meta.rewrite_reasoning ? `<div class="rewrite-reasoning">Reasoning: ${meta.rewrite_reasoning}</div>` : ''}
      </div>
    `;
  }
  banner.innerHTML = html;
  banner.style.display = 'block';
  banner.classList.remove('animate-in');
  void banner.offsetWidth; // force reflow
  banner.classList.add('animate-in');
}

// ── Append a single result item (without judge score initially) ──
function appendResultItem(pipeline, result, rank, showJudge) {
  const container = document.getElementById(`results-${pipeline}`);
  const div = document.createElement('div');
  div.className = 'result-item result-enter';
  div.dataset.rank = rank;
  div.innerHTML = `
    <div class="result-rank">${result.rank}</div>
    <div class="result-info">
      <div class="result-name" title="${result.name}">${result.name}</div>
      <div class="result-category">${result.category}</div>
      <div class="result-scores">
        <span class="result-score-tag">Hybrid: ${result.score.toFixed(3)}</span>
        <span class="result-score-tag">BM25: ${result.bm25_score.toFixed(3)}</span>
        <span class="result-score-tag">Dense: ${result.dense_score.toFixed(3)}</span>
      </div>
    </div>
    <div class="judge-badge judge-hidden" id="judge-${pipeline}-${rank}">?</div>
  `;
  container.appendChild(div);
}

// ── Reveal a judge score with pop animation ──
function revealJudgeScore(pipeline, rank, score) {
  const badge = document.getElementById(`judge-${pipeline}-${rank}`);
  if (!badge) return;
  badge.className = `judge-badge judge-${score} judge-pop`;
  badge.textContent = score;
  badge.title = `Judge Score: ${score}/5`;
}

// ── Per-Query Metrics ──
function computeAndRenderMetrics(queryResults) {
  const metrics = {};
  for (const p of pipelineNames) {
    const qr = queryResults[p];
    const judgeScores = qr.judge_scores.map(j => j.score);
    metrics[p] = {
      ndcg5:  roundTo(computeNDCG(judgeScores, 5), 4),
      ndcg10: roundTo(computeNDCG(judgeScores, 10), 4),
      mrr:    roundTo(computeMRR(judgeScores), 4),
      judge:  roundTo(computeMeanJudge(judgeScores), 4)
    };
    cumulativeMetrics[p].ndcg5.push(metrics[p].ndcg5);
    cumulativeMetrics[p].ndcg10.push(metrics[p].ndcg10);
    cumulativeMetrics[p].mrr.push(metrics[p].mrr);
    cumulativeMetrics[p].judge.push(metrics[p].judge);
  }

  const metricKeys = ['ndcg5', 'ndcg10', 'mrr', 'judge'];
  const metricLabels = { ndcg5: 'NDCG@5', ndcg10: 'NDCG@10', mrr: 'MRR', judge: 'Mean Judge' };
  const panel = document.getElementById('per-query-metrics');
  let html = '<div class="card-header"><h3>This Query\'s Metrics</h3></div><div class="card-body"><div class="metrics-grid">';
  for (const mk of metricKeys) {
    const vals = pipelineNames.map(p => metrics[p][mk]);
    const maxVal = Math.max(...vals);
    html += `<div class="metric-card metric-enter"><div class="metric-label">${metricLabels[mk]}</div><div class="metric-values">`;
    pipelineNames.forEach((p, i) => {
      const pipeClass = p === 'basic_hybrid' ? 'basic' : p === 'enriched_hybrid' ? 'enriched' : 'rewrite';
      const isBest = Math.abs(vals[i] - maxVal) < 0.0001;
      html += `<div class="metric-val">
        <span class="metric-num ${isBest ? 'metric-best' : ''}">${vals[i].toFixed(4)}</span>
        <span class="metric-pipe ${pipeClass}">${pipelineLabels[p].split(' ')[0]}</span>
        ${isBest ? '<span class="winner-tag">BEST</span>' : ''}
      </div>`;
    });
    html += '</div></div>';
  }
  html += '</div></div>';
  panel.innerHTML = html;
  panel.style.display = 'block';
  panel.classList.remove('section-enter');
  void panel.offsetWidth;
  panel.classList.add('section-enter');
}

// ── Cumulative Metrics ──
function renderCumulativeMetrics() {
  const section = document.getElementById('cumulative-section');
  const n = cumulativeMetrics.basic_hybrid.ndcg5.length;
  if (n === 0) { section.style.display = 'none'; return; }
  section.style.display = 'block';

  const avg = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
  const metricKeys = ['ndcg5', 'ndcg10', 'mrr', 'judge'];
  const metricLabels = { ndcg5: 'NDCG@5', ndcg10: 'NDCG@10', mrr: 'MRR', judge: 'Mean Judge' };

  const bests = {};
  for (const mk of metricKeys) {
    const vals = pipelineNames.map(p => avg(cumulativeMetrics[p][mk]));
    bests[mk] = Math.max(...vals);
  }

  let html = `<table class="cumulative-table"><thead><tr><th style="text-align:left">Pipeline</th>`;
  for (const mk of metricKeys) html += `<th>${metricLabels[mk]}</th>`;
  html += `<th>Queries</th></tr></thead><tbody>`;

  for (const p of pipelineNames) {
    const pipeClass = p === 'basic_hybrid' ? 'basic' : p === 'enriched_hybrid' ? 'enriched' : 'rewrite';
    html += `<tr><td class="pipe-cell"><span class="metric-pipe ${pipeClass}" style="font-size:0.8rem">${pipelineLabels[p]}</span></td>`;
    for (const mk of metricKeys) {
      const val = roundTo(avg(cumulativeMetrics[p][mk]), 4);
      const isBest = Math.abs(val - bests[mk]) < 0.0001;
      html += `<td class="${isBest ? 'best-val' : ''}">${val.toFixed(4)}</td>`;
    }
    html += `<td>${n}</td></tr>`;
  }
  html += '</tbody></table>';
  document.getElementById('cumulative-body').innerHTML = html;

  section.classList.remove('section-enter');
  void section.offsetWidth;
  section.classList.add('section-enter');
}

// ── Auto-Play ──
function toggleAutoPlay() {
  const btn = document.getElementById('btn-autoplay');
  if (autoPlayOn) {
    autoPlayOn = false;
    btn.textContent = '▶ Auto-Play';
    btn.classList.remove('btn-danger');
    btn.classList.add('btn-success');
  } else {
    autoPlayOn = true;
    btn.textContent = '⏸ Pause';
    btn.classList.remove('btn-success');
    btn.classList.add('btn-danger');
    if (!isAnimating) advanceQuery();
  }
}

// ── Reset ──
function resetSimulation() {
  autoPlayOn = false;
  const btn = document.getElementById('btn-autoplay');
  btn.textContent = '▶ Auto-Play';
  btn.classList.remove('btn-danger');
  btn.classList.add('btn-success');

  isAnimating = false;
  currentQueryIdx = -1;
  for (const p of pipelineNames) {
    cumulativeMetrics[p] = { ndcg5: [], ndcg10: [], mrr: [], judge: [] };
  }
  renderEmptyState();
  setButtonStates();
}

function renderEmptyState() {
  document.getElementById('query-banner').style.display = 'none';
  document.getElementById('query-banner').innerHTML = '';
  for (const p of pipelineNames) {
    document.getElementById(`results-${p}`).innerHTML =
      '<div class="empty-state"><div class="icon">&#128203;</div><p>Click "Next Query" or "Auto-Play" to begin</p></div>';
  }
  document.getElementById('per-query-metrics').style.display = 'none';
  document.getElementById('cumulative-section').style.display = 'none';
}

// ── Final Results Overlay ──
function showFinalResults() {
  const avg = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
  const metricKeys = ['ndcg5', 'ndcg10', 'mrr', 'judge'];
  const metricLabels = { ndcg5: 'NDCG@5', ndcg10: 'NDCG@10', mrr: 'MRR', judge: 'Mean Judge Score' };

  const bests = {};
  for (const mk of metricKeys) {
    const vals = pipelineNames.map(p => avg(cumulativeMetrics[p][mk]));
    bests[mk] = Math.max(...vals);
  }

  let tableHtml = `<table class="cumulative-table"><thead><tr><th style="text-align:left">Pipeline</th>`;
  for (const mk of metricKeys) tableHtml += `<th>${metricLabels[mk]}</th>`;
  tableHtml += `</tr></thead><tbody>`;
  for (const p of pipelineNames) {
    const pipeClass = p === 'basic_hybrid' ? 'basic' : p === 'enriched_hybrid' ? 'enriched' : 'rewrite';
    tableHtml += `<tr><td class="pipe-cell"><span class="metric-pipe ${pipeClass}" style="font-size:0.85rem">${pipelineLabels[p]}</span></td>`;
    for (const mk of metricKeys) {
      const val = roundTo(avg(cumulativeMetrics[p][mk]), 4);
      const isBest = Math.abs(val - bests[mk]) < 0.0001;
      tableHtml += `<td class="${isBest ? 'best-val' : ''}" style="font-size:1rem">${val.toFixed(4)}</td>`;
    }
    tableHtml += `</tr>`;
  }
  tableHtml += '</tbody></table>';

  const judgeBasic = roundTo(avg(cumulativeMetrics.basic_hybrid.judge), 4);
  const judgeEnriched = roundTo(avg(cumulativeMetrics.enriched_hybrid.judge), 4);
  const improvement = roundTo(((judgeEnriched - judgeBasic) / judgeBasic) * 100, 1);

  const overlay = document.createElement('div');
  overlay.className = 'final-overlay';
  overlay.id = 'final-overlay';
  overlay.innerHTML = `
    <div class="final-card">
      <h2>Simulation Complete</h2>
      <p class="final-subtitle">All 50 queries evaluated across 3 pipelines</p>
      ${tableHtml}
      <div style="text-align:center; margin-top:1.5rem; padding:1rem; background:rgba(0,245,196,0.06); border-radius:8px; border:1px solid rgba(0,245,196,0.15)">
        <div style="font-size:0.75rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.8px; margin-bottom:0.5rem">Key Finding</div>
        <div style="font-size:1rem; color:var(--accent-teal); font-weight:600">Document enrichment improved relevance by +${improvement}%</div>
        <div style="font-size:0.8rem; color:var(--text-secondary); margin-top:0.3rem">with zero query-time latency cost</div>
      </div>
      <button class="btn btn-primary final-close" onclick="document.getElementById('final-overlay').remove()">Close</button>
    </div>
  `;
  document.body.appendChild(overlay);
}

// ═══════════════════════════════════════════════════════════════════════
//  TOTAL RESULT — instant full experiment overview
// ═══════════════════════════════════════════════════════════════════════

function showTotalResult() {
  if (!DATA) return;
  const avg = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
  const metricKeys = ['ndcg5', 'ndcg10', 'mrr', 'judge'];
  const metricLabels = { ndcg5: 'NDCG@5', ndcg10: 'NDCG@10', mrr: 'MRR', judge: 'Mean Judge Score' };

  // Compute all metrics from raw data
  const allMetrics = {};
  const perQuery = {}; // {qid: {pipeline: {metrics}}}
  for (const p of pipelineNames) {
    const qs = DATA.pipelines[p].per_query;
    allMetrics[p] = { ndcg5: [], ndcg10: [], mrr: [], judge: [] };
    qs.forEach(q => {
      const js = q.judge_scores.map(j => j.score);
      const m = {
        ndcg5: roundTo(computeNDCG(js, 5), 4),
        ndcg10: roundTo(computeNDCG(js, 10), 4),
        mrr: roundTo(computeMRR(js), 4),
        judge: roundTo(computeMeanJudge(js), 4)
      };
      allMetrics[p].ndcg5.push(m.ndcg5);
      allMetrics[p].ndcg10.push(m.ndcg10);
      allMetrics[p].mrr.push(m.mrr);
      allMetrics[p].judge.push(m.judge);
      if (!perQuery[q.query_id]) perQuery[q.query_id] = { text: q.query_text };
      perQuery[q.query_id][p] = m;
    });
  }

  // Aggregate table
  const bests = {};
  for (const mk of metricKeys) {
    const vals = pipelineNames.map(p => avg(allMetrics[p][mk]));
    bests[mk] = Math.max(...vals);
  }

  let aggTable = `<table class="cumulative-table"><thead><tr><th style="text-align:left">Pipeline</th>`;
  for (const mk of metricKeys) aggTable += `<th>${metricLabels[mk]}</th>`;
  aggTable += `</tr></thead><tbody>`;
  for (const p of pipelineNames) {
    const pipeClass = p === 'basic_hybrid' ? 'basic' : p === 'enriched_hybrid' ? 'enriched' : 'rewrite';
    aggTable += `<tr><td class="pipe-cell"><span class="metric-pipe ${pipeClass}" style="font-size:0.85rem">${pipelineLabels[p]}</span></td>`;
    for (const mk of metricKeys) {
      const val = roundTo(avg(allMetrics[p][mk]), 4);
      const isBest = Math.abs(val - bests[mk]) < 0.0001;
      aggTable += `<td class="${isBest ? 'best-val' : ''}" style="font-size:1rem">${val.toFixed(4)}</td>`;
    }
    aggTable += `</tr>`;
  }
  aggTable += '</tbody></table>';

  // Per-query breakdown table (scrollable)
  let pqTable = `<div style="max-height:300px;overflow-y:auto;margin-top:1rem;border:1px solid var(--border-glass);border-radius:8px">
    <table class="cumulative-table" style="font-size:0.75rem">
    <thead style="position:sticky;top:0;background:var(--bg-secondary);z-index:1"><tr>
      <th style="text-align:left">Query</th><th style="text-align:left">Text</th>
      <th>Basic Judge</th><th>Enriched Judge</th><th>Rewrite Judge</th>
      <th>Basic NDCG@10</th><th>Enriched NDCG@10</th><th>Rewrite NDCG@10</th>
    </tr></thead><tbody>`;
  for (const qid of Object.keys(perQuery).sort()) {
    const q = perQuery[qid];
    const jVals = pipelineNames.map(p => q[p].judge);
    const maxJ = Math.max(...jVals);
    const nVals = pipelineNames.map(p => q[p].ndcg10);
    const maxN = Math.max(...nVals);
    pqTable += `<tr>
      <td style="font-weight:600;white-space:nowrap">${qid}</td>
      <td style="text-align:left;max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${q.text}">${q.text}</td>
      <td class="${jVals[0]===maxJ?'best-val':''}">${jVals[0].toFixed(1)}</td>
      <td class="${jVals[1]===maxJ?'best-val':''}">${jVals[1].toFixed(1)}</td>
      <td class="${jVals[2]===maxJ?'best-val':''}">${jVals[2].toFixed(1)}</td>
      <td class="${nVals[0]===maxN?'best-val':''}">${nVals[0].toFixed(4)}</td>
      <td class="${nVals[1]===maxN?'best-val':''}">${nVals[1].toFixed(4)}</td>
      <td class="${nVals[2]===maxN?'best-val':''}">${nVals[2].toFixed(4)}</td>
    </tr>`;
  }
  pqTable += '</tbody></table></div>';

  // Key findings
  const judgeBasic = roundTo(avg(allMetrics.basic_hybrid.judge), 4);
  const judgeEnriched = roundTo(avg(allMetrics.enriched_hybrid.judge), 4);
  const judgeRewrite = roundTo(avg(allMetrics.rewrite_enriched.judge), 4);
  const enrichImprovement = roundTo(((judgeEnriched - judgeBasic) / judgeBasic) * 100, 1);
  const rewriteVsEnriched = roundTo(((judgeRewrite - judgeEnriched) / judgeEnriched) * 100, 1);
  const mrrBasic = roundTo(avg(allMetrics.basic_hybrid.mrr), 4);
  const mrrEnriched = roundTo(avg(allMetrics.enriched_hybrid.mrr), 4);
  const mrrImprovement = roundTo(((mrrEnriched - mrrBasic) / mrrBasic) * 100, 1);

  // Count per-query wins
  let wins = { basic_hybrid: 0, enriched_hybrid: 0, rewrite_enriched: 0 };
  for (const qid of Object.keys(perQuery)) {
    const jVals = pipelineNames.map(p => perQuery[qid][p].judge);
    const maxJ = Math.max(...jVals);
    pipelineNames.forEach((p, i) => { if (jVals[i] === maxJ) wins[p]++; });
  }

  // Rewrite drift examples
  let driftCases = [];
  for (const qid of Object.keys(perQuery)) {
    const q = perQuery[qid];
    if (q.enriched_hybrid.judge > q.rewrite_enriched.judge + 0.5) {
      driftCases.push({ qid, text: q.text, enriched: q.enriched_hybrid.judge, rewrite: q.rewrite_enriched.judge });
    }
  }

  const old = document.getElementById('total-overlay');
  if (old) old.remove();

  const overlay = document.createElement('div');
  overlay.className = 'final-overlay';
  overlay.id = 'total-overlay';
  overlay.innerHTML = `
    <div class="final-card" style="max-width:1000px;max-height:90vh;overflow-y:auto">
      <h2>Complete Experiment Results</h2>
      <p class="final-subtitle">50 queries | 3 pipelines | 1,000 products | LLM-as-Judge evaluation</p>

      <div style="margin-bottom:1.5rem">
        <div style="font-size:0.8rem;font-weight:600;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.75rem">Aggregate Metrics</div>
        ${aggTable}
      </div>

      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.75rem;margin-bottom:1.5rem">
        <div style="text-align:center;padding:1rem;background:rgba(0,245,196,0.06);border-radius:8px;border:1px solid rgba(0,245,196,0.15)">
          <div style="font-size:0.65rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.6px;margin-bottom:0.4rem">Document Enrichment Impact</div>
          <div style="font-size:1.5rem;font-weight:800;color:var(--accent-teal)">+${enrichImprovement}%</div>
          <div style="font-size:0.75rem;color:var(--text-secondary)">Judge Score improvement</div>
          <div style="font-size:0.7rem;color:var(--text-muted);margin-top:0.2rem">Zero query-time latency</div>
        </div>
        <div style="text-align:center;padding:1rem;background:rgba(167,139,250,0.06);border-radius:8px;border:1px solid rgba(167,139,250,0.15)">
          <div style="font-size:0.65rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.6px;margin-bottom:0.4rem">MRR Improvement</div>
          <div style="font-size:1.5rem;font-weight:800;color:var(--accent-purple)">+${mrrImprovement}%</div>
          <div style="font-size:0.75rem;color:var(--text-secondary)">First relevant result faster</div>
          <div style="font-size:0.7rem;color:var(--text-muted);margin-top:0.2rem">${mrrBasic.toFixed(4)} &#8594; ${mrrEnriched.toFixed(4)}</div>
        </div>
        <div style="text-align:center;padding:1rem;background:rgba(244,114,182,0.06);border-radius:8px;border:1px solid rgba(244,114,182,0.15)">
          <div style="font-size:0.65rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.6px;margin-bottom:0.4rem">Query Rewriting Marginal</div>
          <div style="font-size:1.5rem;font-weight:800;color:var(--pipeline-rewrite)">+${rewriteVsEnriched}%</div>
          <div style="font-size:0.75rem;color:var(--text-secondary)">Over enrichment alone</div>
          <div style="font-size:0.7rem;color:var(--text-muted);margin-top:0.2rem">${driftCases.length} rewrite drift cases</div>
        </div>
      </div>

      <div style="margin-bottom:1.5rem;padding:1rem;background:rgba(255,255,255,0.02);border-radius:8px;border:1px solid var(--border-glass)">
        <div style="font-size:0.8rem;font-weight:600;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.6rem">Per-Query Judge Score Wins (out of 50)</div>
        <div style="display:flex;gap:2rem;justify-content:center">
          <div style="text-align:center"><span style="font-size:1.3rem;font-weight:800;color:var(--pipeline-basic)">${wins.basic_hybrid}</span><div style="font-size:0.7rem;color:var(--text-muted)">Basic</div></div>
          <div style="text-align:center"><span style="font-size:1.3rem;font-weight:800;color:var(--pipeline-enriched)">${wins.enriched_hybrid}</span><div style="font-size:0.7rem;color:var(--text-muted)">Enriched</div></div>
          <div style="text-align:center"><span style="font-size:1.3rem;font-weight:800;color:var(--pipeline-rewrite)">${wins.rewrite_enriched}</span><div style="font-size:0.7rem;color:var(--text-muted)">Rewrite</div></div>
        </div>
      </div>

      ${driftCases.length > 0 ? `
      <div style="margin-bottom:1.5rem;padding:1rem;background:rgba(239,68,68,0.04);border-radius:8px;border:1px solid rgba(239,68,68,0.12)">
        <div style="font-size:0.8rem;font-weight:600;color:#ef4444;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.6rem">Rewrite Drift Cases (enriched beat rewrite)</div>
        <div style="font-size:0.75rem;color:var(--text-secondary);line-height:1.7">
          ${driftCases.slice(0, 8).map(d => `<div><span style="color:var(--accent-cyan);font-weight:600">${d.qid}</span> "${d.text}" &mdash; Enriched: <span style="color:var(--score-4)">${d.enriched.toFixed(1)}</span> vs Rewrite: <span style="color:var(--score-2)">${d.rewrite.toFixed(1)}</span></div>`).join('')}
        </div>
      </div>` : ''}

      <div style="margin-bottom:1rem">
        <div style="font-size:0.8rem;font-weight:600;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.5rem">Per-Query Breakdown</div>
        ${pqTable}
      </div>

      <div style="text-align:center;margin-top:1.5rem;padding:1rem;background:rgba(0,245,196,0.06);border-radius:8px;border:1px solid rgba(0,245,196,0.15)">
        <div style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.5rem">Practical Recommendation</div>
        <div style="font-size:1rem;color:var(--accent-teal);font-weight:600">Document-side enrichment is the highest-value intervention for production</div>
        <div style="font-size:0.8rem;color:var(--text-secondary);margin-top:0.3rem">Query rewriting helps case-by-case but introduces rewrite drift risk on precise queries</div>
      </div>

      <button class="btn btn-primary final-close" onclick="document.getElementById('total-overlay').remove()">Close</button>
    </div>
  `;
  document.body.appendChild(overlay);
}


// ═══════════════════════════════════════════════════════════════════════
//  PRODUCT DATABASE
// ═══════════════════════════════════════════════════════════════════════

let dbPage = 0;
const DB_PAGE_SIZE = 24;
let filteredProducts = [];

function initProductDB() {
  filteredProducts = PRODUCTS.map((p, i) => ({ ...p, _idx: i }));
  renderCategoryChart();
  renderProductList();
  document.getElementById('db-search').addEventListener('input', debounce(filterProducts, 200));
  document.getElementById('db-category').addEventListener('change', filterProducts);

  const cats = [...new Set(PRODUCTS.map(p => p.category))].sort();
  const sel = document.getElementById('db-category');
  cats.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c; opt.textContent = c;
    sel.appendChild(opt);
  });
}

function filterProducts() {
  const search = document.getElementById('db-search').value.toLowerCase().trim();
  const cat = document.getElementById('db-category').value;
  filteredProducts = PRODUCTS.map((p, i) => ({ ...p, _idx: i })).filter(p => {
    if (cat && p.category !== cat) return false;
    if (search) {
      const hay = `${p.name} ${p.category} ${p.description} ${p.materials.join(' ')} ${p.reuse_applications.join(' ')}`.toLowerCase();
      return hay.includes(search);
    }
    return true;
  });
  dbPage = 0;
  renderProductList();
}

function renderProductList() {
  const total = filteredProducts.length;
  const totalPages = Math.ceil(total / DB_PAGE_SIZE);
  const start = dbPage * DB_PAGE_SIZE;
  const page = filteredProducts.slice(start, start + DB_PAGE_SIZE);

  document.getElementById('db-stats').textContent =
    `Showing ${Math.min(start + 1, total)}-${Math.min(start + DB_PAGE_SIZE, total)} of ${total} products`;

  const grid = document.getElementById('products-grid');
  grid.innerHTML = page.map(p => `
    <div class="product-card">
      <div class="product-name">${p.name}</div>
      <div class="product-category">${p.category}</div>
      <div class="product-meta">
        <span class="product-tag">${p.condition}</span>
        <span class="product-tag">${p.weight_kg} kg</span>
        <span class="product-tag">${p.dimensions}</span>
        ${p.materials.map(m => `<span class="product-tag">${m}</span>`).join('')}
      </div>
      <div class="product-desc">${p.description}</div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:0.5rem;flex-wrap:wrap;gap:0.3rem">
        <div class="product-price">&#8377;${p.price_per_unit_inr.toFixed(2)}</div>
        <div class="product-meta" style="margin:0">
          ${(p.reuse_applications || []).slice(0, 2).map(a =>
            `<span class="product-tag" style="border-color:rgba(0,212,255,0.15);color:var(--accent-cyan)">${a}</span>`
          ).join('')}
        </div>
      </div>
    </div>
  `).join('');

  // pagination
  const pagDiv = document.getElementById('db-pagination');
  if (totalPages <= 1) { pagDiv.innerHTML = ''; return; }
  let pagHtml = '';
  const maxBtns = 7;
  let startP = Math.max(0, dbPage - Math.floor(maxBtns / 2));
  let endP = Math.min(totalPages, startP + maxBtns);
  if (endP - startP < maxBtns) startP = Math.max(0, endP - maxBtns);
  if (dbPage > 0) pagHtml += `<button class="page-btn" onclick="goToPage(${dbPage - 1})">&#8249;</button>`;
  for (let i = startP; i < endP; i++) {
    pagHtml += `<button class="page-btn ${i === dbPage ? 'active' : ''}" onclick="goToPage(${i})">${i + 1}</button>`;
  }
  if (dbPage < totalPages - 1) pagHtml += `<button class="page-btn" onclick="goToPage(${dbPage + 1})">&#8250;</button>`;
  pagDiv.innerHTML = pagHtml;
}

function goToPage(p) { dbPage = p; renderProductList(); }

function renderCategoryChart() {
  const cats = {};
  PRODUCTS.forEach(p => { cats[p.category] = (cats[p.category] || 0) + 1; });
  const sorted = Object.entries(cats).sort((a, b) => b[1] - a[1]);
  const max = sorted[0][1];
  const container = document.getElementById('cat-chart');
  container.innerHTML = sorted.map(([name, count]) => `
    <div class="cat-bar-row">
      <span class="cat-bar-label" title="${name}">${name}</span>
      <div class="cat-bar-track"><div class="cat-bar-fill" style="width:${(count / max) * 100}%"></div></div>
      <span class="cat-bar-count">${count}</span>
    </div>
  `).join('');
}

function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

// ── Boot ──
document.addEventListener('DOMContentLoaded', loadData);
