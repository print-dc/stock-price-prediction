/* ─────────────────────────────────────────────────────────
   StockCNN Dashboard  ·  script.js
   ───────────────────────────────────────────────────────── */

'use strict';

// ── Chart instances ──────────────────────────────────────
let historyChart = null;
let batchChart   = null;
let lossChart    = null;
let accChart     = null;

// ── Common chart defaults ────────────────────────────────
Chart.defaults.color          = '#4a6080';
Chart.defaults.borderColor    = '#1a2540';
Chart.defaults.font.family    = "'Space Mono', monospace";
Chart.defaults.font.size      = 11;

const CHART_BG = '#0d1520';
const ACCENT   = '#00e5ff';
const ACCENT2  = '#ff6b35';
const UP       = '#00e676';
const DOWN     = '#ff1744';

function makeGradient(ctx, color, height = 300) {
  const g = ctx.createLinearGradient(0, 0, 0, height);
  g.addColorStop(0,   color.replace(')', ', 0.25)').replace('rgb', 'rgba'));
  g.addColorStop(1,   color.replace(')', ', 0)').replace('rgb', 'rgba'));
  return g;
}

function baseChartOptions(xLabel = '', yLabel = '') {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: {
        labels: { color: '#6a8090', boxWidth: 12, font: { size: 10 } },
      },
      tooltip: {
        backgroundColor: '#0a1520',
        borderColor: '#1a2540',
        borderWidth: 1,
        titleColor: '#00e5ff',
        bodyColor: '#c8d8e8',
      },
    },
    scales: {
      x: {
        ticks: { maxTicksLimit: 8, maxRotation: 0, color: '#3a5060' },
        grid:  { color: '#10202e' },
      },
      y: {
        ticks: { color: '#3a5060' },
        grid:  { color: '#10202e' },
      },
    },
  };
}

// ── Loader ───────────────────────────────────────────────

const loaderEl   = document.getElementById('loader');
const loaderText = document.getElementById('loaderText');
const statusDot  = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');

function showLoader(msg = 'Loading…') {
  loaderText.textContent = msg;
  loaderEl.classList.add('visible');
  setStatus('loading', msg);
}
function hideLoader() {
  loaderEl.classList.remove('visible');
}
function setStatus(state, text) {
  statusDot.className = 'status-dot ' + state;
  statusText.textContent = text;
}

// ── KPI helpers ──────────────────────────────────────────

function setKPI(cardId, value, extraClass = '') {
  const card = document.getElementById(cardId);
  const val  = card.querySelector('.kpi-value');
  val.textContent = value;
  val.className   = 'kpi-value ' + extraClass;
  card.classList.add('loaded');
}

// ── Destroy & recreate chart ─────────────────────────────

function destroyChart(instance) {
  if (instance) { try { instance.destroy(); } catch (_) {} }
  return null;
}

// ── History chart (90-day close) ─────────────────────────

function renderHistoryChart(dates, closes) {
  historyChart = destroyChart(historyChart);
  const ctx = document.getElementById('historyChart').getContext('2d');
  historyChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: dates,
      datasets: [{
        label: 'Close',
        data: closes,
        borderColor: ACCENT,
        borderWidth: 1.5,
        backgroundColor: makeGradient(ctx, 'rgb(0,229,255)'),
        pointRadius: 0,
        fill: true,
        tension: 0.3,
      }],
    },
    options: baseChartOptions('Date', 'Price (USD)'),
  });
}

// ── Batch (actual vs predicted) ──────────────────────────

function renderBatchChart(dates, actual, predicted) {
  batchChart = destroyChart(batchChart);
  const ctx  = document.getElementById('batchChart').getContext('2d');
  batchChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: dates,
      datasets: [
        {
          label: 'Actual',
          data: actual,
          borderColor: UP,
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.2,
        },
        {
          label: 'Predicted',
          data: predicted,
          borderColor: ACCENT2,
          borderWidth: 1.5,
          borderDash: [4, 3],
          pointRadius: 0,
          tension: 0.2,
        },
      ],
    },
    options: baseChartOptions('Date', 'Price (USD)'),
  });
}

// ── Loss curve ───────────────────────────────────────────

function renderLossChart(history) {
  lossChart = destroyChart(lossChart);
  const ctx  = document.getElementById('lossChart').getContext('2d');
  const epochs = history.loss ? history.loss.map((_, i) => i + 1) : [];

  lossChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: epochs,
      datasets: [
        {
          label: 'Train Loss',
          data: history.loss || [],
          borderColor: ACCENT,
          borderWidth: 1.5,
          pointRadius: 0,
        },
        {
          label: 'Val Loss',
          data: history.val_loss || [],
          borderColor: DOWN,
          borderWidth: 1.5,
          borderDash: [4, 3],
          pointRadius: 0,
        },
      ],
    },
    options: baseChartOptions('Epoch', 'Loss'),
  });
}

// ── Accuracy curve ───────────────────────────────────────

function renderAccChart(history) {
  accChart = destroyChart(accChart);
  const ctx   = document.getElementById('accChart').getContext('2d');
  const epochs = history.loss ? history.loss.map((_, i) => i + 1) : [];

  const trainAcc = history['trend_output_accuracy'] || [];
  const valAcc   = history['val_trend_output_accuracy'] || [];

  accChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: epochs,
      datasets: [
        {
          label: 'Train Acc',
          data: trainAcc,
          borderColor: ACCENT,
          borderWidth: 1.5,
          pointRadius: 0,
        },
        {
          label: 'Val Acc',
          data: valAcc,
          borderColor: UP,
          borderWidth: 1.5,
          borderDash: [4, 3],
          pointRadius: 0,
        },
      ],
    },
    options: {
      ...baseChartOptions('Epoch', 'Accuracy'),
      scales: {
        ...baseChartOptions().scales,
        y: {
          min: 0.4, max: 1.0,
          ticks: { color: '#3a5060', callback: v => (v * 100).toFixed(0) + '%' },
          grid: { color: '#10202e' },
        },
      },
    },
  });
}

// ── Main load ────────────────────────────────────────────

async function loadTicker(ticker) {
  ticker = ticker.trim().toUpperCase();
  if (!ticker) return;

  document.getElementById('statusTicker').textContent = ticker;
  showLoader(`Fetching ${ticker}…`);

  try {
    // 1. Latest prediction
    showLoader(`Predicting ${ticker}…`);
    const predRes = await fetch(`/api/predict/${ticker}`);
    if (!predRes.ok) {
      const err = await predRes.json();
      throw new Error(err.error || predRes.statusText);
    }
    const pred = await predRes.json();

    const isUp  = pred.trend === 'UP';
    const sign  = isUp ? '+' : '';

    setKPI('kpi-last',  `$${pred.last_close.toLocaleString('en-US', {minimumFractionDigits:2})}`);
    setKPI('kpi-pred',  `$${pred.predicted_price.toLocaleString('en-US', {minimumFractionDigits:2})}`);
    setKPI('kpi-chg',   `${sign}${pred.predicted_change_pct}%`, isUp ? 'up' : 'down');
    setKPI('kpi-trend', pred.trend, isUp ? 'up' : 'down');
    document.getElementById('kpi-conf').textContent = `${pred.trend_confidence}% confidence`;

    renderHistoryChart(pred.dates, pred.close_history);

    // 2. Batch predictions (test set)
    showLoader('Running back-test…');
    const batchRes = await fetch(`/api/batch/${ticker}`);
    if (batchRes.ok) {
      const batch = await batchRes.json();
      renderBatchChart(batch.dates, batch.actual, batch.predicted);
    }

    // 3. Metrics & training history
    showLoader('Loading metrics…');
    const metRes = await fetch(`/api/metrics/${ticker}`);
    if (metRes.ok) {
      const metData = await metRes.json();
      const acc = metData.metrics['trend_output_accuracy'];
      if (acc !== undefined) {
        setKPI('kpi-acc', `${(acc * 100).toFixed(1)}%`);
      }
      if (metData.history && Object.keys(metData.history).length > 0) {
        renderLossChart(metData.history);
        renderAccChart(metData.history);
      }
    }

    setStatus('active', `Loaded ${ticker} successfully`);

  } catch (err) {
    setStatus('error', `Error: ${err.message}`);
    console.error(err);
  } finally {
    hideLoader();
  }
}

// ── Event listeners ──────────────────────────────────────

document.getElementById('loadBtn').addEventListener('click', () => {
  loadTicker(document.getElementById('tickerInput').value);
});

document.getElementById('tickerInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') loadTicker(e.target.value);
});

// ── Live clock ───────────────────────────────────────────

function updateClock() {
  const now = new Date();
  document.getElementById('liveTime').textContent =
    now.toUTCString().replace('GMT', 'UTC');
}
setInterval(updateClock, 1000);
updateClock();

// ── Boot ─────────────────────────────────────────────────
// Auto-load if a trained model exists for default ticker
(async () => {
  const res = await fetch('/api/tickers').catch(() => null);
  if (res && res.ok) {
    const data  = await res.json();
    const first = data.tickers[0];
    if (first) {
      document.getElementById('tickerInput').value = first;
      loadTicker(first);
    }
  }
})();
