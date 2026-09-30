/**
 * Charts Engine for Fraud Detection Platform
 * Manages Chart.js instances for Risk Distribution, Merchant Categories, PR Curves, and ROC Curves.
 */

// Global Chart instances
let riskDoughnutChart = null;
let categoryBarChart = null;
let prCurveChart = null;
let rocCurveChart = null;

// Palette definitions
const CHART_COLORS = {
  emerald: '#10B981',
  amber: '#F59E0B',
  rose: '#F43F5E',
  cyan: '#06B6D4',
  blue: '#3B82F6',
  purple: '#8B5CF6',
  grid: 'rgba(255, 255, 255, 0.07)',
  text: '#94A3B8',
};

// Set Global Chart.js Defaults
if (window.Chart) {
  Chart.defaults.color = CHART_COLORS.text;
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.borderColor = CHART_COLORS.grid;
}

function initDashboardCharts() {
  // 1. Risk Tier Doughnut Chart
  const doughnutCtx = document.getElementById('riskDoughnutChart');
  if (doughnutCtx) {
    riskDoughnutChart = new Chart(doughnutCtx, {
      type: 'doughnut',
      data: {
        labels: ['Low Risk', 'Medium Risk', 'High Risk'],
        datasets: [{
          data: [20, 3, 2],
          backgroundColor: [CHART_COLORS.emerald, CHART_COLORS.amber, CHART_COLORS.rose],
          borderWidth: 0,
          hoverOffset: 6,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => ` ${ctx.label}: ${ctx.raw} txns`
            }
          }
        },
        cutout: '72%',
      }
    });
  }

  // 2. Category Bar Chart
  const catCtx = document.getElementById('categoryBarChart');
  if (catCtx) {
    categoryBarChart = new Chart(catCtx, {
      type: 'bar',
      data: {
        labels: ['Grocery', 'Retail', 'Crypto', 'Electro', 'Dining', 'Wire'],
        datasets: [{
          label: 'Txn Count',
          data: [12, 8, 3, 4, 9, 2],
          backgroundColor: 'rgba(6, 182, 212, 0.65)',
          borderRadius: 4,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false } },
          y: { grid: { color: CHART_COLORS.grid }, beginAtZero: true }
        }
      }
    });
  }

  // 3. PR Curve Chart
  const prCtx = document.getElementById('prCurveChart');
  if (prCtx) {
    prCurveChart = new Chart(prCtx, {
      type: 'line',
      data: {
        datasets: [{
          label: 'Precision-Recall Curve',
          data: [],
          borderColor: CHART_COLORS.cyan,
          backgroundColor: 'rgba(6, 182, 212, 0.15)',
          fill: true,
          tension: 0.2,
          pointRadius: 1,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: 'Recall (Sensitivity)' },
            min: 0,
            max: 1.0,
            grid: { color: CHART_COLORS.grid },
          },
          y: {
            title: { display: true, text: 'Precision' },
            min: 0,
            max: 1.05,
            grid: { color: CHART_COLORS.grid },
          }
        }
      }
    });
  }

  // 4. ROC Curve Chart
  const rocCtx = document.getElementById('rocCurveChart');
  if (rocCtx) {
    rocCurveChart = new Chart(rocCtx, {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'ROC Curve (ML Model)',
            data: [],
            borderColor: CHART_COLORS.emerald,
            backgroundColor: 'rgba(16, 185, 129, 0.12)',
            fill: true,
            tension: 0.2,
            pointRadius: 1,
          },
          {
            label: 'Random Guess Baseline',
            data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            borderColor: 'rgba(255, 255, 255, 0.25)',
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: 'False Positive Rate' },
            min: 0,
            max: 1.0,
            grid: { color: CHART_COLORS.grid },
          },
          y: {
            title: { display: true, text: 'True Positive Rate (Recall)' },
            min: 0,
            max: 1.05,
            grid: { color: CHART_COLORS.grid },
          }
        }
      }
    });
  }
}

function updateRiskDoughnut(lowCount, medCount, highCount) {
  if (!riskDoughnutChart) return;
  riskDoughnutChart.data.datasets[0].data = [lowCount, medCount, highCount];
  riskDoughnutChart.update();
}

function updateCategoryChart(catMap) {
  if (!categoryBarChart) return;
  const labels = Object.keys(catMap);
  const data = Object.values(catMap);
  categoryBarChart.data.labels = labels;
  categoryBarChart.data.datasets[0].data = data;
  categoryBarChart.update();
}

function updateModelCurves(prPoints, rocPoints) {
  if (prCurveChart && prPoints && prPoints.length) {
    prCurveChart.data.datasets[0].data = prPoints;
    prCurveChart.update();
  }
  if (rocCurveChart && rocPoints && rocPoints.length) {
    rocCurveChart.data.datasets[0].data = rocPoints;
    rocCurveChart.update();
  }
}
