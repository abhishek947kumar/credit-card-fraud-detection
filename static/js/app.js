/**
 * FraudGuard.AI - Main Client Application Logic
 */

// Application State
const state = {
  activeModel: 'xgboost',
  isStreamRunning: true,
  streamTimer: null,
  ws: null,
  transactions: [],
  kpis: {
    totalProcessed: 0,
    fraudCount: 0,
    savingsUsd: 0,
    avgLatency: 0,
  },
  categoriesCount: {},
  modelsMetadata: null,
  selectedTxn: null,
};

// Attack & Normal Presets
const PRESETS = {
  cloning: {
    amount: 1450.00,
    category: 'luxury_goods',
    dist_last: 4850.0,
    time_since: 22.0, // 13,200 km/h teleportation!
    vel_1h: 2,
    vel_24h: 4,
    ratio: 18.5,
    device_trust: 35,
    foreign: "1",
    failed_pin: 1,
  },
  testing: {
    amount: 2.25,
    category: 'online_retail',
    dist_last: 15.0,
    time_since: 0.4, // rapid automated bot
    vel_1h: 12,
    vel_24h: 18,
    ratio: 0.1,
    device_trust: 20,
    foreign: "1",
    failed_pin: 2,
  },
  crypto: {
    amount: 3850.00,
    category: 'crypto_forex',
    dist_last: 450.0,
    time_since: 35.0,
    vel_1h: 4,
    vel_24h: 7,
    ratio: 15.2,
    device_trust: 12,
    foreign: "1",
    failed_pin: 2,
  },
  pin_brute: {
    amount: 890.00,
    category: 'electronics',
    dist_last: 110.0,
    time_since: 15.0,
    vel_1h: 3,
    vel_24h: 5,
    ratio: 8.0,
    device_trust: 15,
    foreign: "1",
    failed_pin: 3,
  },
  coffee: {
    amount: 4.75,
    category: 'restaurant_dining',
    dist_last: 1.5,
    time_since: 340.0,
    vel_1h: 0,
    vel_24h: 2,
    ratio: 0.85,
    device_trust: 98,
    foreign: "0",
    failed_pin: 0,
  },
  grocery: {
    amount: 142.50,
    category: 'grocery',
    dist_last: 4.2,
    time_since: 180.0,
    vel_1h: 1,
    vel_24h: 3,
    ratio: 1.15,
    device_trust: 92,
    foreign: "0",
    failed_pin: 0,
  }
};

// DOM Elements
const elements = {
  tabs: document.querySelectorAll('.nav-tab-btn'),
  tabContents: document.querySelectorAll('.tab-content'),
  modelSelect: document.getElementById('modelSelect'),
  feedContainer: document.getElementById('liveFeedContainer'),
  btnToggleStream: document.getElementById('btnToggleStream'),
  streamToggleText: document.getElementById('streamToggleText'),
  streamToggleIcon: document.getElementById('streamToggleIcon'),
  btnEmitSingle: document.getElementById('btnEmitSingle'),
  btnAttackBurst: document.getElementById('btnAttackBurst'),
  btnRetrain: document.getElementById('btnRetrain'),
  simulatorForm: document.getElementById('simulatorForm'),
  analystTableBody: document.getElementById('analystTableBody'),
  modelsTableBody: document.getElementById('modelsTableBody'),
  // KPIs
  kpiTotalProcessed: document.getElementById('kpiTotalProcessed'),
  kpiFraudCount: document.getElementById('kpiFraudCount'),
  kpiFraudRate: document.getElementById('kpiFraudRate'),
  kpiSavingsUsd: document.getElementById('kpiSavingsUsd'),
  kpiLatency: document.getElementById('kpiLatency'),
  // Modal
  modal: document.getElementById('investigationModal'),
  modalCloseBtn: document.getElementById('modalCloseBtn'),
  btnModalApprove: document.getElementById('btnModalApprove'),
  btnModalConfirmFraud: document.getElementById('btnModalConfirmFraud'),
  toastContainer: document.getElementById('toastContainer'),
};

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
  initDashboardCharts();
  setupTabs();
  setupPresets();
  setupFormSubmit();
  setupModelSelector();
  setupStreamControls();
  setupModal();
  setupRoiCalculator();

  // Load Initial Data
  await loadModelMetadata();
  await loadRecentTransactions();
  startWebSocket();
  startSimulatedFeedTicker();
});

// Tab Switching
function setupTabs() {
  elements.tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      elements.tabs.forEach((t) => t.classList.remove('active'));
      elements.tabContents.forEach((c) => c.classList.remove('active'));

      tab.classList.add('active');
      const targetId = tab.getAttribute('data-tab');
      const targetContent = document.getElementById(targetId);
      if (targetContent) targetContent.classList.add('active');
    });
  });
}

// Preset Loader
function setupPresets() {
  document.querySelectorAll('.preset-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const key = chip.getAttribute('data-preset');
      const p = PRESETS[key];
      if (!p) return;

      document.getElementById('formAmount').value = p.amount.toFixed(2);
      document.getElementById('formCategory').value = p.category;
      document.getElementById('formDistLast').value = p.dist_last;
      document.getElementById('formTimeLast').value = p.time_since;
      document.getElementById('formVel1h').value = p.vel_1h;
      document.getElementById('formVel24h').value = p.vel_24h;
      document.getElementById('formRatio').value = p.ratio;
      document.getElementById('formDeviceTrust').value = p.device_trust;
      document.getElementById('formForeign').value = p.foreign;
      document.getElementById('formFailedPin').value = p.failed_pin;

      showToast(`Loaded "${chip.textContent.trim()}" preset values!`, 'cyan');
    });
  });
}

// Interactive Manual Tester Form Submission
function setupFormSubmit() {
  elements.simulatorForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const payload = {
      amount: parseFloat(document.getElementById('formAmount').value) || 50.0,
      category: document.getElementById('formCategory').value,
      distance_from_last_txn: parseFloat(document.getElementById('formDistLast').value) || 2.0,
      time_since_last_txn: parseFloat(document.getElementById('formTimeLast').value) || 120.0,
      velocity_1h: parseInt(document.getElementById('formVel1h').value) || 0,
      velocity_24h: parseInt(document.getElementById('formVel24h').value) || 1,
      amount_to_user_mean_ratio: parseFloat(document.getElementById('formRatio').value) || 1.0,
      device_trust_score: parseFloat(document.getElementById('formDeviceTrust').value) || 85.0,
      foreign_transaction: parseInt(document.getElementById('formForeign').value) || 0,
      failed_pin_attempts_24h: parseInt(document.getElementById('formFailedPin').value) || 0,
    };

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error('Prediction API call failed');
      const data = await res.json();
      renderAssessmentResult(data);
      showToast(`Scored: ${data.action} (${data.risk_score}%)`, data.badge_color);
    } catch (err) {
      showToast(err.message, 'rose');
    }
  });
}

function renderAssessmentResult(data) {
  const scoreElem = document.getElementById('resultScore');
  const probElem = document.getElementById('resultProb');
  const latencyBadge = document.getElementById('resultLatencyBadge');
  const banner = document.getElementById('resultActionBanner');
  const actionTitle = document.getElementById('resultActionTitle');
  const actionDesc = document.getElementById('resultActionDesc');
  const factorsList = document.getElementById('resultFactorsList');

  // Gauge & Prob
  scoreElem.textContent = `${data.risk_score}%`;
  scoreElem.className = `gauge-score ${data.risk_tier.toLowerCase()}`;
  probElem.textContent = data.fraud_probability.toFixed(4);
  latencyBadge.textContent = `Scored in: ${data.latency_ms} ms`;

  // Action Banner
  banner.className = 'risk-action-banner';
  if (data.action === 'AUTO_APPROVE') {
    banner.classList.add('auto-approve');
  } else if (data.action === 'STEP_UP_AUTH') {
    banner.classList.add('step-up');
  } else {
    banner.classList.add('freeze');
  }
  actionTitle.textContent = data.action.replace(/_/g, ' ');
  actionDesc.textContent = data.action_description;

  // Factors List
  factorsList.innerHTML = '';
  if (data.risk_factors && data.risk_factors.length > 0) {
    data.risk_factors.forEach((f) => {
      const item = document.createElement('div');
      item.className = 'factor-item';
      item.innerHTML = `
        <div class="factor-header">
          <span class="factor-name">${f.factor}</span>
          <span class="severity-pill sev-${f.severity.toLowerCase()}">${f.severity}</span>
        </div>
        <div class="factor-detail">${f.detail}</div>
      `;
      factorsList.appendChild(item);
    });
  } else {
    factorsList.innerHTML = `
      <div class="factor-item">
        <div class="factor-header">
          <span class="factor-name">Normal Cardholder Telemetry</span>
          <span class="severity-pill sev-low">LOW RISK</span>
        </div>
        <div class="factor-detail">Transaction patterns align with verified cardholder baseline behavior.</div>
      </div>
    `;
  }
}

// Model Switching
function setupModelSelector() {
  elements.modelSelect.addEventListener('change', async (e) => {
    const selectedModel = e.target.value;
    try {
      const res = await fetch('/api/models/switch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: selectedModel }),
      });
      const data = await res.json();
      state.activeModel = data.active_model;
      showToast(`Active ML engine switched to ${selectedModel.toUpperCase()}`, 'cyan');
      updateActiveTableRow(selectedModel);
    } catch (err) {
      showToast(`Failed to switch model: ${err.message}`, 'rose');
    }
  });

  elements.btnRetrain.addEventListener('click', async () => {
    elements.btnRetrain.disabled = true;
    elements.btnRetrain.innerHTML = '<span>⏳</span> Retraining Models...';
    showToast('Retraining all machine learning models on synthetic pipeline...', 'cyan');

    try {
      const res = await fetch('/api/retrain', { method: 'POST' });
      const data = await res.json();
      showToast(data.message, 'emerald');
      await loadModelMetadata();
    } catch (err) {
      showToast(`Retrain error: ${err.message}`, 'rose');
    } finally {
      elements.btnRetrain.disabled = false;
      elements.btnRetrain.innerHTML = '<span>🔄</span> Trigger Retraining Pipeline';
    }
  });
}

// Live Stream Feed & Controls
function setupStreamControls() {
  elements.btnToggleStream.addEventListener('click', () => {
    state.isStreamRunning = !state.isStreamRunning;
    if (state.isStreamRunning) {
      elements.streamToggleText.textContent = 'Pause Stream';
      elements.streamToggleIcon.textContent = '⏸️';
      showToast('Live authorization stream resumed', 'cyan');
    } else {
      elements.streamToggleText.textContent = 'Resume Stream';
      elements.streamToggleIcon.textContent = '▶️';
      showToast('Live authorization stream paused', 'amber');
    }
  });

  elements.btnEmitSingle.addEventListener('click', async () => {
    try {
      await fetch('/api/simulator/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ count: 1 }),
      });
    } catch (e) {
      console.error(e);
    }
  });

  elements.btnAttackBurst.addEventListener('click', async () => {
    elements.btnAttackBurst.disabled = true;
    showToast('Simulating aggressive 5-transaction fraud attack burst...', 'rose');
    try {
      await fetch('/api/simulator/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ count: 5, force_fraud: true }),
      });
    } catch (e) {
      console.error(e);
    } finally {
      setTimeout(() => {
        elements.btnAttackBurst.disabled = false;
      }, 1500);
    }
  });
}

// WebSocket Connection
function startWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/stream`;

  try {
    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
      console.log('WebSocket stream connected.');
      document.getElementById('systemStatusText').textContent = 'REAL-TIME STREAM: ACTIVE (WS)';
    };

    state.ws.onmessage = (event) => {
      try {
        const txn = JSON.parse(event.data);
        handleIncomingTransaction(txn);
      } catch (err) {
        console.warn('Non-JSON WS message:', event.data);
      }
    };

    state.ws.onclose = () => {
      console.log('WS closed, falling back to polling.');
      document.getElementById('systemStatusText').textContent = 'REAL-TIME INFERENCE: ACTIVE (POLL)';
      setTimeout(startWebSocket, 5000);
    };
  } catch (err) {
    console.warn('WS Init error:', err);
  }
}

// Periodic simulator feed ticker
function startSimulatedFeedTicker() {
  state.streamTimer = setInterval(async () => {
    if (!state.isStreamRunning) return;
    try {
      await fetch('/api/simulator/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ count: 1 }),
      });
    } catch (e) {
      // background ticker failure ignored
    }
  }, 2800);
}

// Handle Incoming Transaction in UI
function handleIncomingTransaction(txn) {
  state.transactions.unshift(txn);
  if (state.transactions.length > 100) state.transactions.pop();

  // Update KPI counters
  state.kpis.totalProcessed += 1;
  if (txn.risk_tier === 'HIGH') {
    state.kpis.fraudCount += 1;
    state.kpis.savingsUsd += txn.amount || 0;
  }
  updateKpiDisplay(txn.latency_ms);

  // Update Category Count
  const cat = txn.category || 'other';
  state.categoriesCount[cat] = (state.categoriesCount[cat] || 0) + 1;
  updateCategoryChart(state.categoriesCount);

  // Update Risk Doughnut
  const low = state.transactions.filter((t) => t.risk_tier === 'LOW').length;
  const med = state.transactions.filter((t) => t.risk_tier === 'MEDIUM').length;
  const high = state.transactions.filter((t) => t.risk_tier === 'HIGH').length;
  updateRiskDoughnut(low, med, high);

  // Render to Live Feed
  prependStreamCard(txn);

  // If High/Medium risk, add to Analyst Review Table
  if (txn.risk_tier === 'HIGH' || txn.risk_tier === 'MEDIUM') {
    prependAnalystRow(txn);
  }
}

// Prepend Stream Card
function prependStreamCard(txn) {
  const card = document.createElement('div');
  card.className = `stream-card tier-${txn.risk_tier.toLowerCase()} new-item`;

  const tierBadgeClass = `badge-${txn.risk_tier.toLowerCase()}`;
  const tierIcon = txn.risk_tier === 'HIGH' ? '🚨' : txn.risk_tier === 'MEDIUM' ? '⚠️' : '✅';

  const categoryName = (txn.category || 'general').replace(/_/g, ' ');
  const primaryAlert = (txn.risk_factors && txn.risk_factors.length > 0)
    ? txn.risk_factors[0].factor
    : 'Routine Spending';

  card.innerHTML = `
    <div class="card-left">
      <span class="card-badge ${tierBadgeClass}">
        <span>${tierIcon}</span> ${txn.risk_tier}
      </span>
      <div class="card-info">
        <div class="info-top">
          <span class="txn-id">${txn.transaction_id}</span>
          <span class="user-tag">${txn.user_id}</span>
        </div>
        <div class="info-sub">
          <span style="text-transform: capitalize;">🛒 ${categoryName}</span>
          <span style="color: ${txn.risk_tier === 'HIGH' ? 'var(--rose)' : 'var(--text-muted)'}; font-weight: 500;">
            ${primaryAlert}
          </span>
        </div>
      </div>
    </div>
    <div class="card-right">
      <div class="card-amount">$${(txn.amount || 0).toFixed(2)}</div>
      <div class="card-score">Score: ${txn.risk_score}% (${txn.latency_ms}ms)</div>
    </div>
  `;

  card.addEventListener('click', () => openModal(txn));

  elements.feedContainer.insertBefore(card, elements.feedContainer.firstChild);

  // Keep max 60 cards in DOM
  if (elements.feedContainer.children.length > 60) {
    elements.feedContainer.removeChild(elements.feedContainer.lastChild);
  }
}

// Prepend Analyst Queue Row
function prependAnalystRow(txn) {
  const tr = document.createElement('tr');
  const alertText = (txn.risk_factors && txn.risk_factors.length > 0)
    ? txn.risk_factors[0].factor
    : 'Anomaly Alert';

  tr.innerHTML = `
    <td style="color: var(--cyan);">${txn.transaction_id}</td>
    <td style="color: var(--text-muted); font-size: 0.75rem;">${txn.timestamp || 'Just now'}</td>
    <td>${txn.user_id}</td>
    <td style="font-weight: 700;">$${(txn.amount || 0).toFixed(2)}</td>
    <td style="text-transform: capitalize;">${(txn.category || '').replace(/_/g, ' ')}</td>
    <td><span class="card-badge badge-${txn.risk_tier.toLowerCase()}">${txn.risk_tier}</span></td>
    <td style="font-weight: 700; color: ${txn.risk_tier === 'HIGH' ? 'var(--rose)' : 'var(--amber)'};">${txn.risk_score}%</td>
    <td style="font-size: 0.78rem;">${alertText}</td>
    <td>
      <button class="btn btn-secondary btn-sm" onclick="handleTableInspect('${txn.transaction_id}')">
        Inspect XAI
      </button>
    </td>
  `;

  elements.analystTableBody.insertBefore(tr, elements.analystTableBody.firstChild);
  if (elements.analystTableBody.children.length > 30) {
    elements.analystTableBody.removeChild(elements.analystTableBody.lastChild);
  }
}

// Global inspector hook
window.handleTableInspect = function (txnId) {
  const txn = state.transactions.find((t) => t.transaction_id === txnId);
  if (txn) openModal(txn);
};

// Modal Handling
function setupModal() {
  elements.modalCloseBtn.addEventListener('click', closeModal);
  elements.modal.addEventListener('click', (e) => {
    if (e.target === elements.modal) closeModal();
  });

  elements.btnModalApprove.addEventListener('click', async () => {
    if (!state.selectedTxn) return;
    try {
      await fetch(`/api/transactions/${state.selectedTxn.transaction_id}/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'APPROVE', analyst_notes: 'Cleared by Fraud Analyst' }),
      });
      showToast(`Transaction ${state.selectedTxn.transaction_id} APPROVED`, 'emerald');
      closeModal();
    } catch (e) {
      showToast(e.message, 'rose');
    }
  });

  elements.btnModalConfirmFraud.addEventListener('click', async () => {
    if (!state.selectedTxn) return;
    try {
      await fetch(`/api/transactions/${state.selectedTxn.transaction_id}/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'CONFIRM_FRAUD', analyst_notes: 'Card Blocked & Dispute Filed' }),
      });
      showToast(`Card locked & Fraud Confirmed for ${state.selectedTxn.transaction_id}`, 'rose');
      closeModal();
    } catch (e) {
      showToast(e.message, 'rose');
    }
  });
}

function openModal(txn) {
  state.selectedTxn = txn;
  document.getElementById('modalTxnTitle').textContent = `Transaction: ${txn.transaction_id}`;
  document.getElementById('modalBadge').className = `card-badge badge-${txn.risk_tier.toLowerCase()}`;
  document.getElementById('modalBadge').textContent = `${txn.risk_tier} RISK`;

  document.getElementById('modalUserId').textContent = txn.user_id || 'USR-XXXX';
  document.getElementById('modalTimestamp').textContent = txn.timestamp || 'N/A';
  document.getElementById('modalAmount').textContent = `$${(txn.amount || 0).toFixed(2)}`;
  document.getElementById('modalCategory').textContent = (txn.category || '').replace(/_/g, ' ');
  
  const dLast = txn.details ? txn.details.distance_from_last_txn : (txn.distance_from_last_txn || 0);
  const tLast = txn.details ? txn.details.time_since_last_txn : (txn.time_since_last_txn || 0);
  document.getElementById('modalGeo').textContent = `${dLast} km (${tLast} min elapsed)`;

  const v1 = txn.details ? txn.details.velocity_1h : (txn.velocity_1h || 0);
  const v24 = txn.details ? txn.details.velocity_24h : (txn.velocity_24h || 0);
  document.getElementById('modalVelocity').textContent = `${v1} txns (1h) / ${v24} txns (24h)`;

  const dev = txn.details ? txn.details.device_trust_score : (txn.device_trust_score || 90);
  document.getElementById('modalDeviceTrust').textContent = `${dev}/100`;

  document.getElementById('modalScore').textContent = `${txn.risk_score}% (${txn.action})`;

  // Render Factors
  const factorsContainer = document.getElementById('modalFactorsList');
  factorsContainer.innerHTML = '';
  if (txn.risk_factors && txn.risk_factors.length > 0) {
    txn.risk_factors.forEach((f) => {
      const item = document.createElement('div');
      item.className = 'factor-item';
      item.innerHTML = `
        <div class="factor-header">
          <span class="factor-name">${f.factor}</span>
          <span class="severity-pill sev-${f.severity.toLowerCase()}">${f.severity}</span>
        </div>
        <div class="factor-detail">${f.detail}</div>
      `;
      factorsContainer.appendChild(item);
    });
  } else {
    factorsContainer.innerHTML = `
      <div class="factor-item">
        <div class="factor-header">
          <span class="factor-name">Normal Spending Profile</span>
          <span class="severity-pill sev-low">LOW RISK</span>
        </div>
        <div class="factor-detail">Telemetry values match expected domestic cardholder baseline.</div>
      </div>
    `;
  }

  elements.modal.classList.add('active');
}

function closeModal() {
  elements.modal.classList.remove('active');
  state.selectedTxn = null;
}

// KPI Display
function updateKpiDisplay(latestLatency) {
  elements.kpiTotalProcessed.textContent = state.kpis.totalProcessed.toLocaleString();
  elements.kpiFraudCount.textContent = state.kpis.fraudCount.toLocaleString();

  const rate = state.kpis.totalProcessed > 0
    ? ((state.kpis.fraudCount / state.kpis.totalProcessed) * 100).toFixed(1)
    : '0.0';
  elements.kpiFraudRate.textContent = `${rate}%`;

  elements.kpiSavingsUsd.textContent = `$${Math.round(state.kpis.savingsUsd).toLocaleString()}`;
  if (latestLatency) {
    elements.kpiLatency.textContent = `${latestLatency.toFixed(1)} ms`;
  }
}

// Model Metadata & Comparison Table
async function loadModelMetadata() {
  try {
    const res = await fetch('/api/models');
    if (!res.ok) return;
    const data = await res.json();
    state.modelsMetadata = data;
    renderModelsTable(data);
  } catch (err) {
    console.error('Failed to load model metadata:', err);
  }
}

function renderModelsTable(meta) {
  elements.modelsTableBody.innerHTML = '';
  const models = meta.models || {};

  let defaultPrPoints = [];
  let defaultRocPoints = [];

  for (const [key, m] of Object.entries(models)) {
    const isActive = key === meta.current_active_model;
    const tr = document.createElement('tr');
    if (isActive) tr.className = 'active-row';

    if (isActive && m.pr_curve) {
      defaultPrPoints = m.pr_curve;
      defaultRocPoints = m.roc_curve;
    }

    const netSavings = m.financial_impact ? m.financial_impact.net_savings_usd : 0;

    tr.innerHTML = `
      <td style="font-weight: 700; color: #FFFFFF;">
        ${m.model_name} ${isActive ? '<span style="color: var(--cyan); font-size: 0.72rem;">[ACTIVE]</span>' : ''}
      </td>
      <td style="color: var(--emerald); font-weight: 700;">${(m.recall * 100).toFixed(1)}%</td>
      <td>${(m.precision * 100).toFixed(1)}%</td>
      <td>${(m.f1 * 100).toFixed(1)}%</td>
      <td style="color: var(--amber); font-weight: 600;">${(m.f2 * 100).toFixed(1)}%</td>
      <td style="color: var(--cyan); font-weight: 700;">${m.pr_auc.toFixed(3)}</td>
      <td>${m.roc_auc.toFixed(3)}</td>
      <td>${m.latency_ms} ms</td>
      <td style="color: var(--emerald); font-weight: 700;">$${netSavings.toLocaleString()}</td>
      <td>
        <button class="btn btn-sm ${isActive ? 'btn-secondary' : 'btn-primary'}" onclick="handleActivateModel('${key}')" ${isActive ? 'disabled' : ''}>
          ${isActive ? 'Active' : 'Activate'}
        </button>
      </td>
    `;
    elements.modelsTableBody.appendChild(tr);
  }

  // Update PR / ROC Charts with active model curves
  if (defaultPrPoints.length) {
    updateModelCurves(defaultPrPoints, defaultRocPoints);
  }
}

function updateActiveTableRow(modelKey) {
  if (state.modelsMetadata) {
    state.modelsMetadata.current_active_model = modelKey;
    renderModelsTable(state.modelsMetadata);
  }
}

window.handleActivateModel = async function (modelKey) {
  elements.modelSelect.value = modelKey;
  elements.modelSelect.dispatchEvent(new Event('change'));
};

// Initial Transactions Load
async function loadRecentTransactions() {
  try {
    const res = await fetch('/api/transactions/recent?limit=25');
    if (!res.ok) return;
    const data = await res.json();
    if (data.transactions && data.transactions.length > 0) {
      data.transactions.reverse().forEach((txn) => {
        handleIncomingTransaction(txn);
      });
    }
  } catch (err) {
    console.error('Failed to load recent transactions:', err);
  }
}

// Financial ROI Calculator
function setupRoiCalculator() {
  const volInput = document.getElementById('calcVolume');
  const rateInput = document.getElementById('calcRate');
  const ticketInput = document.getElementById('calcTicket');

  const lossElem = document.getElementById('calcLossNoML');
  const savingsElem = document.getElementById('calcNetSavings');
  const roiElem = document.getElementById('calcRoiPercent');

  function calculate() {
    const volume = parseFloat(volInput.value) || 250000;
    const rate = (parseFloat(rateInput.value) || 0.8) / 100.0;
    const ticket = parseFloat(ticketInput.value) || 385.0;

    const annualFrauds = volume * 12 * rate;
    const lossWithoutML = annualFrauds * ticket;
    
    // With 99.5% detection and minimal false alarms
    const saved = lossWithoutML * 0.993;
    const roiReduction = 99.3;

    lossElem.textContent = `$${Math.round(lossWithoutML).toLocaleString()}`;
    savingsElem.textContent = `$${Math.round(saved).toLocaleString()}`;
    roiElem.textContent = `${roiReduction.toFixed(1)}%`;
  }

  volInput.addEventListener('input', calculate);
  rateInput.addEventListener('input', calculate);
  ticketInput.addEventListener('input', calculate);
  calculate();
}

// Toast Notifications
function showToast(message, color = 'cyan') {
  const toast = document.createElement('div');
  toast.className = 'toast';
  if (color === 'rose') toast.style.borderColor = 'var(--rose)';
  if (color === 'emerald') toast.style.borderColor = 'var(--emerald)';
  if (color === 'amber') toast.style.borderColor = 'var(--amber)';
  toast.textContent = message;

  elements.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}
