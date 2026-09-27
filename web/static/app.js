/**
 * THULI RESEARCH AGENT - INTERACTIVE FRONTEND APPLICATION
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const questionInput = document.getElementById('questionInput');
  const submitBtn = document.getElementById('submitBtn');
  const clearBtn = document.getElementById('clearBtn');
  const modeTabs = document.querySelectorAll('.mode-tab');
  const suggestionChips = document.querySelectorAll('.chip');
  const pipelineSection = document.getElementById('pipelineSection');
  const pipelineStatusText = document.getElementById('pipelineStatusText');
  const resultsDashboard = document.getElementById('resultsDashboard');
  const copyAnswerBtn = document.getElementById('copyAnswerBtn');
  const toastMessage = document.getElementById('toastMessage');

  // History Drawer Elements
  const toggleHistoryBtn = document.getElementById('toggleHistoryBtn');
  const closeHistoryBtn = document.getElementById('closeHistoryBtn');
  const historyOverlay = document.getElementById('historyOverlay');
  const historyDrawer = document.getElementById('historyDrawer');
  const historyList = document.getElementById('historyList');

  // Tab Elements
  const tabButtons = document.querySelectorAll('.res-tab');
  const tabPanes = document.querySelectorAll('.tab-pane');

  // Status Indicators
  const geminiStatusPill = document.getElementById('geminiStatusPill');
  const tavilyStatusPill = document.getElementById('tavilyStatusPill');

  // State
  let currentMode = 'NORMAL_ANALYST';
  let isExecuting = false;
  let activeStepInterval = null;

  // Initialize System Status
  checkSystemStatus();

  // Mode Selection
  modeTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      modeTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      currentMode = tab.dataset.mode;
    });
  });

  // Suggestion Chips
  suggestionChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const q = chip.dataset.query;
      questionInput.value = q;
      questionInput.focus();
      triggerResearch(q);
    });
  });

  // Input Clear Button
  clearBtn.addEventListener('click', () => {
    questionInput.value = '';
    questionInput.focus();
  });

  // Submit on Button Click or Ctrl+Enter
  submitBtn.addEventListener('click', () => {
    triggerResearch(questionInput.value);
  });

  questionInput.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
      e.preventDefault();
      triggerResearch(questionInput.value);
    }
  });

  // Tab Navigation
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetPane = document.getElementById(btn.dataset.target);
      if (targetPane) targetPane.classList.add('active');
    });
  });

  // Copy Answer Button
  copyAnswerBtn.addEventListener('click', () => {
    const text = document.getElementById('answerBody').innerText;
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      showToast('Answer copied to clipboard!');
    }).catch(() => {
      showToast('Failed to copy to clipboard.');
    });
  });

  // History Drawer Toggles
  toggleHistoryBtn.addEventListener('click', openHistoryDrawer);
  closeHistoryBtn.addEventListener('click', closeHistoryDrawer);
  historyOverlay.addEventListener('click', closeHistoryDrawer);

  function openHistoryDrawer() {
    historyOverlay.classList.add('active');
    historyDrawer.classList.add('active');
    loadHistoryRuns();
  }

  function closeHistoryDrawer() {
    historyOverlay.classList.remove('active');
    historyDrawer.classList.remove('active');
  }

  // Toast Notification
  function showToast(msg) {
    toastMessage.textContent = msg;
    toastMessage.classList.add('show');
    setTimeout(() => {
      toastMessage.classList.remove('show');
    }, 3200);
  }

  // Check System Readiness
  async function checkSystemStatus() {
    try {
      const res = await fetch('/api/status');
      if (!res.ok) return;
      const data = await res.json();

      if (data.has_gemini_key) {
        geminiStatusPill.innerHTML = `<span class="status-dot"></span><span>${data.gemini_model || 'Gemini 2.5'} Active</span>`;
      } else {
        geminiStatusPill.innerHTML = `<i class="fa-solid fa-cloud" style="color:#f59e0b;"></i><span>Deterministic Mode</span>`;
      }

      if (data.has_tavily_key) {
        tavilyStatusPill.innerHTML = `<span class="status-dot"></span><span>Live Tavily Search</span>`;
      } else {
        tavilyStatusPill.innerHTML = `<i class="fa-solid fa-server" style="color:#94a3b8;"></i><span>Corporate Knowledge</span>`;
      }
    } catch (e) {
      console.warn('Status check failed:', e);
    }
  }

  // Pipeline Stepper Simulation during async execution
  const STEPS = ['step-init', 'step-planner', 'step-search', 'step-fetch', 'step-extract', 'step-gate', 'step-analyst', 'step-auditor'];
  const STEP_MESSAGES = [
    'Initializing budget & operational memory...',
    'Planner formulating research sub-questions...',
    'Executing parallel web searches...',
    'Fetching primary disclosure pages...',
    'Extracting passage-level evidence...',
    'Enforcing strict Python claim gate...',
    'Analyst synthesizing verified answer...',
    'Auditor running independent fact verification...'
  ];

  function startPipelineAnimation() {
    pipelineSection.style.display = 'block';
    resultsDashboard.style.display = 'none';

    // Reset nodes
    STEPS.forEach(id => {
      const node = document.getElementById(id);
      if (node) {
        node.classList.remove('active', 'completed');
      }
    });

    let currentStepIdx = 0;
    const updateStep = () => {
      if (currentStepIdx < STEPS.length) {
        const prevId = STEPS[currentStepIdx - 1];
        if (prevId) {
          const prevNode = document.getElementById(prevId);
          if (prevNode) {
            prevNode.classList.remove('active');
            prevNode.classList.add('completed');
          }
        }
        const currNode = document.getElementById(STEPS[currentStepIdx]);
        if (currNode) currNode.classList.add('active');
        pipelineStatusText.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> ${STEP_MESSAGES[currentStepIdx]}`;
        currentStepIdx++;
      }
    };

    updateStep();
    activeStepInterval = setInterval(updateStep, 1800);
  }

  function finishPipelineAnimation() {
    clearInterval(activeStepInterval);
    STEPS.forEach(id => {
      const node = document.getElementById(id);
      if (node) {
        node.classList.remove('active');
        node.classList.add('completed');
      }
    });
    pipelineStatusText.innerHTML = `<i class="fa-solid fa-circle-check" style="color:var(--emerald);"></i> Research & Independent Audit Completed`;
  }

  // Trigger Research
  async function triggerResearch(query) {
    const q = query.trim();
    if (!q) {
      showToast('Please enter a research question.');
      questionInput.focus();
      return;
    }

    if (isExecuting) return;
    isExecuting = true;
    submitBtn.disabled = true;
    submitBtn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Running...`;

    startPipelineAnimation();

    try {
      const res = await fetch('/api/research', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: q,
          mode: currentMode,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server returned error ${res.status}`);
      }

      const data = await res.json();
      finishPipelineAnimation();
      renderResults(data);
    } catch (err) {
      clearInterval(activeStepInterval);
      pipelineStatusText.innerHTML = `<i class="fa-solid fa-triangle-exclamation" style="color:var(--rose);"></i> Execution Failed: ${err.message}`;
      showToast(`Error: ${err.message}`);
      console.error(err);
    } finally {
      isExecuting = false;
      submitBtn.disabled = false;
      submitBtn.innerHTML = `<span class="btn-label"><i class="fa-solid fa-bolt"></i> Run Research</span><span class="btn-shortcut">Ctrl+Enter</span>`;
    }
  }

  // Render Research Results
  function renderResults(data) {
    resultsDashboard.style.display = 'block';

    const answer = data.answer || {};
    const metrics = data.metrics || {};
    const audits = data.audit_results || [];
    const evidences = data.evidences || [];
    const claims = data.verified_claims || [];
    const searchResults = data.search_results || [];
    const fetchedPages = data.fetched_pages || {};
    const plan = data.plan || {};

    // 1. Metrics Bar
    document.getElementById('metricLatency').textContent = `${(metrics.latency_ms / 1000).toFixed(2)} s`;
    document.getElementById('metricTokens').textContent = (metrics.total_tokens || 0).toLocaleString();
    document.getElementById('metricTokensDetail').textContent = `In: ${metrics.input_tokens || 0} | Out: ${metrics.output_tokens || 0}`;
    document.getElementById('metricCostInr').textContent = `Rs. ${(metrics.estimated_cost_inr || 0).toFixed(4)}`;
    document.getElementById('metricCostUsd').textContent = `$${(metrics.estimated_cost_usd || 0).toFixed(6)} USD`;
    document.getElementById('metricToolCalls').textContent = (metrics.search_calls || 0) + (metrics.fetch_calls || 0) + (metrics.llm_calls || 0);
    document.getElementById('metricToolCallsDetail').textContent = `${metrics.search_calls || 0}s, ${metrics.fetch_calls || 0}f, ${metrics.llm_calls || 0}l`;

    const supportedCount = audits.filter(a => a.status === 'SUPPORTED').length;
    document.getElementById('metricAuditSummary').textContent = `${supportedCount}/${audits.length} Supported`;

    // 2. Badges on tabs
    document.getElementById('auditorBadge').textContent = audits.length;
    document.getElementById('evidenceBadge').textContent = evidences.length;
    document.getElementById('sourcesBadge').textContent = Object.keys(fetchedPages).length || searchResults.length;

    // 3. Tab 1: Synthesized Answer
    document.getElementById('answerModeBadge').textContent = (data.mode || 'NORMAL_ANALYST').replace('_', ' ');
    document.getElementById('answerQuestionTitle').textContent = data.question || 'Research Answer';
    
    const formattedAnswer = formatMarkdownWithCitations(answer.answer || 'No answer generated.');
    document.getElementById('answerBody').innerHTML = formattedAnswer;

    // Unanswered / limitations
    const unanswered = answer.unanswered_aspects || [];
    const unansweredBox = document.getElementById('unansweredBox');
    const unansweredList = document.getElementById('unansweredList');
    if (unanswered && unanswered.length > 0) {
      unansweredBox.style.display = 'block';
      unansweredList.innerHTML = unanswered.map(u => `<li>${escapeHtml(u)}</li>`).join('');
    } else {
      unansweredBox.style.display = 'none';
    }

    // 4. Tab 2: Auditor Verification
    renderAuditorTab(audits);

    // 5. Tab 3: Evidence & Claim Gate
    renderEvidenceTab(evidences, claims);

    // 6. Tab 4: Sources
    renderSourcesTab(searchResults, fetchedPages);

    // 7. Tab 5: Plan & Policies
    renderPlanTab(plan, data.learned_policies || []);

    // Scroll to results smoothly
    resultsDashboard.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // Format Markdown with clickable citation pills
  function formatMarkdownWithCitations(rawText) {
    if (!rawText) return '<p>No answer text returned.</p>';

    // Escape HTML first
    let text = escapeHtml(rawText);

    // Convert [Source: URL] or [URL] into beautiful clickable pills
    text = text.replace(/\[(?:Source:\s*)?(https?:\/\/[^\s\]]+)\]/g, (match, url) => {
      try {
        const domain = new URL(url).hostname.replace('www.', '');
        return `<a href="${url}" target="_blank" rel="noopener noreferrer" class="citation-pill" title="${url}"><i class="fa-solid fa-arrow-up-right-from-square"></i> ${domain}</a>`;
      } catch (e) {
        return `<a href="${url}" target="_blank" rel="noopener noreferrer" class="citation-pill"><i class="fa-solid fa-link"></i> Source</a>`;
      }
    });

    // Convert bullet lists
    const lines = text.split('\n');
    let inList = false;
    let html = '';

    for (let line of lines) {
      line = line.trim();
      if (!line) {
        if (inList) { html += '</ul>'; inList = false; }
        continue;
      }

      if (line.startsWith('- ') || line.startsWith('* ')) {
        if (!inList) { html += '<ul>'; inList = true; }
        html += `<li>${formatInlineMarkdown(line.substring(2))}</li>`;
      } else {
        if (inList) { html += '</ul>'; inList = false; }
        html += `<p>${formatInlineMarkdown(line)}</p>`;
      }
    }
    if (inList) html += '</ul>';

    return html;
  }

  function formatInlineMarkdown(str) {
    // Bold: **text**
    str = str.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Code: `code`
    str = str.replace(/`(.*?)`/g, '<code>$1</code>');
    return str;
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Render Auditor Tab
  function renderAuditorTab(audits) {
    const container = document.getElementById('auditsList');
    if (!audits || audits.length === 0) {
      container.innerHTML = '<p class="text-muted">No claims audited for this query.</p>';
      return;
    }

    const supportedCount = audits.filter(a => a.status === 'SUPPORTED').length;
    const allSupported = supportedCount === audits.length;

    const summaryIcon = document.getElementById('auditorSummaryIcon');
    const summaryHeadline = document.getElementById('auditorSummaryHeadline');
    const summarySub = document.getElementById('auditorSummarySub');

    if (allSupported) {
      summaryIcon.className = 'auditor-icon-wrap all-supported';
      summaryIcon.innerHTML = '<i class="fa-solid fa-shield-check"></i>';
      summaryHeadline.textContent = `100% Verified: All ${audits.length} Claims Directly Supported`;
      summarySub.textContent = 'The Auditor confirmed every stated metric and entity in primary fetched disclosures.';
    } else {
      summaryIcon.className = 'auditor-icon-wrap has-issues';
      summaryIcon.innerHTML = '<i class="fa-solid fa-shield-halved"></i>';
      summaryHeadline.textContent = `Auditor Review: ${supportedCount}/${audits.length} Supported`;
      summarySub.textContent = 'Claims with discrepancies, scope mismatches, or missing citations were flagged.';
    }

    container.innerHTML = audits.map((audit, idx) => {
      const status = audit.status || 'UNAUDITED';
      const badgeClass = `badge-${status}`;
      const icon = status === 'SUPPORTED' ? 'fa-check' :
                   status === 'CONTRADICTED' ? 'fa-xmark' :
                   status === 'MISSING_CITATION' ? 'fa-link-slash' : 'fa-triangle-exclamation';

      return `
        <div class="audit-card">
          <div class="audit-card-top">
            <div class="audit-claim-text">"${escapeHtml(audit.claim)}"</div>
            <div class="status-verdict-badge ${badgeClass}">
              <i class="fa-solid ${icon}"></i> ${status}
            </div>
          </div>
          <div class="audit-reason">
            <strong>Auditor Assessment:</strong> ${escapeHtml(audit.reason || 'Verified against cited disclosure.')}
          </div>
          ${audit.evidence ? `<div class="audit-evidence-passage"><strong>Cited Passage:</strong> "${escapeHtml(audit.evidence)}"</div>` : ''}
          ${audit.source_url ? `
            <div>
              <a href="${audit.source_url}" target="_blank" rel="noopener noreferrer" class="audit-citation-link">
                <i class="fa-solid fa-external-link"></i> ${audit.source_url}
              </a>
            </div>
          ` : ''}
        </div>
      `;
    }).join('');
  }

  // Render Evidence Tab
  function renderEvidenceTab(evidences, claims) {
    const grid = document.getElementById('evidenceGrid');
    if (!evidences || evidences.length === 0) {
      grid.innerHTML = '<p class="text-muted">No evidence passages extracted.</p>';
      return;
    }

    grid.innerHTML = evidences.map((ev, idx) => {
      return `
        <div class="evidence-card">
          <div class="evidence-meta-tags">
            ${ev.entity ? `<span class="meta-tag entity"><i class="fa-solid fa-tag"></i> ${escapeHtml(ev.entity)}</span>` : ''}
            ${ev.business_segment ? `<span class="meta-tag segment">${escapeHtml(ev.business_segment)}</span>` : ''}
            ${ev.metric ? `<span class="meta-tag metric">${escapeHtml(ev.metric)}</span>` : ''}
            ${ev.period || ev.reported_period ? `<span class="meta-tag period">${escapeHtml(ev.period || ev.reported_period)}</span>` : ''}
            ${ev.source_status ? `<span class="meta-tag status-verified">${escapeHtml(ev.source_status)}</span>` : ''}
          </div>
          <div class="evidence-passage-text">
            "${escapeHtml(ev.passage)}"
          </div>
          <a href="${ev.url}" target="_blank" rel="noopener noreferrer" class="audit-citation-link">
            <i class="fa-solid fa-file-invoice"></i> ${ev.publisher || ev.url}
          </a>
        </div>
      `;
    }).join('');
  }

  // Render Sources Tab
  function renderSourcesTab(searchResults, fetchedPages) {
    const container = document.getElementById('sourcesList');
    const urls = Object.keys(fetchedPages);

    if (urls.length === 0 && searchResults.length === 0) {
      container.innerHTML = '<p class="text-muted">No external sources fetched.</p>';
      return;
    }

    // Combine fetched pages and search results
    const items = urls.map(u => {
      const page = fetchedPages[u] || {};
      return {
        url: u,
        title: page.title || u,
        status_code: page.status_code || 200,
        success: page.success !== false,
        method: page.method_used || 'fetch',
        publisher: page.publisher || '',
        source_type: page.source_type || 'LIVE',
      };
    });

    container.innerHTML = items.map(s => {
      const isOk = s.success && s.status_code < 400;
      const statusClass = isOk ? 'status-code-200' : 'status-code-err';
      return `
        <div class="source-item">
          <div class="source-item-info">
            <div class="source-item-title">${escapeHtml(s.title)}</div>
            <a href="${s.url}" target="_blank" rel="noopener noreferrer" class="source-item-url">
              ${escapeHtml(s.url)}
            </a>
          </div>
          <div class="source-item-badges">
            <span class="source-status-badge ${statusClass}">
              ${isOk ? `<i class="fa-solid fa-check"></i> HTTP ${s.status_code}` : `<i class="fa-solid fa-xmark"></i> FAILED ${s.status_code}`}
            </span>
            <span class="meta-tag">${escapeHtml(s.source_type)}</span>
          </div>
        </div>
      `;
    }).join('');
  }

  // Render Plan & Policies Tab
  function renderPlanTab(plan, policies) {
    const entitiesBox = document.getElementById('planEntities');
    const subQBox = document.getElementById('planSubQuestions');
    const queriesBox = document.getElementById('planQueries');
    const verifBox = document.getElementById('planVerification');
    const policiesList = document.getElementById('learnedPoliciesList');

    // Entities
    const entities = plan.entities || [];
    entitiesBox.innerHTML = entities.length ? entities.map(e => `<span class="entity-pill">${escapeHtml(e)}</span>`).join('') : '<span class="text-muted">None</span>';

    // Sub-questions
    const subQ = plan.sub_questions || [];
    subQBox.innerHTML = subQ.length ? subQ.map(q => `<li>${escapeHtml(q)}</li>`).join('') : '<li>General inquiry</li>';

    // Queries
    const queries = plan.search_queries || [];
    queriesBox.innerHTML = queries.length ? queries.map(q => `<li>${escapeHtml(q)}</li>`).join('') : '<li>None</li>';

    // Verification
    const verif = plan.verification_requirements || [];
    verifBox.innerHTML = verif.length ? verif.map(v => `<li>${escapeHtml(v)}</li>`).join('') : '<li>Passage-level fidelity verification</li>';

    // Policies
    if (policies && policies.length > 0) {
      policiesList.innerHTML = policies.map(p => `<li><i class="fa-solid fa-shield-halved" style="color:var(--cyan); margin-right:8px;"></i>${escapeHtml(p)}</li>`).join('');
    } else {
      policiesList.innerHTML = '<li>No previous audit failures triggered feedback rules yet.</li>';
    }
  }

  // Load Past Runs into History Drawer
  async function loadHistoryRuns() {
    historyList.innerHTML = '<div class="loading-state"><i class="fa-solid fa-circle-notch fa-spin"></i> Loading recent runs...</div>';

    try {
      const res = await fetch('/api/history?limit=20');
      if (!res.ok) throw new Error('Failed to load history');
      const data = await res.json();
      const runs = data.history || [];

      if (runs.length === 0) {
        historyList.innerHTML = '<p class="text-muted" style="text-align:center; padding:20px;">No previous runs recorded.</p>';
        return;
      }

      historyList.innerHTML = runs.map(r => {
        const timeAgo = formatTimeAgo(r.created_at);
        return `
          <div class="history-item" data-question="${escapeHtml(r.question)}">
            <div class="history-q-text">${escapeHtml(r.question)}</div>
            <div class="history-meta">
              <span>${(r.latency_ms / 1000).toFixed(2)}s | ${r.total_tokens || 0} tokens</span>
              <span>${timeAgo}</span>
            </div>
          </div>
        `;
      }).join('');

      // Add click listeners to history items
      document.querySelectorAll('.history-item').forEach(item => {
        item.addEventListener('click', () => {
          const q = item.dataset.question;
          questionInput.value = q;
          closeHistoryDrawer();
          triggerResearch(q);
        });
      });
    } catch (e) {
      historyList.innerHTML = `<p class="text-muted">Error loading history: ${e.message}</p>`;
    }
  }

  function formatTimeAgo(dateStr) {
    if (!dateStr) return '';
    try {
      const d = new Date(dateStr);
      const diffMs = Date.now() - d.getTime();
      const diffMins = Math.floor(diffMs / 60000);
      if (diffMins < 1) return 'Just now';
      if (diffMins < 60) return `${diffMins}m ago`;
      const diffHours = Math.floor(diffMins / 60);
      if (diffHours < 24) return `${diffHours}h ago`;
      return `${Math.floor(diffHours / 24)}d ago`;
    } catch (e) {
      return dateStr;
    }
  }

});
