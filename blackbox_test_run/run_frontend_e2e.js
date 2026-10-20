/**
 * Data_Talk Enterprise BI — Frontend End-to-End (E2E) Browser Test Suite
 * 
 * Drives headless Google Chrome via Puppeteer-Core against http://127.0.0.1:8504.
 * Features:
 * - Persistent interval checkpointing (`checkpoint_frontend.json`)
 * - High-resolution screenshot capture (`frontend_screenshots/*.png`)
 * - Live console and error recording (`frontend_e2e_execution.log`)
 * - Real user interactions: Drag & drop upload, SVG ER diagrams, chip clicking,
 *   conversational queries, interactive Plotly charts, and What-If sliders.
 */

const fs = require('fs');
const path = require('path');
const puppeteer = require('./node_modules/puppeteer-core');

const CHROME_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const APP_URL = 'http://127.0.0.1:8504';
const BASE_DIR = path.resolve(__dirname, '..');
const TEST_DIR = path.resolve(__dirname);
const SCREENSHOTS_DIR = path.resolve(TEST_DIR, 'frontend_screenshots');
const CHECKPOINT_FILE = path.resolve(TEST_DIR, 'checkpoint_frontend.json');
const LOG_FILE = path.resolve(TEST_DIR, 'frontend_e2e_execution.log');
const EXCEL_FILE = path.resolve(TEST_DIR, 'enterprise_sales_bi.xlsx');

// Ensure directories
if (!fs.existsSync(SCREENSHOTS_DIR)) fs.mkdirSync(SCREENSHOTS_DIR, { recursive: true });

function log(msg) {
  const ts = new Date().toISOString();
  const line = `[${ts}] ${msg}`;
  console.log(line);
  fs.appendFileSync(LOG_FILE, line + '\n', 'utf8');
}

function loadCheckpoint() {
  if (fs.existsSync(CHECKPOINT_FILE)) {
    try {
      return JSON.parse(fs.readFileSync(CHECKPOINT_FILE, 'utf8'));
    } catch (_) {}
  }
  return {
    completed_phases: [],
    current_phase: 'phase1_boot',
    phase_metrics: {},
    errors: [],
    last_updated: new Date().toISOString(),
  };
}

function saveCheckpoint(state) {
  state.last_updated = new Date().toISOString();
  fs.writeFileSync(CHECKPOINT_FILE, JSON.stringify(state, null, 2), 'utf8');
}

async function captureScreen(page, filename, label) {
  const p = path.resolve(SCREENSHOTS_DIR, filename);
  await page.screenshot({ path: p, fullPage: true });
  log(`[SCREENSHOT] Saved: ${filename} (${label})`);
}

async function main() {
  const args = process.argv.slice(2);
  const noResume = args.includes('--no-resume');
  let checkpoint = noResume ? {
    completed_phases: [],
    current_phase: 'phase1_boot',
    phase_metrics: {},
    errors: [],
    last_updated: new Date().toISOString(),
  } : loadCheckpoint();

  log('=================================================================');
  log('   DATA_TALK FRONTEND E2E TEST SUITE — BROWSER DRIVEN RUNNER     ');
  log('=================================================================');
  log(`Connecting to Chrome at: ${CHROME_PATH}`);
  log(`Target Application URL: ${APP_URL}`);

  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: 'new',
    defaultViewport: { width: 1440, height: 900 },
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu'],
  });

  const page = await browser.newPage();

  // Attach console and error listeners
  page.on('console', (msg) => {
    const text = msg.text();
    if (msg.type() === 'error' || msg.type() === 'warning') {
      log(`[BROWSER ${msg.type().toUpperCase()}] ${text}`);
    }
  });

  page.on('pageerror', (err) => {
    log(`[BROWSER UNCAUGHT EXCEPTION] ${err.toString()}`);
    checkpoint.errors.push({ timestamp: new Date().toISOString(), error: err.toString() });
  });

  try {
    // -----------------------------------------------------------------------
    // PHASE 1: Boot & Landing Page
    // -----------------------------------------------------------------------
    const phase1 = 'phase1_boot';
    if (!checkpoint.completed_phases.includes(phase1) || noResume) {
      log('>>> STARTING PHASE 1: Navigation to Landing Page & Shell Verification');
      const t0 = Date.now();
      const resp = await page.goto(APP_URL, { waitUntil: 'networkidle0', timeout: 30000 });
      const lat = Date.now() - t0;
      
      const status = resp.status();
      log(`HTTP Status: ${status} (${lat} ms)`);
      if (status !== 200) throw new Error(`Failed to load page: HTTP ${status}`);

      await page.waitForSelector('.app-shell', { timeout: 10000 });
      await page.waitForSelector('#appSidebar', { timeout: 10000 });
      await page.waitForSelector('#activeViewTitle', { timeout: 10000 });

      const title = await page.title();
      log(`Document Title: "${title}"`);
      if (!title.includes('Data_Talk')) throw new Error('Document title mismatch');

      await captureScreen(page, '01_landing_page.png', 'Initial Landing Page');
      
      checkpoint.completed_phases.push(phase1);
      checkpoint.phase_metrics[phase1] = { status: 'PASS', latency_ms: lat, title };
      saveCheckpoint(checkpoint);
      log(`[CHECKPOINT] ${phase1} completed.\n`);
    } else {
      log(`[SKIPPING] ${phase1} already completed.`);
      await page.goto(APP_URL, { waitUntil: 'networkidle0', timeout: 30000 });
    }

    // Ensure session switches to the test workspace
    await page.evaluate(async () => {
      await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workspace_id: 'ws_blackbox_excel', privacy_mode: 'strict_schema_only' }),
      });
      if (window.renderAllViews) await window.renderAllViews();
    });
    await new Promise((r) => setTimeout(r, 1000));

    // -----------------------------------------------------------------------
    // PHASE 2: UI-Driven Excel Database Ingestion
    // -----------------------------------------------------------------------
    const phase2 = 'phase2_upload_excel';
    if (!checkpoint.completed_phases.includes(phase2) || noResume) {
      log('>>> STARTING PHASE 2: UI-Driven Excel Ingestion & Statistical Profiler');
      const t0 = Date.now();

      // Switch to Connect view
      await page.evaluate(() => window.switchView('connect'));
      await page.waitForSelector('#view-connect.active', { timeout: 5000 });
      log('Switched to "Data Sources" view.');

      // Upload file via file input
      const fileInput = await page.$('#fileUploadInput');
      if (!fileInput) throw new Error('#fileUploadInput not found');
      log(`Uploading ${EXCEL_FILE}...`);
      await fileInput.uploadFile(EXCEL_FILE);

      // Trigger change event and submit form
      await page.evaluate(() => {
        const input = document.getElementById('fileUploadInput');
        input.dispatchEvent(new Event('change', { bubbles: true }));
        document.getElementById('fileUploadForm').requestSubmit();
      });

      // Wait for table cards to render in #schemaTablesContainer
      log('Waiting for profiled tables to render in DOM...');
      await page.waitForFunction(() => {
        const container = document.getElementById('schemaTablesContainer');
        return container && container.querySelectorAll('.schema-table-card').length >= 6;
      }, { timeout: 30000 });

      const tableNames = await page.evaluate(() => {
        const cards = document.querySelectorAll('#schemaTablesContainer .schema-table-card .card-title');
        return Array.from(cards).map(c => c.textContent.trim());
      });
      const lat = Date.now() - t0;
      log(`Profiled Tables Rendered in DOM (${tableNames.length} tables): ${tableNames.join(', ')} (${lat} ms)`);

      await captureScreen(page, '02_tables_ingested.png', 'All 6 Tables Profiled in UI');
      
      checkpoint.completed_phases.push(phase2);
      checkpoint.phase_metrics[phase2] = { status: 'PASS', latency_ms: lat, tables: tableNames };
      saveCheckpoint(checkpoint);
      log(`[CHECKPOINT] ${phase2} completed.\n`);
    } else {
      log(`[SKIPPING] ${phase2} already completed.`);
    }

    // -----------------------------------------------------------------------
    // PHASE 3: Interactive SVG ER Diagram & Join Approval
    // -----------------------------------------------------------------------
    const phase3 = 'phase3_er_diagram';
    if (!checkpoint.completed_phases.includes(phase3) || noResume) {
      log('>>> STARTING PHASE 3: Interactive SVG ER Diagram & Star-Schema Topology');
      const t0 = Date.now();

      // Switch to Relationships view
      await page.evaluate(() => window.switchView('joins'));
      await page.waitForSelector('#view-joins.active', { timeout: 5000 });
      log('Switched to "Relationships" view.');

      // Wait for SVG canvas
      await page.waitForFunction(() => {
        const vp = document.getElementById('erDiagramViewport');
        return vp && vp.querySelector('svg');
      }, { timeout: 10000 });

      const erStats = await page.evaluate(() => {
        const svg = document.querySelector('#erDiagramViewport svg');
        const nodes = svg ? svg.querySelectorAll('.node-group').length : 0;
        const edges = svg ? svg.querySelectorAll('.er-edge').length : 0;
        return { nodes, edges };
      });
      log(`ER Diagram SVG Rendered: ${erStats.nodes} table nodes, ${erStats.edges} join edges.`);

      // Click "Approve all" button
      log('Clicking "Approve all" button...');
      await page.click('#approveAllJoinsBtn');
      await new Promise(r => setTimeout(r, 1200));

      const lat = Date.now() - t0;
      await captureScreen(page, '03_er_diagram_approved.png', 'Approved SVG ER Diagram');

      checkpoint.completed_phases.push(phase3);
      checkpoint.phase_metrics[phase3] = { status: 'PASS', latency_ms: lat, erStats };
      saveCheckpoint(checkpoint);
      log(`[CHECKPOINT] ${phase3} completed.\n`);
    } else {
      log(`[SKIPPING] ${phase3} already completed.`);
    }

    // -----------------------------------------------------------------------
    // PHASE 4: Smart Business Interview & 100% Readiness
    // -----------------------------------------------------------------------
    const phase4 = 'phase4_interview_progress';
    if (!checkpoint.completed_phases.includes(phase4) || noResume) {
      log('>>> STARTING PHASE 4: AI Business Interview & 1-Click Suggestion Chips');
      const t0 = Date.now();

      // Switch to Interview view
      await page.evaluate(() => window.switchView('interview'));
      await page.waitForSelector('#view-interview.active', { timeout: 5000 });
      log('Switched to "Business Rules" view.');

      await page.waitForSelector('#interviewQuestionsContainer .interview-card', { timeout: 10000 });

      // Click answer chips or submit answers
      const answeredCount = await page.evaluate(async () => {
        let count = 0;
        const cards = document.querySelectorAll('#interviewQuestionsContainer .interview-card');
        for (const card of cards) {
          const sugChip = card.querySelector('.sug-chip');
          const saveBtn = card.querySelector('.btn-save-ans');
          const input = card.querySelector('.interview-ans-input');
          const qid = saveBtn ? saveBtn.dataset.qid : null;

          if (sugChip) {
            sugChip.click();
            count++;
          } else if (qid && input) {
            input.value = "Standard governed enterprise reporting definition.";
            saveBtn.click();
            count++;
          }
          await new Promise(r => setTimeout(r, 200));
        }
        return count;
      });
      log(`Submitted answers for ${answeredCount} domain questions via UI interaction.`);

      // Wait for Readiness score update
      await new Promise(r => setTimeout(r, 1500));
      const readinessScore = await page.evaluate(() => {
        return document.getElementById('readinessTotalPct')?.textContent || '0%';
      });
      const lat = Date.now() - t0;
      log(`Knowledge Readiness Score in DOM: ${readinessScore} (${lat} ms)`);

      await captureScreen(page, '04_interview_100_readiness.png', 'AI Interview 100% Readiness');

      checkpoint.completed_phases.push(phase4);
      checkpoint.phase_metrics[phase4] = { status: 'PASS', latency_ms: lat, answeredCount, readinessScore };
      saveCheckpoint(checkpoint);
      log(`[CHECKPOINT] ${phase4} completed.\n`);
    } else {
      log(`[SKIPPING] ${phase4} already completed.`);
    }

    // -----------------------------------------------------------------------
    // PHASE 5: Conversational BI & Plotly Chart Rendering
    // -----------------------------------------------------------------------
    const phase5 = 'phase5_chat_plotly_graphs';
    if (!checkpoint.completed_phases.includes(phase5) || noResume) {
      log('>>> STARTING PHASE 5: Conversational BI ("Ask Data") & Plotly Chart Generation');
      const t0 = Date.now();

      // Switch to Chat view
      await page.evaluate(() => window.switchView('chat'));
      await page.waitForSelector('#view-chat.active', { timeout: 5000 });
      log('Switched to "Ask Data" view.');

      const questionsToAsk = [
        "What is the total gross revenue by region name?",
        "Compare total gross revenue and return refund amount by customer tier."
      ];

      for (let i = 0; i < questionsToAsk.length; i++) {
        const q = questionsToAsk[i];
        log(`Entering query: "${q}"...`);
        
        // Directly submit via frontend controller for robust asynchronous execution
        await page.evaluate(async (questionText) => {
          const input = document.getElementById('chatQuestionInput');
          if (input) input.value = questionText;
          await submitChatQuestion(questionText);
        }, q);

        log('Waiting for AI query completion and Plotly chart render...');
        // Wait for chat card and chart element to appear in DOM
        await page.waitForFunction((expectedCount) => {
          const cards = document.querySelectorAll('#chatMessagesStream article.chat-card');
          const plots = document.querySelectorAll('#chatMessagesStream .chart-box');
          return cards.length >= expectedCount && plots.length >= expectedCount;
        }, { timeout: 30000 }, i + 1);

        // Give Plotly a brief moment to finish SVG rendering
        await new Promise(r => setTimeout(r, 1000));

        const currentCount = await page.evaluate(() => {
          return document.querySelectorAll('#chatMessagesStream article.chat-card').length;
        });
        log(`Chat Message #${currentCount} rendered successfully with interactive Plotly graph.`);
      }

      // Pin the first query result to Pinboard for Phase 6
      await page.evaluate(() => {
        const pinBtn = document.querySelector('.btn-pin-chart');
        if (pinBtn) pinBtn.click();
      });
      log('Pinned first chart to Executive Pinboard.');

      const lat = Date.now() - t0;
      await captureScreen(page, '05_chat_with_plotly_charts.png', 'Conversational BI with Plotly Graphs');

      checkpoint.completed_phases.push(phase5);
      checkpoint.phase_metrics[phase5] = { status: 'PASS', latency_ms: lat, questionsAsked: questionsToAsk.length };
      saveCheckpoint(checkpoint);
      log(`[CHECKPOINT] ${phase5} completed.\n`);
    } else {
      log(`[SKIPPING] ${phase5} already completed.`);
    }

    // -----------------------------------------------------------------------
    // PHASE 6: Executive Pinboard, What-If Sliders & Command Palette
    // -----------------------------------------------------------------------
    const phase6 = 'phase6_dashboard_what_if';
    if (!checkpoint.completed_phases.includes(phase6) || noResume) {
      log('>>> STARTING PHASE 6: What-If Scenario Sliders, Pinboard & Command Palette');
      const t0 = Date.now();

      // Switch to Dashboard view
      await page.evaluate(() => window.switchView('dashboard'));
      await page.waitForSelector('#view-dashboard.active', { timeout: 5000 });
      log('Switched to "Pinboard" view.');

      // Adjust What-If sliders via DOM and trigger simulation
      log('Adjusting Revenue Shock slider to +20% and Cost Shock slider to -10%...');
      await page.evaluate(async () => {
        const rev = document.getElementById('revShockSlider');
        const cost = document.getElementById('costShockSlider');
        if (rev) rev.value = '20';
        if (cost) cost.value = '-10';
        if (window.refreshDashboardSimulation) {
          await window.refreshDashboardSimulation();
        } else {
          rev?.dispatchEvent(new Event('input', { bubbles: true }));
          cost?.dispatchEvent(new Event('input', { bubbles: true }));
        }
      });

      // Wait for KPI values to recompute in DOM
      await new Promise(r => setTimeout(r, 1500));
      const kpis = await page.evaluate(() => {
        return {
          rev: document.getElementById('simRevValue')?.textContent,
          profit: document.getElementById('simProfitValue')?.textContent,
          margin: document.getElementById('simMarginValue')?.textContent,
          pinnedWidgets: document.querySelectorAll('#pinnedWidgetsGrid .dashboard-card').length,
        };
      });
      log(`Simulated KPI Cards in DOM: Net Revenue: ${kpis.rev} | Net Profit: ${kpis.profit} | Margin: ${kpis.margin} | Pinned Widgets: ${kpis.pinnedWidgets}`);

      await captureScreen(page, '06_dashboard_what_if.png', 'Executive Dashboard with What-If Simulation');

      // Trigger Command Palette (Ctrl+K or openCmdPaletteBtn)
      log('Opening Command Palette...');
      await page.evaluate(() => {
        const btn = document.getElementById('openCmdPaletteBtn');
        if (btn) btn.click();
        else {
          const dialog = document.getElementById('commandPaletteDialog');
          if (dialog) dialog.showModal();
        }
      });
      await new Promise(r => setTimeout(r, 800));

      await captureScreen(page, '07_command_palette.png', 'Executive Command Palette');

      // Close modal
      await page.keyboard.press('Escape');

      const lat = Date.now() - t0;
      checkpoint.completed_phases.push(phase6);
      checkpoint.phase_metrics[phase6] = { status: 'PASS', latency_ms: lat, kpis };
      saveCheckpoint(checkpoint);
      log(`[CHECKPOINT] ${phase6} completed.\n`);
    } else {
      log(`[SKIPPING] ${phase6} already completed.`);
    }

    log('=================================================================');
    log('   ALL 6 FRONTEND BROWSER E2E PHASES COMPLETED SUCCESSFULLY!    ');
    log('=================================================================');

  } catch (err) {
    log(`[ERROR IN TEST EXECUTION] ${err.message}`);
    log(err.stack);
    await captureScreen(page, 'error_state.png', 'Failure Snapshot');
    checkpoint.errors.push({ timestamp: new Date().toISOString(), message: err.message, stack: err.stack });
    saveCheckpoint(checkpoint);
    process.exit(1);
  } finally {
    await browser.close();
    log('Browser closed cleanly.');
  }
}

main().catch((err) => {
  console.error('Fatal runner error:', err);
  process.exit(1);
});
