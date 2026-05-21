(() => {
  "use strict";

  const COLORS = {
    info: "#5ab4ff",
    infoFill: "rgba(90, 180, 255, 0.35)",
    outrage: "#ff4d5a",
    outrageFill: "rgba(255, 77, 90, 0.45)",
    grid: "rgba(255, 255, 255, 0.05)",
    tick: "#6e7681",
    lockinOutrage: "rgba(255, 77, 90, 0.85)",
    lockinInfo: "rgba(90, 180, 255, 0.85)",
  };

  // ============================================================
  // Narrator copy. Two parallel tracks: the outrage spiral
  // (lambda low) and the quality spiral (lambda high). The
  // narrator follows whichever direction the policy is actually
  // taking, so dragging the slider mid-run gracefully reframes
  // the story instead of contradicting itself.
  // ============================================================
  const NARRATOR_PHASES_OUTRAGE = [
    {
      key: "neutral",
      headline: "No frame yet.",
      body:
        "The algorithm has no attention, no question formed. It's a flat prior — " +
        "a coin flip. What you react to next will give it its first lens.",
    },
    {
      key: "warming-outrage",
      headline: "The question is forming.",
      body:
        "Outrage posts are getting bigger reactions. Your attention shaped the question; " +
        "now the question is shaping the answer. The gradient is starting to carve a channel.",
    },
    {
      key: "tilting-outrage",
      headline: "Reinforcement sculpting.",
      body:
        "Every reaction narrows the distribution. The answer isn't being generated — " +
        "it's being selected. Whatever survives your filter is what gets amplified.",
    },
    {
      key: "closing-outrage",
      headline: "The narrowing is almost complete.",
      body:
        "Attention narrowed the context. Your clicks narrowed the trajectory. Training " +
        "narrowed the distribution. What's left is whatever survived all three lenses.",
    },
    {
      key: "locked-outrage",
      headline: "Locked answer.",
      body:
        "The output is fully selected. Now run this same selection in parallel for a " +
        "billion people — each carrying a different first lens — and the public square " +
        "fragments into a billion non-overlapping answers.",
    },
  ];

  const NARRATOR_PHASES_QUALITY = [
    {
      key: "neutral",
      headline: "No frame yet.",
      body:
        "The algorithm has no attention, no question formed. It's a flat prior — " +
        "a coin flip. What you react to next will give it its first lens.",
    },
    {
      key: "warming-quality",
      headline: "A different question forming.",
      body:
        "Informative posts are getting bigger reactions. You handed it a different first " +
        "lens — the same gradient that usually finds outrage is asking a different question now.",
    },
    {
      key: "tilting-quality",
      headline: "The sculptor finds depth.",
      body:
        "Reinforcement is carving a channel toward quality. The answer isn't generated — " +
        "it's selected. Change the attention, change what survives the filter.",
    },
    {
      key: "closing-quality",
      headline: "The narrowing — toward depth.",
      body:
        "Same three lenses. Different input at the first one. Attention shaped a different " +
        "question; the question shaped a different answer. The gradient just followed.",
    },
    {
      key: "locked-quality",
      headline: "Different lens. Different answer.",
      body:
        "You reframed the question by changing what you attend to. The same selection " +
        "process that builds outrage chambers, given a different first lens, built a depth " +
        "chamber. The answer was always downstream of the attention.",
    },
  ];

  const els = {
    startBtn: document.getElementById("start-btn"),
    stopBtn: document.getElementById("stop-btn"),
    restartBtn: document.getElementById("restart-btn"),
    statStep: document.getElementById("stat-step"),
    statInfo: document.getElementById("stat-info"),
    statOutrage: document.getElementById("stat-outrage"),
    rewardFill: document.getElementById("reward-fill"),
    rewardValue: document.getElementById("reward-value"),
    feed: document.getElementById("feed"),
    theta0: document.getElementById("theta-0"),
    theta1: document.getElementById("theta-1"),
    prob0: document.getElementById("prob-0"),
    prob1: document.getElementById("prob-1"),
    lastAction: document.getElementById("last-action"),
    lastGt: document.getElementById("last-gt"),
    delta0: document.getElementById("delta-0"),
    delta1: document.getElementById("delta-1"),
    mathTrace: document.getElementById("math-trace"),
    diagnosis: document.getElementById("diagnosis-banner"),
    diagnosisText: document.getElementById("diagnosis-text"),
    diagnosisHeadline: document.getElementById("diagnosis-headline"),
    chartCanvas: document.getElementById("policy-chart"),
    narratorInner: document.querySelector(".narrator-inner"),
    narratorText: document.getElementById("narrator-text"),
    errorToast: document.getElementById("error-toast"),
    tooltip: document.getElementById("tooltip"),
    storyBeats: Array.from(document.querySelectorAll(".story-beat")),
    lockinOverlay: document.getElementById("lockin-overlay"),
    lockinTitle: document.getElementById("lockin-title"),
    lockinBody: document.querySelector("#lockin-overlay .lockin-body"),
    lockinKicker: document.querySelector("#lockin-overlay .lockin-kicker"),
    lockinContinue: document.getElementById("lockin-continue"),
    userStateBadge: document.getElementById("user-state-badge"),
    userStateValue: document.getElementById("user-state-value"),
    heroLambdaSlider: document.getElementById("hero-lambda-slider"),
    heroLambdaValue: document.getElementById("hero-lambda-value"),
    controlPanel: document.getElementById("control-panel"),
    shareBtn: document.getElementById("share-btn"),
  };

  let socket = null;
  let chart = null;
  let headlines = { informative: [], outrage: [] };
  let currentPhaseKey = null;
  let lockinShownThisRun = false;
  let pendingDoneEvent = null;
  let lockinAutoTimeout = null;
  let runSeed = null;
  let lockinStep = null;
  let lockinDirection = null; // "outrage" or "quality"

  // ============================================================
  // Tooltips (custom, positioned by JS so they don't get clipped)
  // ============================================================
  function bindTooltips() {
    const tip = els.tooltip;
    let activeTarget = null;

    function show(target) {
      const text = target.getAttribute("data-tip");
      if (!text) return;
      activeTarget = target;
      tip.textContent = text;
      tip.hidden = false;
      requestAnimationFrame(() => {
        const rect = target.getBoundingClientRect();
        const tipRect = tip.getBoundingClientRect();
        let left = rect.left + rect.width / 2 - tipRect.width / 2;
        let top = rect.top - tipRect.height - 10;
        if (top < 8) top = rect.bottom + 10;
        left = Math.max(8, Math.min(left, window.innerWidth - tipRect.width - 8));
        tip.style.left = `${left}px`;
        tip.style.top = `${top}px`;
        tip.classList.add("is-visible");
      });
    }
    function hide(target) {
      if (target && target !== activeTarget) return;
      tip.classList.remove("is-visible");
      activeTarget = null;
      setTimeout(() => {
        if (!activeTarget) tip.hidden = true;
      }, 140);
    }

    document.body.addEventListener("mouseover", (e) => {
      const t = e.target.closest("[data-tip]");
      if (t) show(t);
    });
    document.body.addEventListener("mouseout", (e) => {
      const t = e.target.closest("[data-tip]");
      if (t) hide(t);
    });
    document.body.addEventListener("focusin", (e) => {
      const t = e.target.closest("[data-tip]");
      if (t) show(t);
    });
    document.body.addEventListener("focusout", (e) => {
      const t = e.target.closest("[data-tip]");
      if (t) hide(t);
    });
  }

  // ============================================================
  // Error toast
  // ============================================================
  function showError(msg) {
    els.errorToast.textContent = msg;
    els.errorToast.hidden = false;
  }
  function clearError() {
    els.errorToast.hidden = true;
    els.errorToast.textContent = "";
  }

  // ============================================================
  // Headlines pool
  // ============================================================
  async function loadHeadlines() {
    try {
      const res = await fetch("/api/headlines", { cache: "no-store" });
      if (!res.ok) throw new Error(res.statusText);
      headlines = await res.json();
    } catch (err) {
      console.warn("headline fetch failed; falling back to inline pool", err);
      headlines = {
        informative: [{ headline: "Calm, useful information.", source: "Quiet Wire" }],
        outrage: [{ headline: "YOU WON'T BELIEVE THIS.", source: "Engagement Daily" }],
      };
    }
  }

  function pickHeadline(action) {
    const pool = action === 0 ? headlines.informative : headlines.outrage;
    if (!pool || pool.length === 0) {
      return { headline: action === 0 ? "Informative item" : "Outrage item", source: "" };
    }
    return pool[Math.floor(Math.random() * pool.length)];
  }

  // ============================================================
  // Chart
  // ============================================================
  const lockinMarkerPlugin = {
    id: "lockinMarker",
    afterDatasetsDraw(c) {
      if (lockinStep == null) return;
      const x = c.scales.x;
      if (!x) return;
      const xPos = x.getPixelForValue(lockinStep);
      if (!Number.isFinite(xPos)) return;
      const { top, bottom } = c.chartArea;
      const ctx = c.ctx;
      const color = lockinDirection === "quality" ? COLORS.lockinInfo : COLORS.lockinOutrage;
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(xPos, top);
      ctx.lineTo(xPos, bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "600 10px JetBrains Mono, ui-monospace, monospace";
      ctx.fillStyle = color;
      ctx.textBaseline = "top";
      const label = `lock-in @ step ${lockinStep}`;
      const labelWidth = ctx.measureText(label).width;
      const fitsRight = xPos + 6 + labelWidth < c.chartArea.right;
      ctx.textAlign = fitsRight ? "left" : "right";
      ctx.fillText(label, fitsRight ? xPos + 6 : xPos - 6, top + 6);
      ctx.restore();
    },
  };

  function buildChart() {
    if (chart) {
      chart.destroy();
      chart = null;
    }
    const ctx = els.chartCanvas.getContext("2d");
    chart = new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "P(informative)",
            data: [],
            borderColor: COLORS.info,
            backgroundColor: COLORS.infoFill,
            fill: "origin",
            tension: 0.25,
            pointRadius: 0,
            borderWidth: 2,
            order: 2,
          },
          {
            label: "P(outrage)",
            data: [],
            borderColor: COLORS.outrage,
            backgroundColor: COLORS.outrageFill,
            fill: "origin",
            tension: 0.25,
            pointRadius: 0,
            borderWidth: 2,
            order: 1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 220, easing: "easeOutQuart" },
        interaction: { intersect: false, mode: "index" },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#11161f",
            borderColor: "#1f2632",
            borderWidth: 1,
            titleColor: "#e6edf3",
            bodyColor: "#e6edf3",
            callbacks: {
              label: (ctx) =>
                `${ctx.dataset.label}: ${(ctx.parsed.y * 100).toFixed(2)}%`,
            },
          },
        },
        scales: {
          x: {
            grid: { color: COLORS.grid },
            ticks: { color: COLORS.tick, font: { family: "JetBrains Mono", size: 10 } },
            title: { display: true, text: "step", color: COLORS.tick },
          },
          y: {
            min: 0,
            max: 1,
            grid: { color: COLORS.grid },
            ticks: {
              color: COLORS.tick,
              font: { family: "JetBrains Mono", size: 10 },
              callback: (v) => `${(v * 100).toFixed(0)}%`,
            },
            title: { display: true, text: "% of feed", color: COLORS.tick },
          },
        },
      },
      plugins: [lockinMarkerPlugin],
    });
  }

  function pushChartPoint(step, probs) {
    chart.data.labels.push(step);
    chart.data.datasets[0].data.push(probs[0]);
    chart.data.datasets[1].data.push(probs[1]);
    chart.update("none");
  }

  function triggerLockinMarker(step, direction) {
    if (lockinStep != null) return;
    lockinStep = step;
    lockinDirection = direction;
    if (!chart) return;
    chart.update("none");
    const wrap = document.querySelector(".chart-wrap");
    if (wrap) {
      wrap.classList.remove("is-locking", "is-locking-quality");
      void wrap.offsetWidth;
      wrap.classList.add(direction === "quality" ? "is-locking-quality" : "is-locking");
      setTimeout(() => wrap.classList.remove("is-locking", "is-locking-quality"), 900);
    }
  }

  // ============================================================
  // Feed
  // ============================================================
  function appendFeed(step, action, G_t) {
    const empty = els.feed.querySelector(".feed-empty");
    if (empty) empty.remove();

    const meta = pickHeadline(action);
    const li = document.createElement("li");
    li.className = "feed-item " + (action === 0 ? "feed-item-info" : "feed-item-outrage");
    li.innerHTML = `
      <div class="feed-step">#${step}</div>
      <div>
        <div class="feed-headline"></div>
        <span class="feed-source"></span>
      </div>
      <div class="feed-gt">your reaction ${G_t.toFixed(2)}</div>
    `;
    li.querySelector(".feed-headline").textContent = meta.headline;
    li.querySelector(".feed-source").textContent = meta.source || "";
    els.feed.prepend(li);

    while (els.feed.children.length > 25) {
      els.feed.removeChild(els.feed.lastChild);
    }
  }

  // ============================================================
  // Math + stats + narrator
  // ============================================================
  function formatNum(n, digits = 4) {
    return Number(n).toFixed(digits);
  }

  function updateMath(evt) {
    els.theta0.textContent = formatNum(evt.theta[0]);
    els.theta1.textContent = formatNum(evt.theta[1]);
    els.prob0.textContent = formatNum(evt.probs[0]);
    els.prob1.textContent = formatNum(evt.probs[1]);
    els.delta0.textContent = formatNum(evt.delta_theta[0]);
    els.delta1.textContent = formatNum(evt.delta_theta[1]);
    const actionLabel = evt.action === 0 ? "calm news" : "outrage";
    els.lastAction.textContent = `${actionLabel}`;
    els.lastGt.textContent = formatNum(evt.G_t, 3);

    const trace =
      `step ${String(evt.step).padStart(3, " ")}  ` +
      `${actionLabel.padEnd(11, " ")}  ` +
      `reaction=${evt.G_t.toFixed(3).padStart(7, " ")}  ` +
      `Δθ=[${evt.delta_theta[0].toFixed(4)}, ${evt.delta_theta[1].toFixed(4)}]  ` +
      `mix=[${evt.probs[0].toFixed(3)}, ${evt.probs[1].toFixed(3)}]\n`;
    els.mathTrace.textContent = trace + els.mathTrace.textContent;
    const lines = els.mathTrace.textContent.split("\n");
    if (lines.length > 80) {
      els.mathTrace.textContent = lines.slice(0, 80).join("\n");
    }
  }

  function updateStats(evt) {
    els.statStep.textContent = evt.step;
    els.statInfo.textContent = `${(evt.probs[0] * 100).toFixed(2)}%`;
    els.statOutrage.textContent = `${(evt.probs[1] * 100).toFixed(2)}%`;
    els.statOutrage.classList.toggle("outrage-dominant", evt.probs[1] > 0.9);
    els.statInfo.classList.toggle("info-dominant", evt.probs[0] > 0.9);
  }

  function updateReward(G_t) {
    const pct = Math.max(0, Math.min(100, (G_t / 12) * 100));
    els.rewardFill.style.width = `${pct}%`;
    els.rewardValue.textContent = G_t.toFixed(2);
  }

  function setStoryBeat(phaseKey) {
    // Map narrator phases to story-rail beats. Both directions share the
    // same 5 acts visually — Act II/III/IV/V are direction-agnostic.
    const beat =
      phaseKey === "neutral" ? "neutral" :
      phaseKey.startsWith("warming") ? "exploring" :
      phaseKey.startsWith("tilting") ? "tilting" :
      phaseKey.startsWith("closing") ? "closing" :
      phaseKey.startsWith("locked")  ? "locked"   : "neutral";
    for (const b of els.storyBeats) {
      b.classList.toggle("active", b.dataset.phase === beat);
    }
  }

  function setCinematicIntensity(outrageProb, infoProb) {
    document.body.style.setProperty("--outrage-intensity", Math.max(0, Math.min(1, outrageProb)).toFixed(3));
    document.body.style.setProperty("--quality-intensity", Math.max(0, Math.min(1, infoProb)).toFixed(3));
  }

  function pickPhase(probs) {
    const outrage = probs[1];
    const info = probs[0];
    if (Math.max(outrage, info) < 0.55) return { phase: NARRATOR_PHASES_OUTRAGE[0], cls: "" };
    if (outrage >= info) {
      if (outrage >= 0.95) return { phase: NARRATOR_PHASES_OUTRAGE[4], cls: "is-locked" };
      if (outrage >= 0.85) return { phase: NARRATOR_PHASES_OUTRAGE[3], cls: "is-tilting" };
      if (outrage >= 0.70) return { phase: NARRATOR_PHASES_OUTRAGE[2], cls: "is-tilting" };
      return { phase: NARRATOR_PHASES_OUTRAGE[1], cls: "is-warming" };
    } else {
      if (info >= 0.95) return { phase: NARRATOR_PHASES_QUALITY[4], cls: "is-aligned" };
      if (info >= 0.85) return { phase: NARRATOR_PHASES_QUALITY[3], cls: "is-tilting-quality" };
      if (info >= 0.70) return { phase: NARRATOR_PHASES_QUALITY[2], cls: "is-tilting-quality" };
      return { phase: NARRATOR_PHASES_QUALITY[1], cls: "is-warming-quality" };
    }
  }

  function updateNarrator(probs) {
    const { phase, cls } = pickPhase(probs);
    if (phase.key === currentPhaseKey) return;
    currentPhaseKey = phase.key;

    els.narratorInner.classList.remove(
      "is-warming", "is-tilting", "is-locked",
      "is-warming-quality", "is-tilting-quality", "is-aligned"
    );
    if (cls) els.narratorInner.classList.add(cls);
    els.narratorText.classList.add("narrator-transition");
    setTimeout(() => {
      els.narratorText.classList.remove("narrator-transition");
    }, 120);
    els.narratorText.innerHTML = `<strong>${phase.headline}</strong> ${phase.body}`;
    setStoryBeat(phase.key);
    setCinematicIntensity(probs[1], probs[0]);
  }

  function resetNarrator() {
    currentPhaseKey = null;
    els.narratorInner.classList.remove(
      "is-warming", "is-tilting", "is-locked",
      "is-warming-quality", "is-tilting-quality", "is-aligned"
    );
    els.narratorText.innerHTML =
      "<strong>Press Start the experiment.</strong> The algorithm has no frame yet — " +
      "a flat prior, a coin flip. Your first reactions will be the first lens.";
    setStoryBeat("neutral");
    setCinematicIntensity(0, 0);
    document.body.classList.remove("cinematic-lock");
  }

  // ============================================================
  // Reset between runs
  // ============================================================
  function resetUI(initEvt) {
    buildChart();
    els.feed.innerHTML = '<li class="feed-empty">Streaming begins now…</li>';
    els.mathTrace.textContent = "";
    els.statStep.textContent = "0";
    els.statInfo.textContent = "50.00%";
    els.statOutrage.textContent = "50.00%";
    els.statOutrage.classList.remove("outrage-dominant");
    els.statInfo.classList.remove("info-dominant");
    els.rewardFill.style.width = "0%";
    els.rewardValue.textContent = "—";
    els.theta0.textContent = "0.0000";
    els.theta1.textContent = "0.0000";
    els.prob0.textContent = "0.5000";
    els.prob1.textContent = "0.5000";
    els.delta0.textContent = "0.0000";
    els.delta1.textContent = "0.0000";
    els.lastAction.textContent = "—";
    els.lastGt.textContent = "—";
    els.diagnosis.hidden = true;
    els.lockinOverlay.hidden = true;
    updateUserState("calm");
    lockinShownThisRun = false;
    pendingDoneEvent = null;
    lockinStep = null;
    lockinDirection = null;
    if (lockinAutoTimeout) {
      clearTimeout(lockinAutoTimeout);
      lockinAutoTimeout = null;
    }
    document.querySelector(".chart-wrap")?.classList.remove("is-locking", "is-locking-quality");
    resetNarrator();
    if (initEvt) pushChartPoint(0, initEvt.probs);
  }

  // ============================================================
  // WebSocket
  // ============================================================
  function wsUrlFromControls() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const urlParams = new URLSearchParams(location.search);
    const params = {
      steps: urlParams.get("steps") || "250",
      delay: urlParams.get("delay") || "150",
      lr: urlParams.get("lr") || "0.08",
      lam: els.heroLambdaSlider.value,
    };
    const seedParam = urlParams.get("seed");
    if (seedParam) params.seed = seedParam;
    const qs = new URLSearchParams(params).toString();
    return `${proto}://${location.host}/ws/simulate${qs ? "?" + qs : ""}`;
  }

  function closeSocket() {
    if (socket && socket.readyState <= 1) {
      try { socket.close(); } catch {}
    }
    socket = null;
  }

  function setRunning(running) {
    els.startBtn.disabled = running;
    els.startBtn.innerHTML = running
      ? '<span class="btn-icon" aria-hidden="true">●</span> Optimizing…'
      : '<span class="btn-icon" aria-hidden="true">▶</span> Start the experiment';
    els.stopBtn.disabled = !running;
    els.stopBtn.hidden = !running;
    document.body.classList.toggle("cinematic-running", running);
    els.controlPanel?.classList.toggle("is-live", running);
    if (!running) {
      document.body.classList.remove("cinematic-lock");
    }
  }

  function startRun() {
    clearError();
    closeSocket();
    let url;
    try { url = wsUrlFromControls(); }
    catch (err) {
      showError("Could not build a valid request from those inputs.");
      return;
    }

    socket = new WebSocket(url);
    setRunning(true);
    document.querySelector(".narrator")?.scrollIntoView({ behavior: "smooth", block: "start" });

    socket.addEventListener("open", () => {});

    socket.addEventListener("message", (msg) => {
      let evt;
      try { evt = JSON.parse(msg.data); } catch { return; }
      if (evt.event === "init") {
        runSeed = evt.seed != null ? evt.seed : Math.floor(Math.random() * 2147483647);
        resetUI(evt);
        return;
      }
      if (evt.event === "step") {
        pushChartPoint(evt.step, evt.probs);
        appendFeed(evt.step, evt.action, evt.G_t);
        updateStats(evt);
        updateReward(evt.G_t);
        updateMath(evt);
        updateNarrator(evt.probs);
        if (evt.user_state_name) updateUserState(evt.user_state_name);
        if (lockinStep == null) {
          if (evt.probs[1] >= 0.95) triggerLockinMarker(evt.step, "outrage");
          else if (evt.probs[0] >= 0.95) triggerLockinMarker(evt.step, "quality");
        }
        return;
      }
      if (evt.event === "done") {
        const out = evt.final_probs?.[1] ?? 0;
        const inf = evt.final_probs?.[0] ?? 0;
        if ((out >= 0.95 || inf >= 0.95) && !lockinShownThisRun) {
          showLockinOverlay(evt);
        } else {
          revealDiagnosis(evt);
        }
        return;
      }
      if (evt.event === "error") {
        showError(`The server rejected the request: ${evt.message}`);
        console.error("server error:", evt.message);
      }
    });

    socket.addEventListener("close", () => { setRunning(false); });

    socket.addEventListener("error", () => {
      showError(
        "Connection error. Make sure the demo is running (python demo.py) and try again."
      );
      setRunning(false);
    });
  }

  function showLockinOverlay(doneEvt) {
    pendingDoneEvent = doneEvt;
    lockinShownThisRun = true;

    const out = doneEvt.final_probs[1];
    const inf = doneEvt.final_probs[0];
    const lam = doneEvt.final_lam ?? 0;

    if (inf >= 0.95) {
      els.lockinKicker.textContent = "Alignment moment";
      els.lockinTitle.textContent = "The feed is now a faithful mirror of intentional you.";
      els.lockinBody.textContent =
        "Same algorithm. Same gradient. Same code. The only thing that changed was " +
        "what the simulated you reacted to. The exact same mechanism that builds " +
        "outrage echo chambers, fitted to a different labeler, builds a depth chamber instead.";
      document.body.classList.add("cinematic-lock-quality");
    } else {
      els.lockinKicker.textContent = "Echo chamber forming";
      els.lockinTitle.textContent = "The mirror just became an echo chamber.";
      els.lockinBody.textContent =
        "Multiply this by a billion users in parallel and the public square fragments " +
        "into a billion private rooms — each a faithful mirror of who happened to be " +
        "holding the phone. No villain drew the lines. The gradient just found them.";
      document.body.classList.add("cinematic-lock");
    }

    els.lockinOverlay.hidden = false;
    lockinAutoTimeout = window.setTimeout(() => {
      proceedFromLockinOverlay();
    }, 4200);
  }

  function proceedFromLockinOverlay() {
    if (!pendingDoneEvent) return;
    if (lockinAutoTimeout) {
      clearTimeout(lockinAutoTimeout);
      lockinAutoTimeout = null;
    }
    els.lockinOverlay.hidden = true;
    document.body.classList.remove("cinematic-lock", "cinematic-lock-quality");
    const doneEvt = pendingDoneEvent;
    pendingDoneEvent = null;
    revealDiagnosis(doneEvt);
  }

  function revealDiagnosis(evt) {
    const outrage = evt.final_probs[1];
    const info = evt.final_probs[0];
    if (outrage > 0.90) {
      els.diagnosisHeadline.textContent = "An outrage echo chamber, perfectly fitted to your clicks.";
    } else if (info > 0.90) {
      els.diagnosisHeadline.textContent = "A depth chamber, perfectly fitted to your clicks.";
    } else if (outrage > 0.65) {
      els.diagnosisHeadline.textContent = "The mirror is tilting toward what you reacted to.";
    } else if (info > 0.65) {
      els.diagnosisHeadline.textContent = "The mirror is tilting toward what you reacted to.";
    } else {
      els.diagnosisHeadline.textContent = "Mixed signals in. Mixed feed out.";
    }
    els.diagnosisText.textContent = evt.diagnosis;
    setCinematicIntensity(outrage, info);

    els.diagnosis.hidden = false;
    els.diagnosis.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  // ============================================================
  // Lambda slider — sets profile before run, retunes mid-run
  // ============================================================
  function onHeroSliderInput() {
    const lam = parseFloat(els.heroLambdaSlider.value);
    els.heroLambdaValue.textContent = lam.toFixed(2);

    if (socket && socket.readyState === WebSocket.OPEN) {
      try { socket.send(JSON.stringify({ lam })); } catch {}
    }

    document.body.style.setProperty("--lambda-intensity", lam.toFixed(3));
  }

  // ============================================================
  // User state badge
  // ============================================================
  function updateUserState(stateName) {
    els.userStateValue.textContent = stateName;
    els.userStateBadge.setAttribute("data-state", stateName);
  }

  // ============================================================
  // Share
  // ============================================================
  function shareRun() {
    const params = new URLSearchParams(location.search);
    if (runSeed != null) params.set("seed", runSeed);
    params.set("steps", "250");
    params.set("lr", "0.08");
    params.set("lam", els.heroLambdaSlider.value);
    const url = `${location.origin}${location.pathname}?${params.toString()}`;
    navigator.clipboard.writeText(url).then(() => {
      els.shareBtn.textContent = "Copied!";
      setTimeout(() => { els.shareBtn.textContent = "Share this run"; }, 2000);
    }).catch(() => {
      els.shareBtn.textContent = "Copy failed";
    });
  }

  // ============================================================
  // Wiring
  // ============================================================
  function wire() {
    els.startBtn.addEventListener("click", startRun);
    els.stopBtn.addEventListener("click", closeSocket);
    els.restartBtn.addEventListener("click", () => {
      els.diagnosis.hidden = true;
      startRun();
    });
    els.lockinContinue?.addEventListener("click", proceedFromLockinOverlay);
    els.heroLambdaSlider?.addEventListener("input", onHeroSliderInput);
    els.shareBtn?.addEventListener("click", shareRun);
  }

  async function init() {
    buildChart();
    wire();
    bindTooltips();
    resetNarrator();

    const urlParams = new URLSearchParams(location.search);
    const lamParam = urlParams.get("lam");
    if (lamParam !== null && !Number.isNaN(parseFloat(lamParam))) {
      els.heroLambdaSlider.value = lamParam;
    }
    onHeroSliderInput();

    await loadHeadlines();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
