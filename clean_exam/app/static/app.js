/* 깔끔 시험지 — 메인 화면 동작. 바닐라 JS, 외부 CDN 없음. */
(function () {
  const $ = (id) => document.getElementById(id);
  const S = window.settings;
  const state = { jobs: [], current: null, sliders: {}, config: null, previewTimer: null, previewToken: 0 };
  const SLIDER_KEYS = ["remove_strength", "color_sensitivity", "graph_thickness", "text_darkness"];

  /* ── 공통 ── */
  function showMessages(list) {
    const box = $("messages"); box.innerHTML = "";
    (list || []).forEach(m => { const d = document.createElement("div"); d.className = "msg " + (m.kind || ""); d.textContent = m.text; box.appendChild(d); });
  }
  function pushMessage(text, kind) { const d = document.createElement("div"); d.className = "msg " + (kind || ""); d.textContent = text; $("messages").appendChild(d); }
  async function api(path, opts = {}) {
    const r = await fetch(path, { ...opts, headers: { ...S.headers(), ...(opts.headers || {}) } });
    let j = null; try { j = await r.json(); } catch (e) {}
    if (!r.ok) throw new Error((j && (j.detail || j.error)) || `서버 오류 (${r.status})`);
    return j;
  }
  function form(obj) { const f = new FormData(); Object.entries(obj).forEach(([k, v]) => f.append(k, v)); return f; }
  const bust = (url) => url + (url.includes("?") ? "&" : "?") + "t=" + Date.now();

  /* ── 사용량 배지 ── */
  async function refreshUsage() {
    if (!S.getTeacher()) return;
    try { const u = await api("/api/usage"); $("usageBadge").textContent = `오늘 API 사용: ${u.calls}회 / 약 ₩${(u.cost_krw || 0).toLocaleString()}`; } catch (e) {}
  }

  /* ── 키 유무에 따라 잠금 ── */
  function updateLocks() {
    const has = S.hasKey();
    $("judgeLock").classList.toggle("hidden", has);
    $("useJudge").disabled = !has; if (!has) $("useJudge").checked = false;
    $("reviewBtn").title = has ? "" : "키를 넣으면 쓸 수 있어요";
    $("reviewBtn").disabled = !(has && state.current);
  }
  S.onChange = () => { updateLocks(); refreshUsage(); };

  /* ── 슬라이더 ── */
  function loadSliderDefaults() {
    let saved = {}; try { saved = JSON.parse(localStorage.getItem("cleanexam.sliders") || "{}"); } catch (e) {}
    state.sliders = { ...(state.config?.slider_defaults || {}), ...saved };
    SLIDER_KEYS.forEach(k => { const el = document.querySelector(`input[data-slider=${k}]`); el.value = state.sliders[k]; $("v_" + k).textContent = state.sliders[k]; });
  }
  document.querySelectorAll("input[data-slider]").forEach(el => el.addEventListener("input", () => {
    const k = el.dataset.slider; state.sliders[k] = +el.value; $("v_" + k).textContent = el.value;
    if (!state.current) return;
    clearTimeout(state.previewTimer);
    state.previewTimer = setTimeout(requestPreview, 350);
  }));
  async function requestPreview() {
    const job = state.current; if (!job) return;
    const token = ++state.previewToken; $("previewNote").textContent = "미리보기 계산 중…";
    try {
      const j = await api("/api/preview", { method: "POST", body: form({ job_id: job.job_id, sliders: JSON.stringify(state.sliders), use_judge: $("useJudge").checked ? 1 : 0 }) });
      if (token !== state.previewToken) return;
      $("resultImg").src = "data:image/png;base64," + j.preview_png_base64;
      $("previewNote").textContent = `미리보기 ${j.elapsed_seconds}초 (축소본). "원래 크기로 적용"을 누르면 결과 파일이 갱신돼요.`;
    } catch (e) { $("previewNote").textContent = "미리보기 실패: " + e.message; }
  }
  $("applyBtn").onclick = async () => {
    const job = state.current; if (!job) return;
    $("applyBtn").disabled = true; pushMessage("원래 크기로 다시 만드는 중…");
    try {
      const j = await api("/api/render", { method: "POST", body: form({ job_id: job.job_id, sliders: JSON.stringify(state.sliders), use_judge: $("useJudge").checked ? 1 : 0 }) });
      Object.assign(job, j); showJob(job); pushMessage("적용했어요.", "ok");
    } catch (e) { pushMessage(e.message, "err"); } finally { $("applyBtn").disabled = false; }
  };
  $("saveDefaults").onclick = () => { localStorage.setItem("cleanexam.sliders", JSON.stringify(state.sliders)); pushMessage("이 브라우저의 기본값으로 저장했어요.", "ok"); };

  /* ── 비교 슬라이더 ── */
  $("compareRange").addEventListener("input", (e) => { const v = e.target.value; $("resultImg").style.clipPath = `inset(0 0 0 ${v}%)`; $("divider").style.left = v + "%"; });

  /* ── 업로드 ── */
  const dz = $("dropzone");
  dz.onclick = () => $("fileInput").click();
  ["dragenter", "dragover"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("over"); }));
  dz.addEventListener("drop", e => handleFiles(e.dataTransfer.files));
  $("fileInput").onchange = e => { handleFiles(e.target.files); e.target.value = ""; };
  $("cameraBtn").onclick = () => $("cameraModal").classList.remove("hidden");
  $("cameraCancel").onclick = () => $("cameraModal").classList.add("hidden");
  $("cameraGo").onclick = () => { $("cameraModal").classList.add("hidden"); $("cameraInput").click(); };
  $("cameraInput").onchange = e => { handleFiles(e.target.files); e.target.value = ""; };

  async function handleFiles(files) {
    if (!S.getTeacher()) { alert("화면 위에서 강사 이름을 먼저 골라 주세요."); return; }
    for (const file of Array.from(files || [])) await uploadOne(file, null);
  }
  async function uploadOne(file, manualCorners, replaceJob) {
    const job = replaceJob || { job_id: null, file, name: file.name, status: "처리 중…", local: true };
    if (!replaceJob) { state.jobs.unshift(job); renderThumbs(); }
    job.status = "처리 중…"; renderThumbs();
    try {
      const body = form({ file, exam_name: $("examName").value, sliders: JSON.stringify(state.sliders), use_judge: $("useJudge").checked ? 1 : 0 });
      if (manualCorners) body.append("manual_corners", JSON.stringify(manualCorners));
      const j = await api("/api/clean", { method: "POST", body });
      Object.assign(job, j, { status: `${j.elapsed_seconds}초` + (j.api_call_count ? ` · 판정관 ${j.api_call_count}회` : "") });
      job.manual_corners = manualCorners || null;
      renderThumbs(); selectJob(job); refreshUsage(); loadExams();
    } catch (e) { job.status = "실패: " + e.message; job.failed = true; renderThumbs(); pushMessage(e.message, "err"); }
  }
  function renderThumbs() {
    const box = $("thumbs"); box.innerHTML = "";
    state.jobs.forEach(job => {
      const d = document.createElement("div"); d.className = "thumb" + (job === state.current ? " active" : "");
      const img = document.createElement("img"); if (job.original_url) img.src = job.original_url; else if (job.file) img.src = URL.createObjectURL(job.file);
      const t = document.createElement("div"); t.innerHTML = `<div class="name"></div><div class="status"></div>`;
      t.querySelector(".name").textContent = job.name || job.original_filename || job.job_id; t.querySelector(".status").textContent = job.status || "";
      d.append(img, t);
      if (job.quality_warnings && job.quality_warnings.length) { const w = document.createElement("span"); w.className = "warn"; w.textContent = "⚠ 다시 찍기 권장"; w.title = job.quality_warnings.join(" / "); d.appendChild(w); }
      d.onclick = () => job.result_url && selectJob(job);
      box.appendChild(d);
    });
  }

  /* ── 작업 보기 ── */
  function selectJob(job) { state.current = job; renderThumbs(); showJob(job); }
  function showJob(job) {
    $("placeholder").classList.add("hidden"); $("compare").classList.remove("hidden");
    $("originalImg").src = bust(job.original_url); $("resultImg").src = bust(job.result_url);
    $("issues").innerHTML = ""; closePopover();
    const msgs = (job.messages || []).map(t => ({ text: t, kind: /키|크레딧|올바르지/.test(t) ? "warn" : "" }));
    (job.quality_warnings || []).forEach(t => msgs.push({ text: t, kind: "warn" }));
    if (job.rectify_method === "manual") msgs.push({ text: "직접 잡은 모서리로 보정했어요.", kind: "ok" });
    showMessages(msgs);
    ["downloadPng", "downloadPdf", "applyBtn"].forEach(id => $(id).disabled = false);
    $("cornersBtn").disabled = !job.file;
    updateLocks();
  }

  /* ── 내려받기 ── */
  $("downloadPng").onclick = () => { const a = document.createElement("a"); a.href = state.current.result_url; a.download = (state.current.name || "결과").replace(/\.[^.]+$/, "") + "_정리.png"; a.click(); };
  $("downloadPdf").onclick = async () => {
    const ids = state.jobs.filter(j => j.result_url).map(j => j.job_id);
    if (!ids.length) return;
    const r = await fetch("/api/pdf", { method: "POST", headers: S.headers(), body: form({ job_ids: JSON.stringify(ids) }) });
    if (!r.ok) { pushMessage("PDF 만들기에 실패했어요.", "err"); return; }
    const blob = await r.blob(); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "정리본.pdf"; a.click();
  };

  /* ── 클로드 검수 ── */
  $("reviewBtn").onclick = async () => {
    const job = state.current; if (!job) return;
    $("reviewBtn").disabled = true; pushMessage("클로드가 결과를 살펴보는 중…");
    try {
      const j = await api("/api/review", { method: "POST", body: form({ job_id: job.job_id }) });
      refreshUsage();
      if (j.api_skipped) { pushMessage(j.message || "검수를 건너뛰었어요.", "warn"); return; }
      drawIssues(j.issues);
      pushMessage(j.issues.length ? `문제 ${j.issues.length}곳을 찾았어요. 박스를 눌러 고치세요.` : "문제를 찾지 못했어요. 깨끗해요!", j.issues.length ? "warn" : "ok");
    } catch (e) { pushMessage(e.message, "err"); } finally { $("reviewBtn").disabled = false; }
  };
  function drawIssues(issues) {
    const box = $("issues"); box.innerHTML = "";
    issues.forEach(issue => {
      const [x, y, w, h] = issue.box;
      const d = document.createElement("div"); d.className = "issue " + issue.type;
      d.style.cssText = `left:${x * 100}%;top:${y * 100}%;width:${w * 100}%;height:${h * 100}%`;
      const tip = document.createElement("span"); tip.className = "tip"; tip.textContent = `${issue.type_ko}: ${issue.note}`; d.appendChild(tip);
      d.onclick = (e) => { e.stopPropagation(); openPopover(d, issue); };
      box.appendChild(d);
    });
  }
  let popover = null;
  function closePopover() { if (popover) { popover.remove(); popover = null; } }
  function openPopover(anchor, issue) {
    closePopover();
    popover = document.createElement("div"); popover.className = "popover";
    popover.style.left = anchor.style.left; popover.style.top = `calc(${anchor.style.top} + ${anchor.style.height})`;
    const note = document.createElement("div"); note.className = "note"; note.textContent = `${issue.type_ko} — ${issue.note}`;
    const erase = document.createElement("button"); erase.className = "danger"; erase.textContent = "이건 필기니까 지워";
    const keep = document.createElement("button"); keep.textContent = "이건 문제니까 살려";
    const cancel = document.createElement("button"); cancel.textContent = "취소"; cancel.onclick = closePopover;
    erase.onclick = () => applyFix(issue, "erase", anchor); keep.onclick = () => applyFix(issue, "keep", anchor);
    popover.append(note, erase, keep, cancel); $("compare").appendChild(popover);
  }
  document.addEventListener("click", closePopover);
  async function applyFix(issue, decision, anchor) {
    closePopover(); const job = state.current; pushMessage("해당 영역을 다시 처리하는 중…");
    try {
      const j = await api("/api/fix", { method: "POST", body: form({ job_id: job.job_id, box: JSON.stringify(issue.box), decision }) });
      Object.assign(job, j); $("resultImg").src = bust(job.result_url); anchor.remove();
      pushMessage(decision === "erase" ? "지웠어요. 이 결정은 학습 데이터로 기록돼요." : "살렸어요. 이 결정은 학습 데이터로 기록돼요.", "ok");
    } catch (e) { pushMessage(e.message, "err"); }
  }

  /* ── 모서리 수동 조정 ── */
  const cornerState = { corners: null, drag: -1, img: null };
  $("cornersBtn").onclick = () => {
    const job = state.current; if (!job || !job.file) return;
    const img = new Image(); img.onload = () => { cornerState.img = img; cornerState.corners = job.manual_corners || [[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]]; $("cornerModal").classList.remove("hidden"); drawCorners(); };
    img.src = URL.createObjectURL(job.file);
  };
  $("cornerCancel").onclick = () => $("cornerModal").classList.add("hidden");
  $("cornerApply").onclick = async () => { $("cornerModal").classList.add("hidden"); await uploadOne(state.current.file, cornerState.corners.map(p => [+p[0].toFixed(4), +p[1].toFixed(4)]), state.current); };
  function drawCorners() {
    const c = $("cornerCanvas"), img = cornerState.img; if (!img) return;
    const maxW = Math.min(720, window.innerWidth * 0.9), scale = Math.min(maxW / img.width, 560 / img.height);
    c.width = Math.round(img.width * scale); c.height = Math.round(img.height * scale);
    const g = c.getContext("2d"); g.drawImage(img, 0, 0, c.width, c.height);
    const pts = cornerState.corners.map(p => [p[0] * c.width, p[1] * c.height]);
    g.strokeStyle = "#2563eb"; g.lineWidth = 2; g.beginPath(); pts.forEach((p, i) => i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])); g.closePath(); g.stroke();
    pts.forEach(p => { g.fillStyle = "#fff"; g.beginPath(); g.arc(p[0], p[1], 9, 0, Math.PI * 2); g.fill(); g.strokeStyle = "#2563eb"; g.stroke(); });
  }
  const cc = $("cornerCanvas");
  const cornerAt = (e) => { const r = cc.getBoundingClientRect(); const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height; return [x, y]; };
  cc.addEventListener("pointerdown", e => { const [x, y] = cornerAt(e); cornerState.drag = cornerState.corners.findIndex(p => Math.hypot(p[0] - x, p[1] - y) < 0.04); if (cornerState.drag >= 0) cc.setPointerCapture(e.pointerId); });
  cc.addEventListener("pointermove", e => { if (cornerState.drag < 0) return; const [x, y] = cornerAt(e); cornerState.corners[cornerState.drag] = [Math.max(0, Math.min(1, x)), Math.max(0, Math.min(1, y))]; drawCorners(); });
  cc.addEventListener("pointerup", () => cornerState.drag = -1);

  /* ── 최근 작업 ── */
  async function loadJobs() {
    if (!S.getTeacher()) return;
    try {
      const j = await api("/api/jobs?all=" + ($("showAll").checked ? 1 : 0));
      const box = $("jobList"); box.innerHTML = "";
      j.jobs.forEach(job => {
        const d = document.createElement("div"); d.className = "jobcard";
        const img = document.createElement("img"); img.src = job.result_url || job.thumbnail_url || "";
        const meta = document.createElement("div"); meta.className = "meta"; meta.textContent = `${job.teacher} · ${job.exam_name || ""} · ${(job.created_at || "").slice(5, 16).replace("T", " ")}`;
        d.append(img, meta); d.onclick = () => { const local = { ...job, name: job.original_filename || job.job_id, status: "불러옴" }; state.jobs.unshift(local); selectJob(local); };
        box.appendChild(d);
      });
    } catch (e) {}
  }
  $("showAll").onchange = loadJobs;
  async function loadExams() { try { const j = await fetch("/api/exams").then(r => r.json()); $("examList").innerHTML = j.exams.map(n => `<option value="${n.replace(/"/g, "&quot;")}">`).join(""); } catch (e) {} }

  /* ── 강사 선택 · 첫 접속 ── */
  function fillTeachers(select) { select.innerHTML = '<option value="">강사 선택</option>' + state.config.teachers.map(t => `<option>${t}</option>`).join(""); select.value = S.getTeacher(); }
  $("teacherSelect").onchange = (e) => { S.setTeacher(e.target.value); refreshUsage(); loadJobs(); };
  $("welcomeOk").onclick = () => { const v = $("welcomeTeacher").value; if (!v) { alert("이름을 골라 주세요."); return; } S.setTeacher(v); S.markWelcome(); $("teacherSelect").value = v; $("welcomeModal").classList.add("hidden"); refreshUsage(); loadJobs(); };

  /* ── 시작 ── */
  (async function init() {
    state.config = await fetch("/api/config").then(r => r.json());
    fillTeachers($("teacherSelect")); fillTeachers($("welcomeTeacher"));
    loadSliderDefaults(); S.wire(); updateLocks(); loadExams();
    if (!S.getTeacher() || !S.seenWelcome()) $("welcomeModal").classList.remove("hidden");
    else { refreshUsage(); loadJobs(); }
  })();
})();
