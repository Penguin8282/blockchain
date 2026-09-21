/* 강사 선택 · Anthropic 키 · 모델 선택. 전부 이 브라우저의 localStorage 에만 저장한다.
   서버에는 요청마다 헤더로 보내고, 서버는 요청 처리 중에만 메모리에 들고 있다. */
window.settings = (function () {
  const KEY = "cleanexam.apiKey", TEACHER = "cleanexam.teacher", MODEL = "cleanexam.model", SEEN = "cleanexam.seenWelcome";
  const get = (k, d = "") => { try { return localStorage.getItem(k) ?? d; } catch (e) { return d; } };
  const set = (k, v) => { try { v == null || v === "" ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch (e) {} };

  const api = {
    getTeacher: () => get(TEACHER), setTeacher: (v) => set(TEACHER, v),
    getKey: () => get(KEY), setKey: (v) => set(KEY, v),
    getModel: () => get(MODEL, "fast"), setModel: (v) => set(MODEL, v),
    hasKey: () => get(KEY).startsWith("sk-ant-"),
    seenWelcome: () => get(SEEN) === "1", markWelcome: () => set(SEEN, "1"),
    /* 서버로 보낼 헤더. 헤더는 ASCII 만 되므로 한글 이름은 encodeURIComponent 로 보낸다. */
    headers() {
      const h = { "X-Teacher": encodeURIComponent(api.getTeacher()), "X-Model": api.getModel() };
      if (api.hasKey()) h["X-Anthropic-Key"] = api.getKey();
      return h;
    },
  };

  /* ── 설정 모달 배선 ── */
  function wire() {
    const modal = document.getElementById("settingsModal");
    const keyInput = document.getElementById("keyInput");
    const eye = document.getElementById("keyEye");
    const status = document.getElementById("keyStatus");
    document.getElementById("openSettings").onclick = () => { keyInput.value = api.getKey(); status.textContent = ""; modal.classList.remove("hidden");
      document.querySelectorAll("input[name=model]").forEach(r => r.checked = r.value === api.getModel()); };
    document.getElementById("closeSettings").onclick = () => { modal.classList.add("hidden"); api.onChange && api.onChange(); };
    eye.onclick = () => { keyInput.type = keyInput.type === "password" ? "text" : "password"; };
    keyInput.oninput = () => { api.setKey(keyInput.value.trim()); };
    document.getElementById("keyClear").onclick = () => { keyInput.value = ""; api.setKey(""); status.textContent = "키를 지웠어요."; status.className = "msg"; api.onChange && api.onChange(); };
    document.getElementById("keyCheck").onclick = async () => {
      api.setKey(keyInput.value.trim());
      status.textContent = "확인 중…"; status.className = "msg";
      try {
        const r = await fetch("/api/key/check", { method: "POST", headers: api.headers() });
        const j = await r.json();
        status.textContent = j.message || j.detail || "확인 결과를 받지 못했어요.";
        status.className = "msg " + (j.ok ? "ok" : "err");
      } catch (e) { status.textContent = "서버에 연결하지 못했어요."; status.className = "msg err"; }
      api.onChange && api.onChange();
    };
    document.querySelectorAll("input[name=model]").forEach(r => r.onchange = () => api.setModel(r.value));
  }
  api.wire = wire;
  return api;
})();
