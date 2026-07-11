const form = document.querySelector("#chatForm");
const input = document.querySelector("#messageInput");
const messages = document.querySelector("#messages");
const surface = document.querySelector("#visualSurface");
const status = document.querySelector("#status");
const loadFolderForm = document.querySelector("#loadFolderForm");
const folderPath = document.querySelector("#folderPath");
const ingestStatusText = document.querySelector("#ingestStatus");
const skillStatus = document.querySelector("#skillStatus");
const resetSession = document.querySelector("#resetSession");
let ingestPollInterval = null;

let sessionId = localStorage.getItem("vision-room-session");

appendMessage("assistant", "Tell me what moment to find in your local footage.");
loadSkillStatus();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  appendMessage("user", message);
  status.textContent = "Thinking";

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message }),
    });
    const payload = await response.json();
    sessionId = payload.session_id;
    localStorage.setItem("vision-room-session", sessionId);
    appendMessage("assistant", payload.reply);
    renderAction(payload.ui_action);
    status.textContent = "Ready";
  } catch (error) {
    appendMessage("assistant", "The bridge is not responding yet. Start the FastAPI server and try again.");
    status.textContent = "Offline";
  }
});

loadFolderForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const path = folderPath.value.trim();
  if (!path) return;

  status.textContent = "Starting Index...";
  try {
    const response = await fetch("/ingest/local-folder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder_path: path }),
    });
    const payload = await response.json();
    if (payload.error) {
      appendMessage("assistant", `Ingestion error: ${payload.error}`);
      status.textContent = "Ready";
      return;
    }
    
    appendMessage("assistant", `Started indexing folder: ${path}`);
    folderPath.value = "";
    
    // Start polling status
    if (ingestPollInterval) clearInterval(ingestPollInterval);
    ingestPollInterval = setInterval(pollIngestStatus, 1000);
  } catch (error) {
    appendMessage("assistant", "I could not start indexing that folder.");
    status.textContent = "Ready";
  }
});

async function pollIngestStatus() {
  try {
    const response = await fetch("/ingest/status");
    const payload = await response.json();
    
    ingestStatusText.textContent = `Status: ${payload.progress} (Indexed: ${payload.indexed_frames})`;
    
    if (!payload.is_running && payload.progress.includes("Finished")) {
      clearInterval(ingestPollInterval);
      status.textContent = "Ready";
      appendMessage("assistant", `Indexing complete. ${payload.indexed_frames} total frames in index.`);
      loadSkillStatus(); // Refresh skills to show semantic search is ready
    }
  } catch (error) {
    console.error("Failed to poll ingest status:", error);
  }
}

resetSession.addEventListener("click", async () => {
  if (!sessionId) {
    surface.innerHTML = '<div class="empty-state"><span>Ask for a moment in your footage</span></div>';
    return;
  }
  status.textContent = "Resetting";
  try {
    await fetch(`/session/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
    appendMessage("assistant", "Session reset. Tell me what moment to find next.");
    surface.innerHTML = '<div class="empty-state"><span>Ask for a moment in your footage</span></div>';
    status.textContent = "Ready";
    await loadSkillStatus();
  } catch (error) {
    appendMessage("assistant", "I could not reset this session.");
    status.textContent = "Ready";
  }
});

async function loadSkillStatus() {
  try {
    const response = await fetch("/skills");
    const payload = await response.json();
    renderSkillStatus(payload.components || [], payload.model_manager);
  } catch (error) {
    skillStatus.innerHTML = "";
  }
}

function renderSkillStatus(components, modelManager) {
  skillStatus.innerHTML = "";
  components.forEach((component) => {
    const chip = document.createElement("span");
    chip.className = `skill-chip ${component.status}`;
    chip.title = component.detail || component.label;
    chip.textContent = component.id
      .replace("frontend_chat_gemma", "Chat")
      .replace("semantic_search", "Search")
      .replace("nano_banana_lite", "NB Lite")
      .replace("omni_flash", "Omni")
      .replace("on_demand_index_backend", "Index");
    skillStatus.appendChild(chip);
  });
  
  if (modelManager) {
    const healthChip = document.createElement("span");
    healthChip.className = `skill-chip ${modelManager.healthy ? 'configured' : 'deterministic_fallback'}`;
    healthChip.title = modelManager.healthy ? 'Models loaded' : 'Models unavailable';
    healthChip.textContent = `Models: ${modelManager.healthy ? 'OK' : 'Error'}`;
    skillStatus.appendChild(healthChip);
  }
}

function appendMessage(role, text) {
  const element = document.createElement("div");
  element.className = `message ${role}`;
  element.textContent = text;
  messages.appendChild(element);
  messages.scrollTop = messages.scrollHeight;
}

function renderAction(action) {
  if (!action || action.type === "none") return;
  if (action.type === "show_frame_gallery") {
    renderFrameGallery(action.payload);
  }
  if (action.type === "show_generated_video") {
    renderGeneratedVideo(action.payload);
  }
  if (action.type === "show_storyboard") {
    renderStoryboard(action.payload);
  }
}

function renderFrameGallery(payload) {
  const primary = payload.primary;
  const frames = payload.frames || [];
  const confirmedFrame = payload.confirmed_frame || primary.frame_id;
  surface.innerHTML = "";

  const image = document.createElement("img");
  image.className = "hero-frame";
  image.src = assetUrl(primary.frame_path);
  image.alt = primary.caption || "Selected frame";
  surface.appendChild(image);

  const caption = document.createElement("div");
  caption.className = "caption";
  caption.textContent = providerLine(primary.caption || "", primary);
  surface.appendChild(caption);

  if (frames.length > 1) {
    const strip = document.createElement("div");
    strip.className = "thumb-strip";
    frames.forEach((frame) => {
      const item = document.createElement("button");
      item.className = `thumb ${frame.frame_id === confirmedFrame ? "selected" : ""}`;
      item.type = "button";
      item.title = "Select frame";
      item.addEventListener("click", () => confirmFrame(frame.frame_id));
      const thumb = document.createElement("img");
      thumb.src = assetUrl(frame.frame_path);
      thumb.alt = frame.caption || "Alternative frame";
      const label = document.createElement("p");
      label.textContent = `${frame.timestamp_s?.toFixed?.(1) ?? ""}s · ${frame.score ?? ""}`;
      item.append(thumb, label);
      strip.appendChild(item);
    });
    surface.appendChild(strip);
  }
}

async function confirmFrame(frameId) {
  if (!sessionId || !frameId) return;
  status.textContent = "Selecting";
  try {
    const response = await fetch("/session/confirm-frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, frame_id: frameId }),
    });
    const payload = await response.json();
    appendMessage("assistant", payload.reply);
    renderAction(payload.ui_action);
    status.textContent = "Ready";
  } catch (error) {
    appendMessage("assistant", "I could not select that frame.");
    status.textContent = "Ready";
  }
}

function renderGeneratedVideo(payload) {
  surface.innerHTML = "";
  const image = document.createElement("img");
  image.className = "hero-frame";
  image.src = payload.video_url;
  image.alt = "Generated video preview";
  surface.appendChild(image);

  const caption = document.createElement("div");
  caption.className = "caption";
  caption.textContent = providerLine(`Generated preview ${payload.video_id}`, payload);
  surface.appendChild(caption);
}

function renderStoryboard(payload) {
  surface.innerHTML = "";
  
  const container = document.createElement("div");
  container.className = `storyboard-container storyboard-${payload.style}`;
  
  const title = document.createElement("h2");
  title.className = "storyboard-title";
  title.textContent = `Storyboard: ${payload.story}`;
  container.appendChild(title);
  
  const grid = document.createElement("div");
  grid.className = "storyboard-grid";
  
  payload.frames.forEach((frame, index) => {
    const panel = document.createElement("div");
    panel.className = "storyboard-panel";
    
    const img = document.createElement("img");
    img.src = assetUrl(frame.frame_path);
    img.alt = `Panel ${index + 1}`;
    
    const caption = document.createElement("div");
    caption.className = "panel-caption";
    caption.textContent = `Panel ${index + 1}: ${providerLine("", frame)}`;
    
    panel.append(img, caption);
    grid.appendChild(panel);
  });
  
  container.appendChild(grid);
  surface.appendChild(container);
}

function providerLine(text, payload) {
  const provider = payload.provider ? ` · ${payload.provider}` : "";
  const fallback = payload.fallback ? " · fallback" : "";
  const attempts = payload.attempts ? ` · ${payload.attempts} attempt${payload.attempts === 1 ? "" : "s"}` : "";
  return `${text}${provider}${fallback}${attempts}`;
}

function assetUrl(path) {
  if (!path) return "";
  if (path.startsWith("/assets/")) return path;
  const normalized = path.replaceAll("\\", "/");
  const uploadsIndex = normalized.lastIndexOf("/data/uploads/");
  if (uploadsIndex >= 0) return `/assets/uploads/${normalized.slice(uploadsIndex + "/data/uploads/".length)}`;
  const generatedIndex = normalized.lastIndexOf("/data/generated/");
  if (generatedIndex >= 0) return `/assets/generated/${normalized.slice(generatedIndex + "/data/generated/".length)}`;
  return path;
}
