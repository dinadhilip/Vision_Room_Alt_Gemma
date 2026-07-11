const form = document.querySelector("#chatForm");
const input = document.querySelector("#messageInput");
const messages = document.querySelector("#messages");
const surface = document.querySelector("#visualSurface");
const status = document.querySelector("#status");
const uploadForm = document.querySelector("#uploadForm");
const frameFile = document.querySelector("#frameFile");
const frameCaption = document.querySelector("#frameCaption");
const skillStatus = document.querySelector("#skillStatus");
const resetSession = document.querySelector("#resetSession");

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

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!frameFile.files.length || !frameCaption.value.trim()) return;

  status.textContent = "Indexing";
  const body = new FormData();
  body.append("frame", frameFile.files[0]);
  body.append("caption", frameCaption.value.trim());
  body.append("video_id", "frontend_upload");
  body.append("timestamp_s", "0");

  try {
    const response = await fetch("/ingest/frame", { method: "POST", body });
    const payload = await response.json();
    appendMessage("assistant", `Indexed: ${payload.frame.caption}`);
    frameFile.value = "";
    frameCaption.value = "";
    status.textContent = "Ready";
  } catch (error) {
    appendMessage("assistant", "I could not index that frame yet.");
    status.textContent = "Ready";
  }
});

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
    renderSkillStatus(payload.components || []);
  } catch (error) {
    skillStatus.innerHTML = "";
  }
}

function renderSkillStatus(components) {
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
  caption.textContent = primary.caption || "";
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
  caption.textContent = `Generated preview ${payload.video_id}`;
  surface.appendChild(caption);
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
