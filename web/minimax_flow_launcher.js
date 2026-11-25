// Adds a "MiniMax H3 Vision" button to ComfyUI's sidebar. It opens the app full screen
// over ComfyUI; the app's own Close button (or Esc on an empty screen) closes it again.
// It also opens by itself the first time ComfyUI loads in a browser tab, unless "Open MiniMax H3 Vision when ComfyUI
// opens" is switched off in ComfyUI's settings. A reload opens it again if it was open, and leaves it closed if not.
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const FRAME_ID = "mmflow-frame";
const OPEN_ON_START = "MiniMaxFlow.OpenOnStart";
const OPENED_KEY = "mmvision.opened-on-start";   // in sessionStorage: once per browser tab
const OPEN_KEY = "mmvision.open";                 // in sessionStorage: open in this tab now, so a reload reopens it

function remember(open) {
  try {
    if (open) sessionStorage.setItem(OPEN_KEY, "1");
    else sessionStorage.removeItem(OPEN_KEY);
  } catch {
    // No sessionStorage: a reload goes to ComfyUI.
  }
}

function closeFlow() {
  document.getElementById(FRAME_ID)?.remove();
  remember(false);
}

function openFlow() {
  if (document.getElementById(FRAME_ID)) return;
  const frame = document.createElement("iframe");
  frame.id = FRAME_ID;
  // A new address each time, so the browser never shows an old copy after an update.
  frame.src = "/mmflow/app/index.html?v=" + Date.now();
  frame.title = "MiniMax H3 Vision";
  frame.allow = "fullscreen; clipboard-write";
  frame.style.cssText = "position:fixed;inset:0;width:100vw;height:100vh;border:0;z-index:10000;background:#121316";
  document.body.appendChild(frame);
  remember(true);
}

// Clip updates go to every window; this keeps the graph page from logging them as unknown.
api.addEventListener("mmflow.clip", () => {});

window.addEventListener("message", (event) => {
  if (event.origin === location.origin && event.data === "mmflow-close") closeFlow();
});

function setting(id, fallback) {
  try {
    const value = app.extensionManager?.setting?.get?.(id) ?? app.ui?.settings?.getSettingValue?.(id);
    return value === undefined || value === null ? fallback : value;
  } catch {
    return fallback;
  }
}

function openOnStart() {
  try {
    if (sessionStorage.getItem(OPEN_KEY)) { openFlow(); return; }
    if (sessionStorage.getItem(OPENED_KEY)) return;
    sessionStorage.setItem(OPENED_KEY, "1");
  } catch {
    // No sessionStorage: open anyway.
  }
  if (setting(OPEN_ON_START, true)) openFlow();
}

app.registerExtension({
  name: "minimax.flow",
  settings: [{
    id: OPEN_ON_START,
    category: ["MiniMax H3 Vision", "Start", "Open on start"],
    name: "Open MiniMax H3 Vision when ComfyUI opens",
    tooltip: "The first time ComfyUI loads in a browser tab, open MiniMax H3 Vision over it. Its Close button goes back to ComfyUI.",
    type: "boolean",
    defaultValue: true,
  }],
  commands: [{ id: "MiniMaxFlow.Open", label: "Open MiniMax H3 Vision", icon: "pi pi-video", function: openFlow }],
  menuCommands: [{ path: ["View"], commands: ["MiniMaxFlow.Open"] }],
  setup() {
    openOnStart();
    app.extensionManager?.registerSidebarTab?.({
      id: "minimax-flow",
      icon: "pi pi-video",
      title: "MiniMax H3 Vision",
      tooltip: "Open MiniMax H3 Vision",
      type: "custom",
      render(element) {
        const button = document.createElement("button");
        button.textContent = "Open MiniMax H3 Vision";
        button.style.cssText = "margin:16px;padding:10px 16px;border-radius:8px;border:0;cursor:pointer;background:#e0a24a;color:#121316;font:600 14px system-ui";
        button.addEventListener("click", openFlow);
        element.replaceChildren(button);
        openFlow();
      },
    });
  },
});
