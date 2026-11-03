"use strict";

const form = document.getElementById("provider-form");
const feedback = document.getElementById("feedback");

function display(message, error) {
  feedback.textContent = message;
  feedback.dataset.status = error ? "error" : "ok";
}

async function getStatus() {
  const response = await fetch("/api/status", {
    method: "GET", credentials: "same-origin", cache: "no-store"
  });
  if (!response.ok) throw new Error("Could not reach the analysis server.");
  const data = await response.json();
  if (!data.available || !data.csrf_token || !data.provider_setup_allowed)
    throw new Error("Provider setup is unavailable on this server.");
  return data;
}

function showStatus(status) {
  const market = status.data_provider_configured ? "connected" : "not connected";
  const model = status.research_model_configured ? "connected" : "not connected";
  display("Market data: " + market + ". Research model: " + model + ".", false);
}

form.addEventListener("submit", async function(event) {
  event.preventDefault();
  const submit = form.querySelector('button[type="submit"]');
  const marketInput = document.getElementById("massive-key");
  const modelInput = document.getElementById("anthropic-key");
  const payload = {};
  if (marketInput.value.trim()) payload.massive = marketInput.value.trim();
  if (modelInput.value.trim()) payload.anthropic = modelInput.value.trim();
  if (!Object.keys(payload).length) {
    display("Enter a provider key, or continue to recorded examples.", true);
    return;
  }
  submit.disabled = true;
  display("Connecting providers…", false);
  try {
    const status = await getStatus();
    const response = await fetch("/api/providers", {
      method: "POST",
      credentials: "same-origin",
      headers: {
        "Content-Type": "application/json",
        "X-Portfolio-Impact": "1",
        "X-CSRF-Token": status.csrf_token
      },
      body: JSON.stringify(payload)
    });
    const reply = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(reply.error || "Could not connect providers.");
    const updated = await getStatus();
    if (updated.data_provider_configured && updated.research_model_configured) {
      window.location.assign("./index.html");
    } else {
      showStatus(updated);
    }
  } catch (error) {
    display(error.message || "Connection failed.", true);
  } finally {
    marketInput.value = "";
    modelInput.value = "";
    submit.disabled = false;
  }
});

getStatus().then(showStatus).catch(error => display(error.message, true));
