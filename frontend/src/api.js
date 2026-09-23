/**
 * Server calls, kept out of the components.
 *
 * The SSE parsing lives here because EventSource only supports GET and /ask is
 * a POST — so the stream is read from the fetch body and framed by hand. That
 * is the kind of detail that has no business inside a component.
 */

export async function streamAsk(question, onEvent, signal) {
  const response = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });

  if (!response.ok) {
    onEvent("error", { message: `The service returned ${response.status}.` });
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);

      const name = frame.match(/^event: (.+)$/m)?.[1];
      const data = frame.match(/^data: (.+)$/m)?.[1];
      if (name && data) onEvent(name, JSON.parse(data));
    }
  }
}

export const getProposals = () => fetch("/proposals").then((r) => r.json());

export async function resync() {
  const response = await fetch("/sync", { method: "POST" });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || "Could not re-read Jira.");
  return body;
}
export const getHealth = () => fetch("/health").then((r) => r.json());

export async function generateReport() {
  const response = await fetch("/report", { method: "POST" });
  const body = await response.json().catch(() => ({}));
  if (!response.ok)
    throw new Error(body.detail || "Could not generate the weekly summary.");
  return body;
}

export async function decide(id, verb, actor, note = "") {
  const response = await fetch(`/proposals/${id}/${verb}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Actor": actor },
    body: JSON.stringify({ note }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || "That change could not be recorded.");
  }
  return response.json();
}
