const API = import.meta.env.VITE_API_URL || "";

export async function getWorkspaces() {
  const r = await fetch(`${API}/api/workspaces`);
  if (!r.ok) throw new Error("backend unreachable");
  return r.json();
}

export async function askQuestion(question, workspaceId) {
  const r = await fetch(`${API}/api/chat/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, workspace_id: workspaceId }),
  });
  return r.json();
}

export async function uploadPdf(workspaceId, file) {
  const form = new FormData();
  form.append("file", file);
  const r = await fetch(`${API}/api/documents/upload?workspace_id=${workspaceId}`, {
    method: "POST", body: form,
  });
  return r.json();
}

export async function generate(kind, workspaceId) {
  const r = await fetch(`${API}/api/generate/${kind}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ workspace_id: workspaceId, count: 5 }),
  });
  return r.json();
}
