import { useState } from "react";
import { generate, uploadPdf } from "../api";

export default function GeneratePanel({ workspaceId }) {
  const [out, setOut] = useState("");
  const [msg, setMsg] = useState("");

  async function onUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    const r = await uploadPdf(workspaceId, file);
    setMsg(r.filename ? `Ingested ${r.filename}: ${r.pages} pages, ${r.chunk_count} chunks` : (r.detail || "Upload failed"));
  }

  async function run(kind) {
    const r = await generate(kind, workspaceId);
    setOut(JSON.stringify(r, null, 2));
  }

  return (
    <section className="panel">
      <h2>Material</h2>
      <input type="file" accept="application/pdf" onChange={onUpload} />
      {msg && <p className="mode">{msg}</p>}
      <div className="row">
        <button onClick={() => run("summary")}>Summary</button>
        <button onClick={() => run("flashcards")}>Flashcards</button>
        <button onClick={() => run("quiz")}>Quiz</button>
      </div>
      {out && <pre className="output">{out}</pre>}
    </section>
  );
}
