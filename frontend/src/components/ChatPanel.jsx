import { useState } from "react";
import { askQuestion } from "../api";
import CitationCard from "./CitationCard";

export default function ChatPanel({ workspaceId }) {
  const [question, setQuestion] = useState("");
  const [style, setStyle] = useState("standard");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);

  async function submit(e) {
    e.preventDefault();
    if (!question.trim()) return;
    setBusy(true);
    try { setResult(await askQuestion(question, workspaceId, style)); }
    finally { setBusy(false); }
  }

  return (
    <section className="panel">
      <h2>Ask your notes</h2>
      <form onSubmit={submit}>
        <input value={question} onChange={(e) => setQuestion(e.target.value)}
               placeholder="e.g. What is a B-tree and when is it used?" />
        <select value={style} onChange={(e) => setStyle(e.target.value)}
                title="Answer style">
          <option value="concise">Concise</option>
          <option value="standard">Standard</option>
          <option value="exam">Exam</option>
        </select>
        <button disabled={busy}>{busy ? "Searching..." : "Ask"}</button>
      </form>
      {result && (
        <div className="answer">
          <p>{result.answer}</p>
          {result.mode && <p className="mode">mode: {result.mode}</p>}
          {result.citations?.map((c, i) => <CitationCard key={i} citation={c} />)}
        </div>
      )}
    </section>
  );
}
