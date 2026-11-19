import { useState } from "react";
import { generate, getDueFlashcards, reviewFlashcard, uploadPdf } from "../api";

const GRADE_BUTTONS = [
  { label: "Again", grade: 1 },
  { label: "Hard", grade: 3 },
  { label: "Good", grade: 4 },
  { label: "Easy", grade: 5 },
];

export default function GeneratePanel({ workspaceId }) {
  const [out, setOut] = useState("");
  const [msg, setMsg] = useState("");
  const [due, setDue] = useState(null);
  const [flipped, setFlipped] = useState(false);

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

  async function startReview() {
    const r = await getDueFlashcards(workspaceId);
    setDue(r.flashcards || []);
    setFlipped(false);
  }

  async function grade(gradeValue) {
    const card = due[0];
    await reviewFlashcard(card.id, gradeValue);
    setDue(due.slice(1));
    setFlipped(false);
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
        <button onClick={startReview}>Review due</button>
      </div>
      {out && <pre className="output">{out}</pre>}
      {due && (
        <div className="review">
          {due.length === 0 ? (
            <p className="mode">No flashcards due. Generate some, or come back later.</p>
          ) : (
            <>
              <p className="mode">{due.length} card{due.length === 1 ? "" : "s"} due</p>
              <div className="flashcard" onClick={() => setFlipped(!flipped)}>
                <p>{flipped ? due[0].back : due[0].front}</p>
                <p className="mode">{flipped ? "Click to see the front" : "Click to flip"}</p>
              </div>
              {flipped && (
                <div className="row">
                  {GRADE_BUTTONS.map((b) => (
                    <button key={b.label} onClick={() => grade(b.grade)}>{b.label}</button>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </section>
  );
}
