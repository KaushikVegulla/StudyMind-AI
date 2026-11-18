export default function Sidebar({ workspaces, active, onSelect }) {
  return (
    <aside className="sidebar">
      <h1>StudyMind AI</h1>
      <p className="tagline">Answers with citations, from your own notes.</p>
      <h2>Subjects</h2>
      <ul>
        {workspaces.map((w) => (
          <li key={w.id}>
            <button className={w.id === active ? "active" : ""} onClick={() => onSelect(w.id)}>
              {w.name}
            </button>
          </li>
        ))}
      </ul>
      <p className="hint">Upload a PDF, then ask a question. If the answer is not in your notes, StudyMind says so.</p>
    </aside>
  );
}
