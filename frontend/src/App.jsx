import { useEffect, useState } from "react";
import Sidebar from "./components/Sidebar";
import ChatPanel from "./components/ChatPanel";
import GeneratePanel from "./components/GeneratePanel";
import { getWorkspaces } from "./api";

const FALLBACK = [
  { id: "local-dsa", name: "Data Structures" },
  { id: "local-db", name: "Databases" },
  { id: "local-cloud", name: "Cloud" },
];

export default function App() {
  const [workspaces, setWorkspaces] = useState(FALLBACK);
  const [active, setActive] = useState(FALLBACK[0].id);

  useEffect(() => {
    getWorkspaces()
      .then((ws) => { if (ws.length) { setWorkspaces(ws); setActive(ws[0].id); } })
      .catch(() => {});
  }, []);

  return (
    <div className="layout">
      <Sidebar workspaces={workspaces} active={active} onSelect={setActive} />
      <main>
        <ChatPanel workspaceId={active} />
        <GeneratePanel workspaceId={active} />
      </main>
    </div>
  );
}
