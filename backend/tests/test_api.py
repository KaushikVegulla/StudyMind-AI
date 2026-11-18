import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_seeded_workspaces():
    r = client.get("/api/workspaces")
    assert r.status_code == 200
    names = {w["name"] for w in r.json()}
    assert {"Data Structures", "Databases", "Cloud"} <= names


def test_create_workspace():
    r = client.post("/api/workspaces", json={"name": "Maths", "description": "test"})
    assert r.status_code == 201
    assert r.json()["name"] == "Maths"


def test_ask_without_material_says_not_found():
    r = client.post("/api/chat/ask", json={"question": "What is a splay tree rotation?"})
    assert r.status_code == 200
    assert r.json()["mode"] in ("no_match", "extractive_scaffold", "llm")
