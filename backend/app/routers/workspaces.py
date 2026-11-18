from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import store

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


class WorkspaceIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = ""


@router.get("")
def list_workspaces():
    return list(store.workspaces.values())


@router.post("", status_code=201)
def create_workspace(body: WorkspaceIn):
    wid = store.next_id("ws")
    ws = {"id": wid, "name": body.name, "description": body.description}
    store.workspaces[wid] = ws
    return ws


@router.get("/{workspace_id}")
def get_workspace(workspace_id: str):
    ws = store.workspaces.get(workspace_id)
    if not ws:
        raise HTTPException(404, "Workspace not found")
    return ws


@router.delete("/{workspace_id}", status_code=204)
def delete_workspace(workspace_id: str):
    if workspace_id not in store.workspaces:
        raise HTTPException(404, "Workspace not found")
    del store.workspaces[workspace_id]
    for cid in [c for c, ch in store.chunks.items() if ch["workspace_id"] == workspace_id]:
        del store.chunks[cid]
    for did in [d for d, doc in store.documents.items() if doc["workspace_id"] == workspace_id]:
        del store.documents[did]
