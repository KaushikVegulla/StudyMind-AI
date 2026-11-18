from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..repositories import get_repository

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


class WorkspaceIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = ""


@router.get("")
def list_workspaces():
    return get_repository().list_workspaces()


@router.post("", status_code=201)
def create_workspace(body: WorkspaceIn):
    return get_repository().create_workspace(body.name, body.description)


@router.get("/{workspace_id}")
def get_workspace(workspace_id: str):
    ws = get_repository().get_workspace(workspace_id)
    if not ws:
        raise HTTPException(404, "Workspace not found")
    return ws


@router.delete("/{workspace_id}", status_code=204)
def delete_workspace(workspace_id: str):
    if not get_repository().delete_workspace(workspace_id):
        raise HTTPException(404, "Workspace not found")
