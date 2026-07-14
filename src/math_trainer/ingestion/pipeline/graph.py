"""Minimal LangGraph pipeline: a linear chain of the 7 stages.

Kept simple by design — each node runs one stage and records progress. Resume is
expressed by ``from_stage``: nodes numbered below it no-op, so re-running the graph
from the start naturally skips completed work.
"""

from typing import TYPE_CHECKING, TypedDict

from langgraph.graph import END, START, StateGraph

if TYPE_CHECKING:
    from math_trainer.ingestion.service import IngestionService


class PipelineState(TypedDict, total=False):
    source_path: str
    source_uuid: str | None
    title: str | None
    from_stage: int


def build_pipeline(service: "IngestionService"):
    builder: StateGraph = StateGraph(PipelineState)

    async def stage_1(state: PipelineState) -> dict:
        if state.get("from_stage", 1) > 1:
            return {}
        uid = await service._provider.create_document(
            source=state["source_path"],
            document_title=state.get("title"),
            source_uuid=state.get("source_uuid"),
        )
        await service._set_stage(uid, 1)
        return {"source_uuid": uid}

    def _make_node(stage_no: int):
        stage = service._stages[stage_no]

        async def node(state: PipelineState) -> dict:
            if state.get("from_stage", 1) > stage_no:
                return {}
            uid = state["source_uuid"]
            if not uid:
                raise ValueError(f"stage {stage_no} requires a source_uuid")
            await stage.run(uid)
            await service._set_stage(uid, stage_no)
            return {}

        return node

    builder.add_node("stage_1_docling", stage_1)
    prev = "stage_1_docling"
    for stage_no in sorted(service._stages):
        name = f"stage_{stage_no}_{service._stages[stage_no].name}"
        builder.add_node(name, _make_node(stage_no))
        builder.add_edge(prev, name)
        prev = name

    builder.add_edge(START, "stage_1_docling")
    builder.add_edge(prev, END)
    return builder.compile()
