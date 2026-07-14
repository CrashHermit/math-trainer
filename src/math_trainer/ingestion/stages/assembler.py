"""Stage 7 — Assembler.

Builds the ``:Block`` overlay: the semantic-unit layer that becomes the embed /
retrieval unit. Elements stay untouched as the structural backbone; each Block
``Groups`` a contiguous run of Elements.

Stateful, sequential walk over the reading chain:
  * **Anchors** (Activity / Instruction / Admonition — units the Extractor already
    found) open a Block and absorb the components that belong to them (e.g. an
    exercise's figure). They also close any open ungrouped region.
  * **Ungrouped content** (Paragraph / Heading / Math / Table / Caption / List /
    Code / Image without an owner) accumulates into a token-bounded **main window**;
    when it fills, an LLM segments it into units. The LLM commits complete units and
    defers a trailing incomplete unit, which is carried into the next window
    (greedy-to-unit-boundary: a unit never splits across windows). A read-only
    **context window** of upcoming blocks is peeked so the LLM can see past the edge.

Idempotent: existing Blocks for the source are dropped and rebuilt (the Element
backbone is never touched).
"""

from math_trainer.core.config import StageConfig
from math_trainer.core.model.types import (
    ABSORBABLE_TYPES,
    ANCHOR_TYPES,
    BlockKind,
    EdgeType,
    NodeType,
    element_subtype,
    has_type,
    is_anchor,
)
from math_trainer.ingestion.stages.base import LMModule
from math_trainer.storage.neo4j.repository import GraphRepository


def _approx_tokens(text: str | None) -> int:
    return max(1, len(text or "") // 4)


def _read(obj: object, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class AssemblerStage:
    name = "assembler"

    def __init__(self, repo: GraphRepository, module: LMModule, config: StageConfig) -> None:
        self._repo = repo
        self._module = module
        self._config = config
        self._main_window = config.main_window_tokens
        self._context_window = config.context_window_tokens
        self._last_summary: str | None = None

    async def run(self, source_uuid: str) -> None:
        elements = await self._repo.source_elements_ordered(source_uuid)
        await self._repo.execute(
            "MATCH (b:`Block` {source_uuid: $s}) DETACH DELETE b", s=source_uuid
        )
        self._last_summary = None

        created: list[str] = []
        order = [0]
        pending: list[dict] = []
        i, n = 0, len(elements)

        while i < n:
            el = elements[i]
            if is_anchor(el):
                # Hard boundary: flush the open prose region, then open the anchor.
                await self._flush(source_uuid, pending, [], created, order, final=True)
                pending = []
                members = [el]
                j = i + 1
                while j < n and element_subtype(elements[j]) in ABSORBABLE_TYPES:
                    members.append(elements[j])
                    j += 1
                await self._create_anchor_block(source_uuid, el, members, created, order)
                i = j
            else:
                pending.append(el)
                if self._tokens(pending) >= self._main_window:
                    trailing = self._peek(elements, i + 1)
                    pending = await self._flush(source_uuid, pending, trailing, created, order, final=False)
                i += 1

        await self._flush(source_uuid, pending, [], created, order, final=True)
        await self._repo.link_chain(created, EdgeType.NEXT)

    # ── ungrouped-region flush (LLM segmentation) ───────────────────────────
    async def _flush(
        self, source_uuid: str, pending: list[dict], trailing: list[dict],
        created: list[str], order: list[int], *, final: bool,
    ) -> list[dict]:
        if not pending:
            return []

        prediction = await self._module.aforward(
            window=[self._payload(e) for e in pending],
            leading_context=self._last_summary,
            trailing=[self._payload(e) for e in trailing],
        )
        if prediction is None:
            units = [{"kind": BlockKind.PROSE.value, "members": [e["uuid"] for e in pending]}]
            deferred: set[str] = set()
        else:
            units = _read(prediction, "units") or [
                {"kind": BlockKind.PROSE.value, "members": [e["uuid"] for e in pending]}
            ]
            deferred = set() if final else set(_read(prediction, "deferred") or [])

        by_uuid = {e["uuid"]: e for e in pending}
        committed: set[str] = set()
        for unit in units:
            member_uuids = [
                mu for mu in (_read(unit, "members") or [])
                if mu in by_uuid and mu not in deferred and mu not in committed
            ]
            if not member_uuids:
                continue
            await self._create_block(
                source_uuid, _read(unit, "kind") or BlockKind.PROSE.value,
                [by_uuid[mu] for mu in member_uuids], created, order,
                label=_read(unit, "label"),
            )
            committed.update(member_uuids)

        leftovers = [e for e in pending if e["uuid"] not in committed and e["uuid"] not in deferred]
        if leftovers:
            if final:
                await self._create_block(source_uuid, BlockKind.PROSE.value, leftovers, created, order)
                committed.update(e["uuid"] for e in leftovers)
            else:
                deferred.update(e["uuid"] for e in leftovers)

        return [e for e in pending if e["uuid"] in deferred]

    # ── block creation ──────────────────────────────────────────────────────
    async def _create_anchor_block(
        self, source_uuid: str, anchor: dict, members: list[dict],
        created: list[str], order: list[int],
    ) -> None:
        kind = ANCHOR_TYPES[element_subtype(anchor)]
        prefix = None
        if kind is BlockKind.EXERCISE:
            prefix = await self._repo.governing_instruction_text(anchor["uuid"])
        await self._create_block(
            source_uuid, kind.value, members, created, order,
            label=anchor.get("label"), prefix=prefix,
        )

    async def _create_block(
        self, source_uuid: str, kind: str, members: list[dict],
        created: list[str], order: list[int], *, label: str | None = None,
        prefix: str | None = None,
    ) -> None:
        order[0] += 1
        content = self._compose(members, prefix)
        block = await self._repo.create_node(
            [NodeType.BLOCK], source_uuid=source_uuid, order_index=order[0],
            kind=kind, label=label, content=content,
        )
        await self._repo.link(EdgeType.CONTAINS, source_uuid, block["uuid"])
        await self._repo.link_children(
            block["uuid"], [m["uuid"] for m in members], EdgeType.GROUPS
        )
        created.append(block["uuid"])
        self._last_summary = f"{kind}: {content[:80]}"

    # ── helpers ─────────────────────────────────────────────────────────────
    def _compose(self, members: list[dict], prefix: str | None) -> str:
        parts: list[str] = []
        if prefix:
            parts.append(prefix)
        for m in members:
            if has_type(m, NodeType.IMAGE):
                if m.get("blurb"):
                    parts.append(f"[figure] {m['blurb']}")
            elif m.get("content"):
                parts.append(m["content"])
        return "\n\n".join(parts)

    def _payload(self, el: dict) -> dict:
        subtype = element_subtype(el)
        return {
            "uuid": el["uuid"],
            "type": subtype.value if subtype else "Element",
            "content": el.get("content") or el.get("blurb") or "",
        }

    def _tokens(self, els: list[dict]) -> int:
        return sum(_approx_tokens(e.get("content") or e.get("blurb")) for e in els)

    def _peek(self, elements: list[dict], start: int) -> list[dict]:
        out: list[dict] = []
        budget = 0
        for el in elements[start:]:
            out.append(el)
            budget += _approx_tokens(el.get("content") or el.get("blurb"))
            if budget >= self._context_window:
                break
        return out
