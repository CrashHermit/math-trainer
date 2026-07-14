"""Canonical node/edge labels and the Docling-item → Element-type mapping.

Neo4j has no type inheritance, so every content node carries the base label
``Element`` plus one concrete type label, e.g. ``(:Element:Paragraph)``. A single
vector index and generic Next-chaining target ``:Element``; type-specific stages
filter on the concrete label.
"""
from __future__ import annotations

from enum import StrEnum


class NodeType(StrEnum):
    """Node labels used throughout the graph."""

    # Structural
    SOURCE = "Source"
    SEGMENT = "Segment"

    # Element base + concrete content types
    ELEMENT = "Element"
    PARAGRAPH = "Paragraph"
    HEADING = "Heading"
    MATH = "Math"
    TABLE = "Table"
    CAPTION = "Caption"
    LIST = "List"
    LIST_ITEM = "ListItem"
    CODE = "Code"
    IMAGE = "Image"
    # Pedagogical types the Extractor/Refiner may promote to
    ADMONITION = "Admonition"
    INSTRUCTION = "Instruction"
    ACTIVITY = "Activity"


# Concrete Element subtypes (everything that is an :Element besides the base).
ELEMENT_SUBTYPES: frozenset[NodeType] = frozenset(
    {
        NodeType.PARAGRAPH,
        NodeType.HEADING,
        NodeType.MATH,
        NodeType.TABLE,
        NodeType.CAPTION,
        NodeType.LIST,
        NodeType.LIST_ITEM,
        NodeType.CODE,
        NodeType.IMAGE,
        NodeType.ADMONITION,
        NodeType.INSTRUCTION,
        NodeType.ACTIVITY,
    }
)


class EdgeType(StrEnum):
    """Relationship types."""

    CONTAINS = "Contains"   # Source→Segment, Segment→Element (membership/structure)
    HAS = "Has"             # Source→head Element (entry into the reading chain)
    NEXT = "Next"           # Element→Element (reading order)


# Docling DocItemLabel (lower-cased) → Element subtype. Unknown labels fall back
# to Paragraph. Docling labels: https://github.com/docling-project/docling-core
DOCLING_LABEL_TO_TYPE: dict[str, NodeType] = {
    "text": NodeType.PARAGRAPH,
    "paragraph": NodeType.PARAGRAPH,
    "section_header": NodeType.HEADING,
    "title": NodeType.HEADING,
    "page_header": NodeType.HEADING,
    "formula": NodeType.MATH,
    "equation": NodeType.MATH,
    "table": NodeType.TABLE,
    "caption": NodeType.CAPTION,
    "list": NodeType.LIST,
    "list_item": NodeType.LIST_ITEM,
    "ordered_list": NodeType.LIST,
    "unordered_list": NodeType.LIST,
    "code": NodeType.CODE,
    "picture": NodeType.IMAGE,
    "figure": NodeType.IMAGE,
}


def docling_label_to_type(label: str) -> NodeType:
    """Map a Docling item label to an Element subtype (default Paragraph)."""
    return DOCLING_LABEL_TO_TYPE.get(label.lower().strip(), NodeType.PARAGRAPH)


# Types the Embedder embeds as text via their `content`. Image is embedded via
# its blurb and is handled separately by the embedder.
TEXT_EMBEDDABLE_TYPES: frozenset[NodeType] = frozenset(
    {
        NodeType.PARAGRAPH,
        NodeType.HEADING,
        NodeType.MATH,
        NodeType.TABLE,
        NodeType.CAPTION,
        NodeType.LIST,
        NodeType.LIST_ITEM,
        NodeType.CODE,
        NodeType.ADMONITION,
        NodeType.INSTRUCTION,
        NodeType.ACTIVITY,
    }
)

def element_subtype(node: dict) -> NodeType | None:
    """The concrete Element subtype of a node dict (needs ``_labels`` projection)."""
    subtype_values = {t.value for t in ELEMENT_SUBTYPES}
    for label in node.get("_labels", []):
        if label in subtype_values:
            return NodeType(label)
    return None


def has_type(node: dict, node_type: NodeType) -> bool:
    return node_type.value in (node.get("_labels") or [])


# Types the Refiner has a specialized signature for.
REFINABLE_TYPES: frozenset[NodeType] = frozenset(
    {
        NodeType.CODE,
        NodeType.ACTIVITY,
        NodeType.INSTRUCTION,
        NodeType.ADMONITION,
    }
)
