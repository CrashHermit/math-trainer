"""DSPy Signatures for the intelligent stages.

The input/output field names match each stage's ``module.aforward(**kwargs)`` call
and the attributes each stage reads off the prediction.
"""
from __future__ import annotations

import dspy
from pydantic import BaseModel


class PictureFilterSignature(dspy.Signature):
    """Decide whether a picture from a technical/math textbook is substantive
    (carries information a reader would lose) versus decorative/navigational noise.
    Use the full page image for context."""

    page_image: dspy.Image | None = dspy.InputField(desc="The page the picture sits on (context).")
    picture_image: dspy.Image | None = dspy.InputField(desc="The picture to judge.")
    blurb: str | None = dspy.InputField(desc="Docling's auto-generated description, if any.")
    is_substantive: bool = dspy.OutputField(desc="True to keep, False to discard.")


class CleanerSignature(dspy.Signature):
    """Ensure the element's content is clean, well-formed Markdown with correct
    LaTeX ($…$ inline, $$…$$ display). Fix transcription artifacts; do not add or
    remove meaning. Return the full content."""

    previous_context: str | None = dspy.InputField(desc="Preceding element (read-only context).")
    current_content: str = dspy.InputField(desc="The content to clean.")
    next_context: str | None = dspy.InputField(desc="Following element (read-only context).")
    content: str = dspy.OutputField(desc="The cleaned, well-formed content.")


class ExtractedItem(BaseModel):
    type: str  # an Element subtype label, e.g. "Paragraph", "Instruction", "Activity"
    content: str


class ExtractorSignature(dspy.Signature):
    """Consolidate one already-typed content item into the correct pedagogical
    element(s). Usually return it unchanged (one item). Split a shared instruction +
    exercise list into one Instruction followed by one Activity per exercise. Retype
    when the item is clearly an Admonition/Instruction/Activity."""

    previous_context: str | None = dspy.InputField(desc="Preceding element (read-only context).")
    current_content: str = dspy.InputField(desc="The item's content.")
    next_context: str | None = dspy.InputField(desc="Following element (read-only context).")
    items: list[ExtractedItem] = dspy.OutputField(
        desc="One or more typed items in reading order that replace the input."
    )


class SeamMergerSignature(dspy.Signature):
    """Two adjacent elements straddle a page boundary. Decide whether the right
    element is a direct continuation of the left (a split sentence/table/equation).
    If so, merge them into one coherent content string."""

    left_content: str = dspy.InputField(desc="End of the earlier page's element.")
    right_content: str = dspy.InputField(desc="Start of the later page's element.")
    merged: bool = dspy.OutputField(desc="True if right continues left.")
    merged_content: str = dspy.OutputField(desc="The merged content (if merged).")


class RefinerSignature(dspy.Signature):
    """Refine a single element of the given type (Code/Activity/Instruction/
    Admonition): normalize structure and formatting without changing meaning.
    Return the full refined content."""

    element_type: str = dspy.InputField(desc="The element's concrete type.")
    current_content: str = dspy.InputField(desc="The content to refine.")
    previous_context: str | None = dspy.InputField(desc="Preceding element (context).")
    next_context: str | None = dspy.InputField(desc="Following element (context).")
    content: str = dspy.OutputField(desc="The refined content.")
