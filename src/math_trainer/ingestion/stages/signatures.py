"""DSPy Signatures for the intelligent stages.

The input/output field names match each stage's ``module.aforward(**kwargs)`` call
and the attributes each stage reads off the prediction.
"""

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
    parts: list[ExtractedItem] = dspy.OutputField(
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


class LinkDecisionSignature(dspy.Signature):
    """Decide whether a section-level Instruction from a textbook governs a specific
    Activity (exercise). The Instruction is a lead line introducing exercises; the
    Activity is a single exercise. Return should_link=true if the activity clearly
    falls under the instruction's scope."""

    instruction_text: str = dspy.InputField(desc="The full lead-instruction text.")
    activity_content: str = dspy.InputField(desc="The exercise content.")
    should_link: bool = dspy.OutputField(desc="True if the instruction governs this activity.")


class AssembledUnit(BaseModel):
    kind: str          # a Block kind, e.g. "prose", "definition", "theorem", "example"
    members: list[str] # uuids of the window blocks (contiguous) that form this unit
    label: str | None = None


class AssemblerSignature(dspy.Signature):
    """Segment a window of ungrouped content blocks into semantic units for a math
    textbook. Group a definition, a theorem (with its statement), a worked example,
    or a coherent prose+math cluster into one unit; keep a statement and its proof,
    or a problem and its solution, as SEPARATE units. Return the units that are
    fully contained in the window in order; put the uuids of a trailing unit that
    is clearly incomplete (continues past the window) into `deferred` so it can be
    completed with more context — do not guess its end."""

    window: list[dict] = dspy.InputField(
        desc="Ordered content blocks {uuid, type, content} to segment."
    )
    leading_context: str | None = dspy.InputField(
        desc="Short summary of the previously committed unit (read-only)."
    )
    trailing: list[dict] = dspy.InputField(
        desc="Upcoming blocks just past the window (read-only, for seeing boundaries)."
    )
    units: list[AssembledUnit] = dspy.OutputField(
        desc="Complete units in reading order; members are window uuids."
    )
    deferred: list[str] = dspy.OutputField(
        desc="Trailing window uuids forming an incomplete unit to carry forward."
    )
