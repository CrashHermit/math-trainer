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
    type: str   # one of the TYPE VOCABULARY below
    content: str


class ExtractorSignature(dspy.Signature):
    r"""Assign one already-extracted textbook item to the correct type, splitting it
    into several items only when it clearly contains more than one.

    TYPE VOCABULARY (use exactly these strings):
      Paragraph, Heading, Math, Table, Caption, List, ListItem, Code,
      Instruction, Activity, Admonition

    WHAT EACH MEANS (read carefully — most items are prose and stay `Paragraph`):
      • Paragraph  — ordinary exposition. **A Definition, Theorem, Lemma, Proposition,
                     Corollary, Proof, or Remark is PROSE → `Paragraph`.** These are
                     statements to read, NOT tasks. NEVER type them Instruction/Activity.
      • Heading    — a section/subsection title, e.g. "3.2 The Derivative", "Exercises",
                     "Problems". A heading is NEVER an Activity. A numbered
                     learning-objective/goal line (e.g. "3.3.1 State the power rule.")
                     is NOT a heading — it is a `ListItem`.
      • Instruction— ONLY a lead line that governs a group of exercises, e.g.
                     "1–20 Differentiate each function." It introduces tasks; it is not
                     itself a task, a theorem, or a proof.
      • Activity   — ONLY a standalone practice problem for the student to solve
                     (typically numbered, in an Exercises/Problems set or a "Checkpoint").
                     A proof, a theorem, or a heading is NEVER an Activity. The problem
                     line INSIDE a worked Example (the "Find …" right after an
                     "Example N" title) is NOT an Activity — it is part of the example,
                     so leave it as `Paragraph`.
      • "Solution" — the word "Solution" and the steps under a worked Example are part
                     of that example → `Paragraph` (never a Heading).
      • Admonition — a boxed callout (Example, Note, Tip, Warning) kept whole.
      • Math/Table/Caption/List/ListItem/Code — as named.

    SPLITTING RULES:
      • Default: return the item UNCHANGED as a single part with its correct type.
      • If one item bundles a lead instruction AND its first exercise (e.g.
        "1–3 Differentiate. 1. f(x)=x^2"), split into an Instruction part followed by
        one Activity part.
      • A numbered exercise list bundled in one item → one Activity per exercise.
      • Use previous_context/next_context to retype a bare item: e.g. a lone
        "f(x)=e^x" that follows other exercises under an exercises heading is an Activity.
      • Never merge two distinct exercises into one part.

    EXAMPLES:
      current="**Theorem 3.4 (Power Rule).** If f(x)=x^n then f'(x)=nx^{n-1}."
        → parts=[{type:"Paragraph", content:"**Theorem 3.4 (Power Rule).** …"}]
      current="Proof. Expand (x+h)^n and cancel."   (a Theorem precedes it)
        → parts=[{type:"Paragraph", content:"Proof. Expand (x+h)^n and cancel."}]
      current="Exercises"
        → parts=[{type:"Heading", content:"Exercises"}]
      current="1–3 Differentiate each function. 1. f(x)=x^5"
        → parts=[{type:"Instruction", content:"1–3 Differentiate each function."},
                 {type:"Activity", content:"1. f(x)=x^5"}]
      current="f(x)=\\sin x"   (prev is exercise 1, under an Exercises heading)
        → parts=[{type:"Activity", content:"2. f(x)=\\sin x"}]
    """

    previous_context: str | None = dspy.InputField(desc="Preceding element (read-only context).")
    current_content: str = dspy.InputField(desc="The item's content.")
    next_context: str | None = dspy.InputField(desc="Following element (read-only context).")
    parts: list[ExtractedItem] = dspy.OutputField(
        desc="One or more typed items in reading order that replace the input (usually one)."
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
    kind: str          # a KIND from the vocabulary below
    members: list[str] # uuids of the window elements (contiguous) that form this unit
    label: str | None = None


class AssemblerSignature(dspy.Signature):
    r"""Group a window of consecutive content elements into semantic units for a math
    textbook. Each unit is the smallest self-contained thing a learner would study.

    KIND VOCABULARY (choose the most specific; use "prose" only for plain exposition):
      definition, theorem, example, remark, proof, prose, figure, table, code, heading

    RULES — be FAITHFUL, do not invent units:
      • Classify each element by the content of THAT element. A unit's kind must
        describe what its member elements actually say. If an element's text is a
        proof, its unit is "proof"; if it is an example, "example"; and so on.
      • NEVER produce more units than elements, and never map an element onto a unit
        whose kind does not match that element's own text (no shifting).
      • If a SINGLE element already mixes several things (e.g. an intro sentence,
        a definition, and a theorem all in one text block), emit ONE unit for that
        element — label it by its dominant kind. Do NOT fabricate extra units for the
        parts you cannot separate.
      • MERGE adjacent elements into one unit when they form one thing: a
        sentence/statement and its own display equation, or a block split mid-way.
      • A worked EXAMPLE is ONE "example" unit spanning ALL its elements — the
        "Example N" title, its problem statement, the word "Solution", every solution
        step, and any figure — up to the next heading / example / exercise. Do NOT
        split an example's statement, "Solution", or steps into separate units.
      • Separate a theorem/definition STATEMENT from its PROOF when they are in
        separate elements (a learner recalls the statement and proves it separately).
      • Group a run of numbered learning objectives / goals into ONE "prose" unit —
        do not alternate heading/prose across the items.
      • Put an identifier in `label` when present (e.g. "Definition 3.1", "Power Rule").
      • Members are a contiguous run of window uuids; cover every window uuid exactly
        once, except a trailing incomplete unit → put its uuids in `deferred`.

    EXAMPLE — window elements (by their own content):
      u1="3.2 The Derivative"            → {kind:"heading", members:["u1"]}
      u2="Definition 3.1. … $f'(a)=…$"   → {kind:"definition", label:"Definition 3.1", members:["u2"]}
      u3="Proof. Expand (x+h)^n …"       → {kind:"proof", members:["u3"]}
      u4="Example 3.6. Differentiate …"  → {kind:"example", label:"Example 3.6", members:["u4"]}
      → units in that order, deferred=[]
    """

    window: list[dict] = dspy.InputField(
        desc="Ordered content elements {uuid, type, content} to group."
    )
    leading_context: str | None = dspy.InputField(
        desc="Short summary of the previously committed unit (read-only)."
    )
    trailing: list[dict] = dspy.InputField(
        desc="Upcoming elements just past the window (read-only, for seeing boundaries)."
    )
    units: list[AssembledUnit] = dspy.OutputField(
        desc="Complete units in reading order; members are window uuids."
    )
    deferred: list[str] = dspy.OutputField(
        desc="Trailing window uuids forming an incomplete unit to carry forward."
    )
