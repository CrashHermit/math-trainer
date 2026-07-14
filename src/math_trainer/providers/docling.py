"""Docling extraction provider.

Responsibilities:
  * ``convert``    — run Docling (remote-VLM by default) on a PDF/image → DoclingDocument.
  * ``normalize``  — flatten a DoclingDocument into backend-agnostic NormalizedItem records
                     (reading order preserved), plus per-page rasters. All Docling-specific
                     attribute access lives here.
  * ``materialize``— write Source → Segment(page) → Element into Neo4j. Pure graph logic,
                     independent of Docling, so it is unit-testable with plain records.

Design: docs/design/material_extraction_architecture.md §3.1, §4.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from math_trainer.core.config import DoclingConfig
from math_trainer.core.model.types import EdgeType, NodeType, docling_label_to_type
from math_trainer.storage.neo4j.repository import GraphRepository


@dataclass
class NormalizedItem:
    """A single Docling item, flattened to a backend-agnostic record."""

    page_no: int
    node_type: NodeType
    content: str | None = None
    image: Any | None = None          # PIL.Image.Image | None (for pictures)
    blurb: str | None = None          # Docling picture-description
    bbox: dict | None = None          # {l, t, r, b, coord_origin} | None


@dataclass
class NormalizedDocument:
    items: list[NormalizedItem] = field(default_factory=list)
    page_images: dict[int, Any] = field(default_factory=dict)   # page_no -> PIL.Image | None


class DoclingProvider:
    def __init__(self, config: DoclingConfig, repo: GraphRepository) -> None:
        self._config = config
        self._repo = repo

    # ── public API ─────────────────────────────────────────────────────────
    async def create_document(
        self,
        source: str | Path,
        *,
        document_title: str | None = None,
        source_uuid: str | None = None,
    ) -> str:
        """Extract `source` and materialize the content graph. Returns the Source uuid.

        Idempotent: if a Source with `source_uuid` already exists, it is reused and
        extraction is skipped (stage 1 is the only non-idempotent stage otherwise).
        """
        source = Path(source)
        if source_uuid is not None:
            existing = await self._repo.get_node(source_uuid, NodeType.SOURCE)
            if existing is not None:
                return source_uuid

        document = self.convert(source)
        normalized = self.normalize(document)
        return await self.materialize(
            normalized,
            source_path=str(source.resolve()),
            document_title=document_title or source.stem,
            source_uuid=source_uuid,
        )

    # ── Docling coupling (deferred to real models/VLM) ──────────────────────
    def convert(self, source: Path) -> Any:
        """Run Docling on the source and return a DoclingDocument.

        Docling is imported lazily so the rest of the module (and tests of
        ``materialize``) do not require docling/torch to be installed.
        """
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, PdfFormatOption

        converter = DocumentConverter(
            format_options=self._format_options(InputFormat, PdfFormatOption)
        )
        result = converter.convert(source=str(source))
        return result.document

    def _format_options(self, InputFormat: Any, PdfFormatOption: Any) -> dict:
        """Build per-format Docling options for the configured mode.

        remote_vlm : full-page vision-LM transcription via a remote API (default).
        local_vlm  : Docling's local VLM pipeline (GPU).
        standard   : classical layout pipeline + formula enrichment.
        Picture-description ("blurbs") is enabled via a remote vision API in every mode.
        """
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            PictureDescriptionApiOptions,
        )

        picdesc = None
        if self._config.picture_description_api_base:
            picdesc = PictureDescriptionApiOptions(
                url=self._config.picture_description_api_base,
                params={"model": self._config.picture_description_model},
                headers=self._api_headers(self._config.picture_description_api_key),
                prompt="Describe this figure from a technical/math textbook in one or two sentences.",
            )

        if self._config.mode == "remote_vlm":
            from docling.datamodel.pipeline_options import VlmPipelineOptions
            from docling.datamodel.pipeline_options_vlm_model import (
                ApiVlmOptions,
                ResponseFormat,
            )
            from docling.pipeline.vlm_pipeline import VlmPipeline

            vlm_opts = VlmPipelineOptions(enable_remote_services=True)
            vlm_opts.vlm_options = ApiVlmOptions(
                url=self._config.vlm_api_base,
                params={"model": self._config.vlm_model},
                headers=self._api_headers(self._config.vlm_api_key),
                prompt="Transcribe this page to clean Markdown. Use $...$/$$...$$ LaTeX for all math.",
                response_format=ResponseFormat.MARKDOWN,
                scale=self._config.image_scale,
            )
            vlm_opts.generate_page_images = True
            vlm_opts.generate_picture_images = True
            if picdesc is not None:
                vlm_opts.do_picture_description = True
                vlm_opts.picture_description_options = picdesc
            pdf_opt = PdfFormatOption(pipeline_cls=VlmPipeline, pipeline_options=vlm_opts)
        else:
            opts = PdfPipelineOptions()
            opts.images_scale = self._config.image_scale
            opts.generate_page_images = True
            opts.generate_picture_images = True
            opts.do_formula_enrichment = self._config.mode == "standard"
            if picdesc is not None:
                opts.do_picture_description = True
                opts.picture_description_options = picdesc
            pdf_opt = PdfFormatOption(pipeline_options=opts)

        return {InputFormat.PDF: pdf_opt, InputFormat.IMAGE: pdf_opt}

    @staticmethod
    def _api_headers(api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def normalize(self, document: Any) -> NormalizedDocument:
        """Flatten a DoclingDocument into NormalizedItem records in reading order."""
        out = NormalizedDocument()

        pages = getattr(document, "pages", {}) or {}
        for page_no, page in pages.items():
            out.page_images[int(page_no)] = self._page_image(page)

        for item, _level in document.iterate_items():
            page_no = self._item_page(item)
            label = self._item_label(item)
            node_type = docling_label_to_type(label)
            if node_type is NodeType.IMAGE:
                out.items.append(
                    NormalizedItem(
                        page_no=page_no,
                        node_type=NodeType.IMAGE,
                        image=self._picture_image(item, document),
                        blurb=self._picture_blurb(item, document),
                        bbox=self._item_bbox(item),
                    )
                )
            else:
                content = self._item_text(item, document)
                if not content:
                    continue
                out.items.append(
                    NormalizedItem(
                        page_no=page_no,
                        node_type=node_type,
                        content=content,
                        bbox=self._item_bbox(item),
                    )
                )
        return out

    # ── graph materialization (pure; Docling-independent) ───────────────────
    async def materialize(
        self,
        document: NormalizedDocument,
        *,
        source_path: str,
        document_title: str,
        source_uuid: str | None = None,
    ) -> str:
        source_uuid = source_uuid or str(uuid4())
        source = await self._repo.create_node(
            [NodeType.SOURCE],
            uuid=source_uuid,
            title=document_title,
            source_path=source_path,
            status="extracting",
            stage=1,
        )
        out_root = Path(self._config.output_dir) / source_uuid

        element_uuids: list[str] = []
        order_index = 0
        prev_segment_uuid: str | None = None

        for page_no in sorted({it.page_no for it in document.items} | set(document.page_images)):
            seg_dir = out_root / f"segment_{page_no:04d}"
            page_raster = self._write_page_raster(
                document.page_images.get(page_no), seg_dir
            )
            segment = await self._repo.create_node(
                [NodeType.SEGMENT], segment_index=page_no, src=page_raster
            )
            await self._repo.link(EdgeType.CONTAINS, source_uuid, segment["uuid"])
            prev_segment_uuid = segment["uuid"]

            page_items = [it for it in document.items if it.page_no == page_no]
            segment_element_uuids: list[str] = []
            for local_idx, item in enumerate(page_items):
                order_index += 1
                el_uuid = await self._create_element(
                    item, source_uuid, order_index, seg_dir, local_idx
                )
                segment_element_uuids.append(el_uuid)
                element_uuids.append(el_uuid)
            await self._repo.link_children(
                segment["uuid"], segment_element_uuids, EdgeType.CONTAINS
            )

        # Reading-order chain across the whole source + entry edge to the head.
        await self._repo.link_chain(element_uuids, EdgeType.NEXT)
        if element_uuids:
            await self._repo.link(EdgeType.HAS, source_uuid, element_uuids[0])

        await self._repo.update_node(source_uuid, status="extracted")
        return source_uuid

    async def _create_element(
        self,
        item: NormalizedItem,
        source_uuid: str,
        order_index: int,
        seg_dir: Path,
        local_idx: int,
    ) -> str:
        if item.node_type is NodeType.IMAGE:
            img_path = self._write_picture(item.image, seg_dir, local_idx)
            node = await self._repo.create_node(
                [NodeType.ELEMENT, NodeType.IMAGE],
                source_uuid=source_uuid,
                order_index=order_index,
                src=img_path,
                blurb=item.blurb,
                page_no=item.page_no,
                bbox=self._bbox_str(item.bbox),
            )
        else:
            node = await self._repo.create_node(
                [NodeType.ELEMENT, item.node_type],
                source_uuid=source_uuid,
                order_index=order_index,
                content=item.content,
                page_no=item.page_no,
                bbox=self._bbox_str(item.bbox),
            )
        return node["uuid"]

    # ── file writing helpers ────────────────────────────────────────────────
    def _write_page_raster(self, image: Any | None, seg_dir: Path) -> str | None:
        if image is None:
            return None
        seg_dir.mkdir(parents=True, exist_ok=True)
        path = seg_dir / "page.png"
        self._save(image, path)
        return str(path)

    def _write_picture(self, image: Any | None, seg_dir: Path, idx: int) -> str | None:
        if image is None:
            return None
        pic_dir = seg_dir / "pictures"
        pic_dir.mkdir(parents=True, exist_ok=True)
        path = pic_dir / f"picture_{idx:03d}.png"
        self._save(image, path)
        return str(path)

    @staticmethod
    def _save(image: Any, path: Path) -> None:
        image.save(str(path))
        close = getattr(image, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _bbox_str(bbox: dict | None) -> str | None:
        if not bbox:
            return None
        return ",".join(
            str(bbox.get(k)) for k in ("l", "t", "r", "b") if bbox.get(k) is not None
        ) or None

    # ── Docling attribute extraction (guarded for testability) ──────────────
    @staticmethod
    def _item_label(item: Any) -> str:
        label = getattr(item, "label", "")
        return str(getattr(label, "value", label) or "")

    @staticmethod
    def _item_page(item: Any) -> int:
        prov = getattr(item, "prov", None) or []
        if prov:
            return int(getattr(prov[0], "page_no", 1) or 1)
        return 1

    @staticmethod
    def _item_bbox(item: Any) -> dict | None:
        prov = getattr(item, "prov", None) or []
        if not prov:
            return None
        bbox = getattr(prov[0], "bbox", None)
        if bbox is None:
            return None
        return {
            "l": getattr(bbox, "l", None),
            "t": getattr(bbox, "t", None),
            "r": getattr(bbox, "r", None),
            "b": getattr(bbox, "b", None),
        }

    def _item_text(self, item: Any, document: Any) -> str | None:
        # Tables know how to render themselves as markdown; text items carry .text.
        export = getattr(item, "export_to_markdown", None)
        if callable(export):
            try:
                return export(doc=document).strip() or None
            except TypeError:
                try:
                    return export().strip() or None
                except Exception:
                    pass
            except Exception:
                pass
        text = getattr(item, "text", None)
        return text.strip() if isinstance(text, str) and text.strip() else None

    @staticmethod
    def _picture_image(item: Any, document: Any) -> Any | None:
        getter = getattr(item, "get_image", None)
        if callable(getter):
            try:
                return getter(document)
            except Exception:
                return None
        return None

    @staticmethod
    def _picture_blurb(item: Any, document: Any) -> str | None:
        for ann in getattr(item, "annotations", None) or []:
            text = getattr(ann, "text", None)
            if isinstance(text, str) and text.strip():
                return text.strip()
        caption = getattr(item, "caption_text", None)
        if callable(caption):
            try:
                cap = caption(document)
                if isinstance(cap, str) and cap.strip():
                    return cap.strip()
            except Exception:
                pass
        return None
