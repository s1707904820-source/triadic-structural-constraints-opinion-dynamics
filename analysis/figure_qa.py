"""Small Matplotlib helpers for panel labels and measured layout manifests."""

from __future__ import annotations

import json
from pathlib import Path


def add_panel_label(ax, label):
    return ax.text(
        -0.15, 1.10, label, transform=ax.transAxes,
        fontsize=8, fontweight="bold", ha="left", va="bottom",
        clip_on=False,
    )


def write_alignment_manifest(
        fig, axes, labels, output_path, row_groups, column_groups,
        grid_cells=None):
    """Save final axes rectangles and label anchors in physical points."""
    fig.canvas.draw()
    width_in, height_in = (float(value) for value in fig.get_size_inches())
    width_pt, height_pt = width_in * 72.0, height_in * 72.0
    panels = []
    for ax, label in zip(axes, labels):
        position = ax.get_position(original=False)
        row_start, row_stop, col_start, col_stop = (
            grid_cells[label] if grid_cells is not None
            else (0, 1, len(panels), len(panels) + 1))
        panel = {
            "id": label,
            "bbox_pt": [
                float(position.x0 * width_pt),
                float(position.y0 * height_pt),
                float(position.x1 * width_pt),
                float(position.y1 * height_pt),
            ],
            "grid_id": "analysis-grid",
            "panel_label": label,
            "row_start": row_start,
            "row_stop": row_stop,
            "col_start": col_start,
            "col_stop": col_stop,
        }
        label_artist = next(
            (artist for artist in ax.texts
             if artist.get_text().strip() == label), None)
        if label_artist is not None:
            display = label_artist.get_transform().transform(
                label_artist.get_position())
            inches = fig.dpi_scale_trans.inverted().transform(display)
            panel["panel_label_anchor_pt"] = [
                float(inches[0] * 72.0), float(inches[1] * 72.0)]
        panels.append(panel)
    tolerance_pt = 1.5
    by_id = {panel["id"]: panel for panel in panels}
    findings = []

    def spread(values):
        return max(values) - min(values) if values else 0.0

    for index, group in enumerate(row_groups):
        members = [by_id[label] for label in group]
        boxes = [panel["bbox_pt"] for panel in members]
        top_spread = spread([box[3] for box in boxes])
        bottom_spread = spread([box[1] for box in boxes])
        height_spread = spread([box[3] - box[1] for box in boxes])
        widths = [box[2] - box[0] for box in boxes]
        if max(top_spread, bottom_spread, height_spread, spread(widths)) > tolerance_pt:
            findings.append({
                "group": "row-{}".format(index + 1),
                "top_spread_pt": top_spread,
                "bottom_spread_pt": bottom_spread,
                "height_spread_pt": height_spread,
                "width_spread_pt": spread(widths),
            })
        gaps = [boxes[j + 1][0] - boxes[j][2]
                for j in range(len(boxes) - 1)]
        if spread(gaps) > tolerance_pt:
            findings.append({
                "group": "row-{}".format(index + 1),
                "horizontal_gutter_spread_pt": spread(gaps),
            })
        anchors = [panel.get("panel_label_anchor_pt") for panel in members]
        if all(anchors) and spread([anchor[1] for anchor in anchors]) > tolerance_pt:
            findings.append({
                "group": "row-{}".format(index + 1),
                "panel_label_y_spread_pt": spread(
                    [anchor[1] for anchor in anchors]),
            })

    for index, group in enumerate(column_groups):
        members = [by_id[label] for label in group]
        boxes = [panel["bbox_pt"] for panel in members]
        left_spread = spread([box[0] for box in boxes])
        right_spread = spread([box[2] for box in boxes])
        width_spread = spread([box[2] - box[0] for box in boxes])
        heights = [box[3] - box[1] for box in boxes]
        if max(left_spread, right_spread, width_spread, spread(heights)) > tolerance_pt:
            findings.append({
                "group": "column-{}".format(index + 1),
                "left_spread_pt": left_spread,
                "right_spread_pt": right_spread,
                "width_spread_pt": width_spread,
                "height_spread_pt": spread(heights),
            })
        gaps = [boxes[j][1] - boxes[j + 1][3]
                for j in range(len(boxes) - 1)]
        if spread(gaps) > tolerance_pt:
            findings.append({
                "group": "column-{}".format(index + 1),
                "vertical_gutter_spread_pt": spread(gaps),
            })
        anchors = [panel.get("panel_label_anchor_pt") for panel in members]
        if all(anchors) and spread([anchor[0] for anchor in anchors]) > tolerance_pt:
            findings.append({
                "group": "column-{}".format(index + 1),
                "panel_label_x_spread_pt": spread(
                    [anchor[0] for anchor in anchors]),
            })

    for left_index, left in enumerate(panels):
        a = left["bbox_pt"]
        for right in panels[left_index + 1:]:
            b = right["bbox_pt"]
            overlap_w = min(a[2], b[2]) - max(a[0], b[0])
            overlap_h = min(a[3], b[3]) - max(a[1], b[1])
            if overlap_w > 0 and overlap_h > 0:
                findings.append({
                    "group": "panel-overlap",
                    "panels": [left["id"], right["id"]],
                    "overlap_pt": [overlap_w, overlap_h],
                })

    payload = {
        "schema_version": 1,
        "backend": "python-matplotlib",
        "figure": {"width_pt": width_pt, "height_pt": height_pt},
        "panels": panels,
        "row_groups": [
            {"id": "row-{}".format(index + 1), "panels": list(group)}
            for index, group in enumerate(row_groups)],
        "column_groups": [
            {"id": "column-{}".format(index + 1), "panels": list(group)}
            for index, group in enumerate(column_groups)],
        "tolerance_pt": tolerance_pt,
        "verdict": "PASS" if not findings else "FIX BEFORE DELIVERY",
        "findings": findings,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if findings:
        raise RuntimeError(
            "Panel alignment failed: {}".format(json.dumps(findings)))
    return payload



def require_matplotlib_panel_alignment(
        fig, axes, labels, output_path, row_groups, column_groups,
        grid_cells=None):
    """Measure and block misaligned rendered panels before figure export.

    This Python 3.7-compatible project helper preserves the measured physical
    geometry in the same manifest consumed by the standalone alignment auditor.
    """
    return write_alignment_manifest(
        fig, axes, labels, output_path, row_groups, column_groups, grid_cells)

def audit_pdf_text_geometry(pdf_path, report_path=None, minimum_font_pt=5.0):
    """Check PDF text size, page clipping, and inter-line text collisions."""
    import fitz

    pdf_path = Path(pdf_path)
    document = fitz.open(str(pdf_path))
    errors = []
    warnings = []
    checked_lines = []
    for page_index, page in enumerate(document, 1):
        page_rect = page.rect
        lines = []
        payload = page.get_text("dict")
        for block in payload.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = [span for span in line.get("spans", [])
                         if span.get("text", "").strip()]
                if not spans:
                    continue
                bbox = [
                    min(span["bbox"][0] for span in spans),
                    min(span["bbox"][1] for span in spans),
                    max(span["bbox"][2] for span in spans),
                    max(span["bbox"][3] for span in spans),
                ]
                label = "".join(span.get("text", "") for span in spans).strip()
                sizes = [float(span.get("size", 0.0)) for span in spans]
                smallest = min(sizes) if sizes else 0.0
                if smallest + 1e-6 < minimum_font_pt:
                    errors.append({
                        "kind": "font-size",
                        "page": page_index,
                        "text": label,
                        "minimum_pt": smallest,
                    })
                if (bbox[0] < -0.5 or bbox[1] < -0.5
                        or bbox[2] > page_rect.width + 0.5
                        or bbox[3] > page_rect.height + 0.5):
                    errors.append({
                        "kind": "page-clipping",
                        "page": page_index,
                        "text": label,
                        "bbox": bbox,
                    })
                lines.append((bbox, label))
                checked_lines.append({
                    "page": page_index, "text": label,
                    "minimum_font_pt": smallest,
                })
        for left_index, (left, left_text) in enumerate(lines):
            for right, right_text in lines[left_index + 1:]:
                overlap_w = min(left[2], right[2]) - max(left[0], right[0])
                overlap_h = min(left[3], right[3]) - max(left[1], right[1])
                if overlap_w <= 0 or overlap_h <= 0:
                    continue
                overlap = overlap_w * overlap_h
                left_area = max((left[2] - left[0]) * (left[3] - left[1]), 1.0)
                right_area = max((right[2] - right[0]) * (right[3] - right[1]), 1.0)
                if overlap > 1.0 and overlap / min(left_area, right_area) > 0.08:
                    errors.append({
                        "kind": "text-text-overlap",
                        "page": page_index,
                        "texts": [left_text, right_text],
                        "overlap_pt2": overlap,
                    })
    document.close()
    report = {
        "pdf": str(pdf_path.name),
        "minimum_font_pt": float(minimum_font_pt),
        "text_lines_checked": len(checked_lines),
        "verdict": "PASS" if not errors else "FIX BEFORE DELIVERY",
        "errors": errors,
        "warnings": warnings,
        "text_lines": checked_lines,
    }
    if report_path is not None:
        Path(report_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError(
            "PDF geometry audit failed: {}".format(json.dumps(errors[:20])))
    return report
