#!/usr/bin/env python3
"""从 Qwen3-VL Transformers 基线 JSON 生成 analysis/qwen3_vl_2b/analysis.xlsx。"""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "artifacts/qwen3-vl-2b-transformers-forward.json"
BENCHMARK_PATH = ROOT / "artifacts/qwen3-vl-2b-transformers-benchmark.json"
OUTPUT_PATH = ROOT / "analysis/qwen3_vl_2b/analysis.xlsx"
COLUMNS = [
    "record_type", "model_alias", "revision", "gpu_name", "dtype", "implementation",
    "phase", "operator", "input_shapes_json", "call_count", "self_device_time_total_us",
    "self_cpu_time_total_us", "self_device_memory_usage_bytes", "self_cpu_memory_usage_bytes",
    "prompt_tokens", "decode_query_tokens", "prefill_cache_tokens", "decode_cache_tokens",
    "phase_latency_ms", "latency_seconds_median", "tokens_per_second_median",
    "peak_cuda_memory_bytes", "peak_cuda_reserved_bytes", "warmup_requests", "iterations",
]


def column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
for key in ("model_alias", "revision", "gpu_name", "dtype", "operator_records"):
    if key not in profile:
        raise ValueError(f"Profile JSON is missing {key!r}: {PROFILE_PATH}")
if not profile["operator_records"]:
    raise ValueError("Profile JSON contains no operator records.")
if profile["decode_query_tokens"] != 1:
    raise ValueError("Decode Profile must measure exactly one query token.")
if profile["decode_cache_tokens_after_forward"] != profile["prefill_cache_tokens"] + 1:
    raise ValueError("Decode Profile Cache length must increase by one token.")

latency_by_phase = {"prefill": profile["prefill_ms"], "decode": profile["decode_ms"]}
rows: list[list[object]] = []
for operator in profile["operator_records"]:
    rows.append([
        "operator", profile["model_alias"], profile["revision"], profile["gpu_name"],
        profile["dtype"], profile["implementation"], operator["phase"], operator["operator"],
        json.dumps(operator["input_shapes"], ensure_ascii=False, separators=(",", ":")),
        operator["call_count"], operator["self_device_time_total_us"], operator["self_cpu_time_total_us"],
        operator["self_device_memory_usage_bytes"], operator["self_cpu_memory_usage_bytes"],
        profile["prompt_tokens"], profile["decode_query_tokens"], profile["prefill_cache_tokens"],
        profile["decode_cache_tokens_after_forward"], latency_by_phase[operator["phase"]],
        None, None, profile["peak_cuda_memory_bytes"], None, None, None,
    ])
rows.append([
    "benchmark", benchmark["model_alias"], benchmark["revision"], benchmark["device"],
    benchmark["dtype"], benchmark["implementation"], "generate", None, None, None, None, None,
    None, None, benchmark["input_tokens"], None, None, None,
    None, benchmark["latency_seconds_median"], benchmark["tokens_per_second_median"],
    benchmark["peak_cuda_memory_bytes"], benchmark["peak_cuda_reserved_bytes"],
    benchmark["warmup_requests"], benchmark["iterations"],
])

main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
ET.register_namespace("", main_ns)
ET.register_namespace("r", rel_ns)
tag = lambda name: f"{{{main_ns}}}{name}"
sheet = ET.Element(tag("worksheet"))
last_cell = f"{column_name(len(COLUMNS))}{len(rows) + 1}"
ET.SubElement(sheet, tag("dimension"), {"ref": f"A1:{last_cell}"})
views = ET.SubElement(sheet, tag("sheetViews"))
view = ET.SubElement(views, tag("sheetView"), {"showGridLines": "0", "workbookViewId": "0"})
ET.SubElement(view, tag("pane"), {"ySplit": "1", "topLeftCell": "A2", "activePane": "bottomLeft", "state": "frozen"})
data = ET.SubElement(sheet, tag("sheetData"))
for row_index, values in enumerate([COLUMNS, *rows], start=1):
    row = ET.SubElement(data, tag("row"), {"r": str(row_index)})
    for column_index, value in enumerate(values, start=1):
        if value is None:
            continue
        cell = ET.SubElement(row, tag("c"), {"r": f"{column_name(column_index)}{row_index}"})
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(float(value)):
                cell.set("t", "n")
                ET.SubElement(cell, tag("v")).text = str(value)
        else:
            cell.set("t", "inlineStr")
            inline = ET.SubElement(cell, tag("is"))
            ET.SubElement(inline, tag("t")).text = str(value)
ET.SubElement(sheet, tag("autoFilter"), {"ref": f"A1:{last_cell}"})

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(OUTPUT_PATH, "w", zipfile.ZIP_DEFLATED) as workbook:
    workbook.writestr(
        "[Content_Types].xml",
        '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>',
    )
    workbook.writestr(
        "_rels/.rels",
        '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>',
    )
    workbook.writestr(
        "xl/workbook.xml",
        f'<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="{main_ns}" xmlns:r="{rel_ns}"><sheets><sheet name="records" sheetId="1" r:id="rId1"/></sheets></workbook>',
    )
    workbook.writestr(
        "xl/_rels/workbook.xml.rels",
        '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>',
    )
    workbook.writestr("xl/worksheets/sheet1.xml", ET.tostring(sheet, encoding="utf-8", xml_declaration=True))

print("operator_records:", len(profile["operator_records"]))
print("benchmark_records: 1")
print("workbook:", OUTPUT_PATH)
