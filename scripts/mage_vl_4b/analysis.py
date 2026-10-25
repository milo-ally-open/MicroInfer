#!/usr/bin/env python3
"""将 Mage-VL forward.py 的原始算子 JSON 转成单表 XLSX。"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = [
    "record_id", "timestamp_utc", "model_alias", "revision", "gpu_name", "torch_version", "cuda_version",
    "dtype", "implementation", "attention_implementation", "batch_size", "query_tokens", "kv_tokens", "block_size_tokens",
    "warmup_iterations", "measured_iterations", "num_hidden_layers", "input_type", "phase", "operator", "input_shapes_json", "call_count",
    "self_cuda_time_total_us", "self_cpu_time_total_us", "self_cuda_memory_usage_bytes",
    "self_cpu_memory_usage_bytes", "cuda_event_ms", "peak_cuda_memory_bytes", "peak_cuda_reserved_bytes",
]


def write_xlsx(path: Path, rows: list[list[object]]) -> None:
    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    ET.register_namespace("", main_ns)
    ET.register_namespace("r", rel_ns)

    def tag(name: str) -> str:
        return f"{{{main_ns}}}{name}"

    def col_name(index: int) -> str:
        result = ""
        while index:
            index, remainder = divmod(index - 1, 26)
            result = chr(65 + remainder) + result
        return result

    values = [COLUMNS, *rows]
    end_cell = f"{col_name(len(COLUMNS))}{len(values)}"
    sheet = ET.Element(tag("worksheet"))
    ET.SubElement(sheet, tag("dimension"), {"ref": f"A1:{end_cell}"})
    views = ET.SubElement(sheet, tag("sheetViews"))
    view = ET.SubElement(views, tag("sheetView"), {"showGridLines": "0", "workbookViewId": "0"})
    ET.SubElement(view, tag("pane"), {"ySplit": "1", "topLeftCell": "A2", "activePane": "bottomLeft", "state": "frozen"})
    data = ET.SubElement(sheet, tag("sheetData"))
    for row_index, row_values in enumerate(values, start=1):
        row = ET.SubElement(data, tag("row"), {"r": str(row_index)})
        for column_index, value in enumerate(row_values, start=1):
            if value is None:
                continue
            cell = ET.SubElement(row, tag("c"), {"r": f"{col_name(column_index)}{row_index}"})
            if isinstance(value, bool):
                cell.set("t", "b")
                ET.SubElement(cell, tag("v")).text = str(int(value))
            elif isinstance(value, (int, float)):
                if not math.isfinite(float(value)):
                    continue
                cell.set("t", "n")
                ET.SubElement(cell, tag("v")).text = str(value)
            else:
                cell.set("t", "inlineStr")
                inline = ET.SubElement(cell, tag("is"))
                ET.SubElement(inline, tag("t")).text = str(value)
    ET.SubElement(sheet, tag("autoFilter"), {"ref": f"A1:{end_cell}"})
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>')
        workbook.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        workbook.writestr("xl/workbook.xml", f'<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="{main_ns}" xmlns:r="{rel_ns}"><sheets><sheet name="records" sheetId="1" r:id="rId1"/></sheets></workbook>')
        workbook.writestr("xl/_rels/workbook.xml.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>')
        workbook.writestr("xl/worksheets/sheet1.xml", ET.tostring(sheet, encoding="utf-8", xml_declaration=True))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, required=True, help="scripts/mage_vl_4b/forward.py 输出的 JSON。")
parser.add_argument("--output", type=Path, default=ROOT / "analysis/mage_vl_4b/analysis.xlsx")
args = parser.parse_args()
report = json.loads(args.input.expanduser().resolve().read_text(encoding="utf-8"))
run = report.get("run")
operator_records = report.get("operator_records")
if not isinstance(run, dict) or not isinstance(operator_records, list) or not operator_records:
    raise ValueError("输入 JSON 必须包含 run 对象和非空 operator_records 数组。")
implementations = report.get("implementations", {})
if not isinstance(implementations, dict):
    raise ValueError("implementations 必须是 mapping。")
rows = []
for index, record in enumerate(operator_records, start=1):
    if not isinstance(record, dict):
        raise ValueError(f"operator_records[{index - 1}] 不是对象。")
    implementation_name = str(record.get("implementation", "transformers"))
    implementation = implementations.get(implementation_name, run)
    if not isinstance(implementation, dict):
        raise ValueError(f"implementations[{implementation_name!r}] 不是对象。")
    phase_ms = implementation.get(f"{record.get('phase')}_cuda_event_ms", run.get(f"{record.get('phase')}_cuda_event_ms"))
    rows.append([
        f"MV-{index:04d}", report.get("timestamp_utc"), run.get("model_alias"), run.get("revision"),
        run.get("gpu_name"), run.get("torch_version"), run.get("cuda_version"), run.get("dtype"),
        implementation_name, implementation.get("attention_implementation", run.get("attention_implementation")),
        run.get("batch_size"), run.get("query_tokens"), run.get("kv_tokens"),
        implementation.get("block_size_tokens", run.get("block_size_tokens")), run.get("warmup_iterations"), run.get("measured_iterations"),
        run.get("num_hidden_layers"), run.get("input_type"), record.get("phase"),
        record.get("operator"), json.dumps(record.get("input_shapes"), ensure_ascii=False, separators=(",", ":")),
        record.get("call_count"), record.get("self_cuda_time_total_us"), record.get("self_cpu_time_total_us"),
        record.get("self_cuda_memory_usage_bytes"), record.get("self_cpu_memory_usage_bytes"), phase_ms,
        implementation.get("peak_cuda_memory_bytes", run.get("peak_cuda_memory_bytes")),
        implementation.get("peak_cuda_reserved_bytes", run.get("peak_cuda_reserved_bytes")),
    ])
output = args.output.expanduser().resolve()
write_xlsx(output, rows)
print(f"operator_records: {len(rows)}")
print(f"workbook: {output}")
