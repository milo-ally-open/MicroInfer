"""汇总六份模型级 JSON，生成便于筛选的 CSV 与 XLSX 表。"""

import csv
import json
import math
import zipfile
from pathlib import Path
from xml.etree import ElementTree as et

root = Path(__file__).resolve().parents[2]
model_dir = root / "artifacts/qwen3_vl_2b"
output_dir = root / "analysis/qwen3_vl_2b"
columns = [
    "model_name", "model_alias", "repo_id", "revision", "implementation", "task", "record_type", "status",
    "device", "gpu_name", "torch_version", "cuda_version", "dtype", "input_type", "prompt_tokens", "max_new_tokens",
    "metric_name", "metric_value", "operator", "input_shapes_json", "call_count", "self_device_time_total_us",
    "self_cpu_time_total_us", "self_device_memory_usage_bytes", "self_cpu_memory_usage_bytes", "trace_path", "recorded_at_utc",
]

print("================= analysis =================")
reports = []
for implementation in ("baseline", "enhanced"):
    for task in ("test", "bench", "prefill"):
        path = model_dir / implementation / f"{task}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("schema_version") != 1 or report.get("implementation") != implementation or report.get("task") != task:
            raise ValueError(f"JSON schema 或路径标识不匹配：{path}")
        if report.get("result", {}).get("status") != "passed":
            raise ValueError(f"不能汇总未通过的测试记录：{path}")
        if task == "prefill" and not report.get("operator_records"):
            raise ValueError(f"Prefill JSON 没有算子记录：{path}")
        reports.append(report)
model = reports[0]["model"]
model_identity = tuple(model.get(key) for key in ("name", "alias", "repo_id", "revision"))
if any(tuple(report["model"].get(key) for key in ("name", "alias", "repo_id", "revision")) != model_identity for report in reports):
    raise ValueError("六份 JSON 的模型身份或 revision 不一致。")
rows = []
for report in reports:
    model_info = report["model"]
    env = report.get("environment", {})
    workload = report.get("workload", {})
    result = report.get("result", {})
    common = {
        "model_name": model_info["name"], "model_alias": model_info["alias"], "repo_id": model_info["repo_id"], "revision": model_info["revision"],
        "implementation": report["implementation"], "task": report["task"], "status": result.get("status"),
        "device": env.get("device"), "gpu_name": env.get("gpu_name"), "torch_version": env.get("torch_version"),
        "cuda_version": env.get("cuda_version"), "dtype": env.get("dtype"), "input_type": workload.get("input_type"),
        "prompt_tokens": workload.get("prompt_tokens"), "max_new_tokens": workload.get("max_new_tokens"),
        "trace_path": report.get("trace_path"), "recorded_at_utc": report.get("recorded_at_utc"),
    }
    rows.append({**common, "record_type": "summary"})
    for metric, value in report.get("metrics", {}).items():
        rows.append({**common, "record_type": "metric", "metric_name": metric, "metric_value": json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value})
    for operator in report.get("operator_records", []):
        rows.append({
            **common, "record_type": "operator", "metric_name": operator.get("phase"),
            "operator": operator.get("operator"),
            "input_shapes_json": json.dumps(operator.get("input_shapes"), ensure_ascii=False, separators=(",", ":")),
            "call_count": operator.get("call_count"),
            "self_device_time_total_us": operator.get("self_device_time_total_us"),
            "self_cpu_time_total_us": operator.get("self_cpu_time_total_us"),
            "self_device_memory_usage_bytes": operator.get("self_device_memory_usage_bytes"),
            "self_cpu_memory_usage_bytes": operator.get("self_cpu_memory_usage_bytes"),
        })

output_dir.mkdir(parents=True, exist_ok=True)
csv_path = output_dir / "analysis.csv"
with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
et.register_namespace("", main_ns)
et.register_namespace("r", rel_ns)
tag = lambda name: f"{{{main_ns}}}{name}"

def column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name

sheet = et.Element(tag("worksheet"))
last_cell = f"{column_name(len(columns))}{len(rows) + 1}"
et.SubElement(sheet, tag("dimension"), {"ref": f"A1:{last_cell}"})
views = et.SubElement(sheet, tag("sheetViews"))
view = et.SubElement(views, tag("sheetView"), {"showGridLines": "0", "workbookViewId": "0"})
et.SubElement(view, tag("pane"), {"ySplit": "1", "topLeftCell": "A2", "activePane": "bottomLeft", "state": "frozen"})
data = et.SubElement(sheet, tag("sheetData"))
for row_index, row in enumerate([{key: key for key in columns}, *rows], start=1):
    xml_row = et.SubElement(data, tag("row"), {"r": str(row_index)})
    for col_index, key in enumerate(columns, start=1):
        value = row.get(key)
        if value is None:
            continue
        cell = et.SubElement(xml_row, tag("c"), {"r": f"{column_name(col_index)}{row_index}"})
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            cell.set("t", "n")
            et.SubElement(cell, tag("v")).text = str(value)
        else:
            cell.set("t", "inlineStr")
            et.SubElement(et.SubElement(cell, tag("is")), tag("t")).text = str(value)
et.SubElement(sheet, tag("autoFilter"), {"ref": f"A1:{last_cell}"})
xlsx_path = output_dir / "analysis.xlsx"
with zipfile.ZipFile(xlsx_path, "w", zipfile.ZIP_DEFLATED) as workbook:
    workbook.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>')
    workbook.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
    workbook.writestr("xl/workbook.xml", f'<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="{main_ns}" xmlns:r="{rel_ns}"><sheets><sheet name="records" sheetId="1" r:id="rId1"/></sheets></workbook>')
    workbook.writestr("xl/_rels/workbook.xml.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>')
    workbook.writestr("xl/worksheets/sheet1.xml", et.tostring(sheet, encoding="utf-8", xml_declaration=True))
print(f"json_inputs: {len(reports)}")
print(f"records: {len(rows)}")
print("csv:", csv_path)
print("xlsx:", xlsx_path)
