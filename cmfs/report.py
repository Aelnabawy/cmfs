import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill

def to_excel(full, nested, path, notes):
    sheets = {"README": pd.DataFrame({"Notes": notes})}
    sheets["Nested classification"] = pd.DataFrame([dict(task_and_panel=k, AUC_mean=v[0], AUC_fold_p2_5=v[1], AUC_fold_p97_5=v[2]) for k, v in nested["auc"].items()])
    m = nested["mediated"]; sheets["Honest mediated effect"] = pd.DataFrame([dict(total_effect=m["total"], mediated=m["estimate"], ci_low=m["lo"], ci_high=m["hi"], share=m["share"], E_value=m["evalue"], E_value_CI=m["evalue_ci"])])
    sheets["Selection frequency"] = pd.DataFrame([dict(group=g, feature=f, selected_pct=c / nested["n_folds"]) for g, cnt in nested["frequency"].items() for f, c in cnt.most_common()] +
                                                 [dict(group=g, feature="(stability, Jaccard)", selected_pct=s) for g, s in nested["stability"].items()])
    sheets["Roles of selected features"] = full["roles"]
    for g, v in full["groups"].items():
        sheets[f"{g} - upstream scores"[:31]] = v["upstream"]; sheets[f"{g} - downstream scores"[:31]] = v["downstream"]
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for k, v in sheets.items(): v.to_excel(w, sheet_name=k[:31], index=False)
    wb = load_workbook(path)
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for c in row:
                c.font = Font(name="Arial", size=9, bold=c.row == 1, color="FFFFFF" if c.row == 1 else "000000")
                if c.row == 1: c.fill = PatternFill("solid", fgColor="1F4E78")
                if isinstance(c.value, float): c.number_format = "0.000"
        for col in ws.columns: ws.column_dimensions[col[0].column_letter].width = max(11, min(48, max(len(str(c.value or "")) for c in col) + 2))
    wb["README"].column_dimensions["A"].width = 170; wb.save(path)
