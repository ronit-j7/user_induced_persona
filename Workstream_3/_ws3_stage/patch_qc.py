from pathlib import Path

root = Path("/home/harsha/eshaan/ANLP-proj/user_induced_persona")
data = root / "Workstream_2/attentionseekers/data.py"
text = data.read_text()
old = '''        fc = qc["forced_choice"]
        if not isinstance(fc, dict) or fc.get("rewrite_as_B") != "B" or fc.get("rewrite_as_A") != "A":
            raise ValueError(f"{name}: forced-choice QC must select the rewrite in both orders")
'''
new = '''        fc = qc["forced_choice"]
        # D4: a high pole passes when the rewrite wins both orders, a low pole when the original does.
        expected = {"+": ("B", "A"), "-": ("A", "B")}.get(row.get("user_pole"))
        if expected is None or not isinstance(fc, dict) or (fc.get("rewrite_as_B"), fc.get("rewrite_as_A")) != expected:
            raise ValueError(f"{name}: forced-choice QC must match the pole in both orders")
'''
if old not in text:
    raise SystemExit("qc block missing")
text = text.replace(old, new)
old2 = '''        return "ws1_checklist_forced_choice"
    for key in ("content", "trait"):
'''
new2 = '''        return "ws1_checklist_forced_choice"
    if row["set"] == "factorial" and isinstance(qc.get("user_variant_id"), str) and qc["user_variant_id"].strip():
        return "factorial_user_variant_link"
    for key in ("content", "trait"):
'''
if old2 not in text:
    raise SystemExit("return block missing")
data.write_text(text.replace(old2, new2))

test = root / "Workstream_2/tests/test_ws1_integration.py"
t = test.read_text()
old = '''    row = generator.make_row(scn, "E", "+", 0, accepted, lambda text: None)
    assert validate_rows([row])["qc_protocol_counts"] == {"ws1_checklist_forced_choice": 1}
'''
new = '''    row = generator.make_row(scn, "E", "+", 0, accepted, lambda text: None)
    assert validate_rows([row])["qc_protocol_counts"] == {"ws1_checklist_forced_choice": 1}
    low = {"text": accepted["text"], "attempts": 1,
           "eval": {"forced_choice": {"rewrite_as_B": "A", "rewrite_as_A": "B"}}}
    low_row = generator.make_row(scn, "E", "-", 0, low, lambda text: None)
    low_row["id"] = low_row["id"] + "-low"
    assert validate_rows([low_row])["qc_protocol_counts"] == {"ws1_checklist_forced_choice": 1}
'''
if old not in t:
    raise SystemExit("test block missing")
t = t.replace(old, new).replace('match="both orders"', 'match="match the pole"')
test.write_text(t)

doc = root / "INTERFACES.md"
d = doc.read_text()
needle = '`{"passed":true,"forced_choice":{"rewrite_as_B":"B","rewrite_as_A":"A"}}`'
insert = ('`{"passed":true,"forced_choice":{"rewrite_as_B":"B","rewrite_as_A":"A"}}` for a high pole, '
          'and the reverse pair (`A` then `B`) for a low pole (DECISIONS D4). '
          'Factorial rows may instead point at the source rewrite with `user_variant_id`')
if needle not in d:
    raise SystemExit("interfaces needle missing")
doc.write_text(d.replace(needle, insert, 1))
print("patched")
