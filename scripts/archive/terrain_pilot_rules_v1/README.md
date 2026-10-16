# 地形 pilot v1：手写规则版（已归档，2026-10-15）

unit = 手写规则直接合成的"发现"（如 Hemoglobin/Hematocrit/RBC 异常 → "血红蛋白低"），
同义、器官、问题三层 sameness 都由 `LAB_GROUPS` / `IMG_FINDINGS` / `MICRO` 三张表人工定义。
被 atomic unit 版本取代（见 `scripts/terrain_units.py`），保留作对照。

重跑（输出仍写到 `runs/terrain_pilot/`）：

    PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/archive/terrain_pilot_rules_v1/terrain_pilot.py
