"""CDR² station metadata and role assignment for the paper data split."""
from __future__ import annotations

from pathlib import Path
import struct
from typing import Iterable

import pandas as pd


def read_dbf(path: str | Path) -> pd.DataFrame:
    """Read the small CDR² DBF metadata table without optional GIS packages."""
    raw = Path(path).read_bytes()
    n = struct.unpack_from("<I", raw, 4)[0]
    header_len = struct.unpack_from("<H", raw, 8)[0]
    record_len = struct.unpack_from("<H", raw, 10)[0]
    fields = []
    for pos in range(32, header_len - 1, 32):
        name = raw[pos:pos + 11].decode("ascii", "ignore").replace("\x00", "").strip()
        length = raw[pos + 16]
        fields.append((name, length))
    rows = []
    for i in range(n):
        record = raw[header_len + i * record_len:header_len + (i + 1) * record_len]
        offset = 1
        row = {}
        for name, length in fields:
            row[name] = record[offset:offset + length].decode("utf-8", "ignore").replace("\x00", "").strip()
            offset += length
        rows.append(row)
    return pd.DataFrame(rows)


STATION_ROLES = {
    60101300: ("Wudongde_upstream_reference", True, False),
    60102100: ("Wudongde_Baihetan_interdam_reference", True, False),
    60104100: ("Xiluodu_Xiangjiaba_interdam_reference", True, False),
    60104800: ("Xiangjiaba_downstream_reference", True, False),
    60105400: ("TGD_upstream_regional_reference", True, False),
    60107170: ("Zhangjiang_downstream_area_reference", False, True),
    60107300: ("Zhangjiang_near_Yichang_candidate", False, True),
}


def build_station_catalog(dbf_path: str | Path, series_dir: str | Path) -> pd.DataFrame:
    meta = read_dbf(dbf_path)
    meta["code"] = meta["code"].astype(int)
    selected = meta[meta["code"].isin(STATION_ROLES)].copy()
    records = []
    for _, row in selected.iterrows():
        code = int(row["code"])
        role, calibration, validation = STATION_ROLES[code]
        csv_path = Path(series_dir) / f"{code}.csv"
        if csv_path.exists():
            data = pd.read_csv(csv_path)
            dates = pd.to_datetime(data["Date"])
            summer = dates.dt.month.between(5, 10)
            n_records = len(data)
            n_summer = int(summer.sum())
            first_date, last_date = dates.min().date(), dates.max().date()
        else:
            n_records = n_summer = 0
            first_date = last_date = None
        records.append({
            "station_id": code,
            "station_name_cdr2": row["rvnm"],
            "latitude": float(row["lat"]),
            "longitude": float(row["lon"]),
            "record_start_metadata": int(row["start"]),
            "record_end_metadata": int(row["end"]),
            "metadata_obs_count": int(float(row["obs"])),
            "mean_q_m3s": float(row["meanQ"]),
            "local_csv_records": n_records,
            "local_summer_records": n_summer,
            "local_first_date": first_date,
            "local_last_date": last_date,
            "role": role,
            "use_for_calibration": calibration,
            "use_for_external_validation": validation,
        })
    return pd.DataFrame(records).sort_values("station_id").reset_index(drop=True)
