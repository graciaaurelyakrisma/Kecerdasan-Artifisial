from flask import Flask, request, render_template_string, jsonify, Response
from supabase import create_client, Client
from dotenv import load_dotenv
import folium
from folium.plugins import Fullscreen, MeasureControl, MiniMap
import geopandas as gpd
import os, zipfile, tempfile, shutil, csv, io, json
import numpy as np
from collections import Counter

load_dotenv("supabes.env")

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

_cache = {}

# ── Path GeoJSON default (satu folder dengan app.py) ─────────────────────────
GEOJSON_DEFAULT = os.path.join(os.path.dirname(__file__), "toraja_admin_slope.geojson")

# ── Kirim data ke Supabase ────────────────────────────────────────────────────
def simpan_ke_supabase(nama, slope, luas, kelas, kode):
    try:
        # Cek dulu apakah data dengan nama ini sudah ada di tabel
        existing = supabase.table("analisis_lereng").select("id").eq("nama", str(nama)).execute()
        if existing.data:
            print(f"⏭ Skip: '{nama}' sudah ada di Supabase, tidak duplikat")
            return False

        data = {
            "nama": str(nama) if nama else "Tidak Diketahui",
            "slope": float(slope),
            "luas": float(luas),
            "kelas": str(kelas),
            "kode": str(kode)
        }
        supabase.table("analisis_lereng").insert(data).execute()
        print(f"💾 Supabase Log: Data poligon '{nama}' berhasil disimpan!")
        return True
    except Exception as e:
        print("❌ Gagal kirim ke Supabase karena:", str(e))
        return False

# ── Klasifikasi Kemiringan Lereng (SNI / Van Zuidam) ─────────────────────────
KELAS_LERENG = [
    {"kelas": "Datar",         "min": 0,    "max": 2,    "warna": "#27ae60", "kode": "I",   "deskripsi": "Sangat cocok untuk pertanian & permukiman"},
    {"kelas": "Landai",        "min": 2,    "max": 8,    "warna": "#f1c40f", "kode": "II",  "deskripsi": "Cocok untuk pertanian dengan teknik konservasi"},
    {"kelas": "Agak Curam",    "min": 8,    "max": 15,   "warna": "#e67e22", "kode": "III", "deskripsi": "Perlu teknik konservasi yang baik"},
    {"kelas": "Curam",         "min": 15,   "max": 25,   "warna": "#e74c3c", "kode": "IV",  "deskripsi": "Tidak disarankan untuk pertanian intensif"},
    {"kelas": "Sangat Curam",  "min": 25,   "max": 45,   "warna": "#8e44ad", "kode": "V",   "deskripsi": "Rawan longsor, sebaiknya hutan lindung"},
    {"kelas": "Sangat Terjal", "min": 45,   "max": 9999, "warna": "#2c3e50", "kode": "VI",  "deskripsi": "Sangat rawan longsor, kawasan lindung mutlak"},
]

REKOMENDASI = {
    "I":  {
        "penggunaan": "Pertanian intensif, sawah irigasi, permukiman, kawasan industri ringan",
        "mitigasi":   "Drainase permukaan sederhana, rotasi tanaman",
        "vegetasi":   "Padi, jagung, sayuran, tanaman pangan",
        "prioritas":  "Rendah",
        "warna_risk": "#27ae60",
        "icon":       "🌾",
    },
    "II": {
        "penggunaan": "Pertanian lahan kering dengan terasering, perkebunan, kebun campuran",
        "mitigasi":   "Teras bangku, strip cropping, mulsa organik",
        "vegetasi":   "Kopi, teh, kakao, tanaman hortikultura",
        "prioritas":  "Rendah–Sedang",
        "warna_risk": "#f1c40f",
        "icon":       "🌿",
    },
    "III": {
        "penggunaan": "Perkebunan tahunan, hutan produksi, wisata alam terbatas",
        "mitigasi":   "Teras gulud, rorak, agroforestry, biopori",
        "vegetasi":   "Karet, kelapa sawit, sengon, bambu",
        "prioritas":  "Sedang",
        "warna_risk": "#e67e22",
        "icon":       "🌲",
    },
    "IV": {
        "penggunaan": "Hutan produksi terbatas, ekowisata dengan jalur khusus",
        "mitigasi":   "Reboisasi wajib, check dam, larangan pembukaan lahan baru",
        "vegetasi":   "Akasia, mahoni, pinus, tanaman endemik",
        "prioritas":  "Tinggi",
        "warna_risk": "#e74c3c",
        "icon":       "🚧",
    },
    "V": {
        "penggunaan": "Kawasan hutan lindung, daerah tangkapan air",
        "mitigasi":   "Vegetasi permanen, larangan penebangan, monitoring berkala",
        "vegetasi":   "Vegetasi hutan alam, cover crop permanen",
        "prioritas":  "Sangat Tinggi",
        "warna_risk": "#8e44ad",
        "icon":       "⛔",
    },
    "VI": {
        "penggunaan": "Kawasan konservasi / lindung mutlak, tidak boleh ada aktivitas manusia",
        "mitigasi":   "Buffer zone minimal 500m, early warning system longsor",
        "vegetasi":   "Hutan alam tak terganggu",
        "prioritas":  "KRITIS",
        "warna_risk": "#2c3e50",
        "icon":       "🚨",
    },
}

def get_kelas(slope_val):
    try:
        val = float(slope_val)
    except (ValueError, TypeError):
        return None
    for k in KELAS_LERENG:
        if k["min"] <= val < k["max"]:
            return k
    return KELAS_LERENG[-1]

def deteksi_kolom_lereng(gdf):
    prioritas = ['SLOPE', 'KEMIRINGAN', 'LERENG', 'KELAS_SLOP', 'GRAD', 'GRADIENT', 'INCLINE']
    for p in prioritas:
        for col in gdf.columns:
            if p in col.upper():
                return col
    for col in gdf.columns:
        if gdf[col].dtype in [np.float64, np.int64, np.float32]:
            return col
    return gdf.columns[0]

def deteksi_nama(gdf):
    kandidat = ["NAMA", "NAMOBJ", "NAME", "KECAMATAN", "DESA", "WILAYAH", "GRIDCODE", "ID"]
    for k in kandidat:
        for col in gdf.columns:
            if col.upper() == k:
                return col
    for k in kandidat:
        for col in gdf.columns:
            if k in col.upper():
                return col
    return gdf.columns[0]

def deteksi_wilayah(gdf):
    adm_cols = ['KABUPATEN', 'KAB', 'PROVINSI', 'PROV', 'KECAMATAN', 'KEC',
                'WILAYAH', 'DAERAH', 'REGION', 'AREA', 'LOKASI', 'NAMOBJ', 'NAMDESA']
    for k in adm_cols:
        for col in gdf.columns:
            if k in col.upper():
                vals = gdf[col].dropna().unique()
                if len(vals) == 1:
                    return str(vals[0])
                elif 1 < len(vals) <= 5:
                    return " / ".join(str(v) for v in sorted(vals)[:3])
                elif len(vals) > 5:
                    mc = Counter(gdf[col].dropna()).most_common(1)
                    return str(mc[0][0]) + " (dan sekitarnya)" if mc else None
    return None

def hitung_risk_score(slope_val):
    try:
        s = float(slope_val)
    except (ValueError, TypeError):
        return 0
    if s < 2:   return 5
    if s < 8:   return 15
    if s < 15:  return 35
    if s < 25:  return 60
    if s < 45:  return 85
    return 98

def build_popup(nama, slope_disp, kelas_nama, kode, warna, luas, deskripsi, rawan, risk_score):
    bar_w = risk_score
    bar_c = "#e74c3c" if risk_score >= 60 else "#e67e22" if risk_score >= 35 else "#27ae60"
    status_text = "⚠ Rawan Longsor" if rawan else "✓ Relatif Aman"
    status_color = "#e74c3c" if rawan else "#27ae60"
    rek = REKOMENDASI.get(kode, {})
    rek_text = rek.get("penggunaan", "-")
    html = f"""
<div style='font-family:Segoe UI,Arial;min-width:220px;max-width:260px;padding:4px'>
  <b style='font-size:14px;color:#222'>{nama}</b>
  <hr style='margin:6px 0;border:none;border-top:1px solid #ddd'>
  <table style='font-size:12px;width:100%;border-collapse:collapse'>
    <tr><td style='color:#666;padding:2px 0'>Kemiringan</td>
        <td><b>{slope_disp}°</b></td></tr>
    <tr><td style='color:#666;padding:2px 0'>Kelas</td>
        <td><b style='color:{warna}'>{kelas_nama} ({kode})</b></td></tr>
    <tr><td style='color:#666;padding:2px 0'>Luas</td>
        <td><b>{luas} km²</b></td></tr>
    <tr><td style='color:#666;padding:2px 0'>Status</td>
        <td><b style='color:{status_color}'>{status_text}</b></td></tr>
  </table>
  <div style='margin:8px 0 4px;font-size:10px;color:#888'>SKOR RISIKO LONGSOR</div>
  <div style='background:#eee;border-radius:4px;height:8px;overflow:hidden'>
    <div style='width:{bar_w}%;height:100%;background:{bar_c};border-radius:4px'></div>
  </div>
  <div style='font-size:10px;color:{bar_c};font-weight:bold;margin-top:2px'>{risk_score}/100</div>
  <div style='margin-top:8px;font-size:10px;background:#f5f5f5;border-radius:4px;padding:6px;color:#444'>
    <b>Rekomendasi:</b> {rek_text}
  </div>
</div>"""
    return html

def generate_analisis(rows_sorted, kelas_dist, wilayah, luas_total, slope_avg, pct_rawan):
    if pct_rawan >= 60:
        level = "KRITIS"
        level_color = "#e74c3c"
        level_icon = "🚨"
        summary = f"Lebih dari separuh wilayah ({pct_rawan}%) termasuk zona rawan longsor. Diperlukan tindakan mitigasi segera dan pemetaan lebih detail."
    elif pct_rawan >= 40:
        level = "TINGGI"
        level_color = "#e67e22"
        level_icon = "⚠️"
        summary = f"Sebanyak {pct_rawan}% area berisiko longsor. Perlu program penghijauan intensif dan pengawasan ketat terhadap aktivitas lahan."
    elif pct_rawan >= 20:
        level = "SEDANG"
        level_color = "#f1c40f"
        level_icon = "🔶"
        summary = f"Terdapat {pct_rawan}% area berpotensi rawan. Penggunaan lahan perlu direncanakan dengan mempertimbangkan konservasi."
    else:
        level = "RENDAH"
        level_color = "#27ae60"
        level_icon = "✅"
        summary = f"Hanya {pct_rawan}% area yang tergolong rawan. Kondisi relatif aman namun tetap perlu mempertahankan tutupan lahan yang baik."

    kritis = [r for r in rows_sorted if r["rawan"]][:8]

    rek_per_kelas = []
    for kode in ["I", "II", "III", "IV", "V", "VI"]:
        if kode in kelas_dist:
            d = kelas_dist[kode]
            r = REKOMENDASI.get(kode, {})
            rek_per_kelas.append({
                "kode": kode,
                "kelas": d["kelas"],
                "warna": d["warna"],
                "luas": round(d["luas"], 2),
                "pct": round(d["luas"] / (luas_total or 1) * 100, 1),
                "n": d["n"],
                "penggunaan": r.get("penggunaan", "-"),
                "mitigasi": r.get("mitigasi", "-"),
                "vegetasi": r.get("vegetasi", "-"),
                "prioritas": r.get("prioritas", "-"),
                "warna_risk": r.get("warna_risk", "#95a5a6"),
                "icon": r.get("icon", "📍"),
            })

    slopes_num = []
    for r in rows_sorted:
        try:
            slopes_num.append(float(r["slope"]))
        except (ValueError, TypeError):
            pass
    slope_min = round(min(slopes_num), 2) if slopes_num else 0
    slope_max = round(max(slopes_num), 2) if slopes_num else 0

    return {
        "level": level,
        "level_color": level_color,
        "level_icon": level_icon,
        "summary": summary,
        "kritis": kritis,
        "rek_per_kelas": rek_per_kelas,
        "slope_min": slope_min,
        "slope_max": slope_max,
    }


# ═════════════════════════════════════════════════════════════════════════════
# FUNGSI TERPUSAT: proses GeoDataFrame → semua variabel tampilan
# Dipanggil baik saat auto-load GeoJSON maupun saat upload shapefile
# ═════════════════════════════════════════════════════════════════════════════
def proses_gdf(gdf, simpan_supabase=False):
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    elif gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    jumlah_poligon = len(gdf)
    wilayah_nama   = deteksi_wilayah(gdf) or "Wilayah Tidak Diketahui"
    kolom_slope    = deteksi_kolom_lereng(gdf)
    kolom_nama     = deteksi_nama(gdf)
    gdf_meter      = gdf.to_crs(gdf.estimate_utm_crs())

    total_luas = total_slope_w = luas_rawan = 0
    kelas_distribusi = {}
    tabel_rows = []

    for idx, row in gdf.iterrows():
        raw_nama  = row[kolom_nama]
        raw_slope = row[kolom_slope]
        try:
            s_val = float(raw_slope)
        except Exception:
            s_val = 0.0

        area_km2 = gdf_meter.loc[idx, 'geometry'].area / 1_000_000
        total_luas    += area_km2
        total_slope_w += s_val * area_km2

        kelas_info = get_kelas(s_val)
        is_rawan   = s_val >= 15.0
        if is_rawan:
            luas_rawan += area_km2

        kode_k = kelas_info["kode"]
        if kode_k not in kelas_distribusi:
            kelas_distribusi[kode_k] = {
                "kelas": kelas_info["kelas"],
                "warna": kelas_info["warna"],
                "luas": 0, "n": 0
            }
        kelas_distribusi[kode_k]["luas"] += area_km2
        kelas_distribusi[kode_k]["n"]    += 1

        risk_sc = hitung_risk_score(s_val)
        r_color = "#e74c3c" if risk_sc >= 60 else "#e67e22" if risk_sc >= 35 else "#27ae60"

        tabel_rows.append({
            "nama": raw_nama, "slope": round(s_val, 2),
            "luas": round(area_km2, 4), "kelas": kelas_info["kelas"],
            "kode": kode_k, "warna": kelas_info["warna"],
            "rawan": is_rawan, "risk_score": risk_sc, "risk_color": r_color
        })

        # Simpan ke Supabase hanya jika diminta (saat upload shapefile baru)
        if simpan_supabase:
            simpan_ke_supabase(
                nama=raw_nama, slope=round(s_val, 2),
                luas=round(area_km2, 4), kelas=kelas_info["kelas"], kode=kode_k
            )

    luas_total_disp  = round(total_luas, 2)
    slope_avg_disp   = round(total_slope_w / total_luas, 2) if total_luas > 0 else 0
    pct_rawan_disp   = round(luas_rawan / total_luas * 100, 1) if total_luas > 0 else 0
    tabel_rows       = sorted(tabel_rows, key=lambda x: x["slope"], reverse=True)
    analisis         = generate_analisis(tabel_rows, kelas_distribusi, wilayah_nama,
                                         total_luas, slope_avg_disp, pct_rawan_disp)

    # ── Buat peta Folium menarik per poligon administrasi ───────────────────
    bounds     = gdf.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles=None
    )

    # Basemap pilihan
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="CartoDB", name="Dark", max_zoom=19
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satelit"
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street Map").add_to(m)

    Fullscreen().add_to(m)
    MeasureControl(position="topleft", active_color="#f0a500", completed_color="#e67e22").add_to(m)
    MiniMap(toggle_display=True, position="bottomright", tile_layer="cartodbdark_matter").add_to(m)

    # ── Render tiap poligon dengan warna kelas + efek visual menarik ─────────
    for idx, row in gdf.iterrows():
        s_val  = float(row[kolom_slope]) if kolom_slope in gdf.columns else 0
        k_inf  = get_kelas(s_val)
        area_p = gdf_meter.loc[idx, "geometry"].area / 1_000_000
        r_sc   = hitung_risk_score(s_val)
        warna  = k_inf["warna"]
        nama   = str(row[kolom_nama])

        popup_content = build_popup(
            nama, round(s_val, 1), k_inf["kelas"],
            k_inf["kode"], warna, round(area_p, 3),
            k_inf["deskripsi"], (s_val >= 15.0), r_sc
        )

        # Tooltip hover
        tooltip_html = f"""
        <div style='font-family:Arial;font-size:12px;background:rgba(0,0,0,0.85);
             color:white;padding:8px 12px;border-radius:6px;border-left:3px solid {warna}'>
            <b>{nama}</b><br>
            <span style='color:{warna}'>■</span> {k_inf["kelas"]} ({k_inf["kode"]})<br>
            Slope: <b>{round(s_val,1)}°</b> | Risiko: <b>{r_sc}/100</b>
        </div>"""

        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda x, w=warna: {
                "fillColor": w,
                "color":     "#ffffff",
                "weight":    1.5,
                "fillOpacity": 0.75,
                "dashArray": ""
            },
            highlight_function=lambda x, w=warna: {
                "fillColor":   w,
                "color":       "#f0a500",
                "weight":      3,
                "fillOpacity": 0.95,
            },
            tooltip=folium.Tooltip(tooltip_html, sticky=True),
            popup=folium.Popup(popup_content, max_width=300)
        ).add_to(m)

        # Label nama kecamatan
        cx = row["geometry"].centroid.x
        cy = row["geometry"].centroid.y
        folium.Marker(
            location=[cy, cx],
            icon=folium.DivIcon(
                html=f'''<div style="font-family:Arial;font-size:8px;font-weight:700;
                     color:white;text-shadow:1px 1px 2px #000,-1px -1px 2px #000,
                     1px -1px 2px #000,-1px 1px 2px #000;
                     text-align:center;white-space:nowrap;pointer-events:none">
                     {nama}</div>''',
                icon_size=(110, 18),
                icon_anchor=(55, 9)
            )
        ).add_to(m)

    # ── Legenda menarik di pojok kanan bawah ─────────────────────────────────
    legend_html = """
    <div style="position:fixed;bottom:40px;right:10px;z-index:9999;
         background:linear-gradient(135deg,rgba(13,17,23,0.95),rgba(22,27,34,0.95));
         border:1px solid rgba(240,165,0,0.4);border-radius:10px;
         padding:14px 16px;font-family:Arial;font-size:11px;color:white;
         min-width:180px;box-shadow:0 4px 20px rgba(0,0,0,0.5)">
      <div style="font-family:Arial Black;font-size:12px;color:#f0a500;
           letter-spacing:1px;margin-bottom:10px;border-bottom:1px solid rgba(240,165,0,0.3);
           padding-bottom:6px">⛰ KELAS LERENG SNI</div>
      <div style="display:flex;align-items:center;gap:8px;margin:5px 0">
        <div style="width:18px;height:14px;background:#27ae60;border-radius:3px;flex-shrink:0"></div>
        <span><b>I</b> — Datar <span style="color:#8b949e">(0–2°)</span></span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin:5px 0">
        <div style="width:18px;height:14px;background:#f1c40f;border-radius:3px;flex-shrink:0"></div>
        <span><b>II</b> — Landai <span style="color:#8b949e">(2–8°)</span></span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin:5px 0">
        <div style="width:18px;height:14px;background:#e67e22;border-radius:3px;flex-shrink:0"></div>
        <span><b>III</b> — Agak Curam <span style="color:#8b949e">(8–15°)</span></span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin:5px 0">
        <div style="width:18px;height:14px;background:#e74c3c;border-radius:3px;flex-shrink:0"></div>
        <span><b>IV</b> — Curam <span style="color:#8b949e">(15–25°)</span></span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin:5px 0">
        <div style="width:18px;height:14px;background:#8e44ad;border-radius:3px;flex-shrink:0"></div>
        <span><b>V</b> — Sangat Curam <span style="color:#8b949e">(25–45°)</span></span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin:5px 0">
        <div style="width:18px;height:14px;background:#2c3e50;border-radius:3px;flex-shrink:0"></div>
        <span><b>VI</b> — Sangat Terjal <span style="color:#8b949e">(&gt;45°)</span></span>
      </div>
      <div style="margin-top:10px;padding-top:6px;border-top:1px solid rgba(255,255,255,0.1);
           font-size:9px;color:#8b949e;text-align:center">
        Klik poligon untuk info detail
      </div>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(position="topleft").add_to(m)
    m.fit_bounds([[bounds[1]-0.01, bounds[0]-0.01], [bounds[3]+0.01, bounds[2]+0.01]])
    peta_html = m._repr_html_()

    # ── Risk bar & alert & proporsi ───────────────────────────────────────────
    risk_bar_html = ""
    dom_data_list = []
    for k in ["I", "II", "III", "IV", "V", "VI"]:
        if k in kelas_distribusi:
            pct_k = round(kelas_distribusi[k]["luas"] / total_luas * 100, 1)
            risk_bar_html += (
                f'<div class="risk-seg" style="flex:{pct_k};'
                f'background:{kelas_distribusi[k]["warna"]}" '
                f'title="{kelas_distribusi[k]["kelas"]}: {pct_k}%"></div>'
            )
            dom_data_list.append({
                "kelas": kelas_distribusi[k]["kelas"],
                "pct": pct_k,
                "warna": kelas_distribusi[k]["warna"]
            })

    if pct_rawan_disp >= 40:
        risk_alert_html = (
            f'<div class="alert-box alert-danger">🚨 <b>Peringatan Kritis:</b> '
            f'Wilayah ini didominasi zona lereng curam ({pct_rawan_disp}% rawan). '
            f'Potensi longsor masif tinggi!</div>'
        )
    elif pct_rawan_disp >= 15:
        risk_alert_html = (
            f'<div class="alert-box alert-warn">🔶 <b>Perhatian:</b> '
            f'Sebagian wilayah ({pct_rawan_disp}%) rawan pergerakan tanah. '
            f'Hindari alih fungsi lahan hutan.</div>'
        )
    else:
        risk_alert_html = (
            f'<div class="alert-box alert-ok">✅ <b>Kondisi Aman:</b> '
            f'Tingkat kerawanan rendah ({pct_rawan_disp}%). '
            f'Cocok untuk pengembangan infrastruktur.</div>'
        )

    # ── Update cache untuk export ─────────────────────────────────────────────
    _cache['last_data'] = {
        "wilayah": wilayah_nama, "total_poligon": jumlah_poligon,
        "luas_total": luas_total_disp, "slope_rata_rata": slope_avg_disp,
        "persentase_rawan": pct_rawan_disp, "rows": tabel_rows
    }

    return {
        "error": None,
        "peta_html": peta_html,
        "wilayah_nama": wilayah_nama,
        "jumlah_poligon": jumlah_poligon,
        "luas_total_disp": luas_total_disp,
        "slope_avg_disp": slope_avg_disp,
        "pct_rawan_disp": pct_rawan_disp,
        "tabel_rows": tabel_rows,
        "risk_bar_html": risk_bar_html,
        "risk_alert_html": risk_alert_html,
        "dom_data_list": dom_data_list,
        "analisis": analisis,
        "chart_data": True,
    }


# ── Template HTML ─────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WebGIS Kemiringan Lereng{% if wilayah_nama %} — {{ wilayah_nama }}{% endif %}</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent: #f0a500;
    --danger: #e74c3c;
    --safe: #27ae60;
    --warn: #e67e22;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:'DM Sans',sans-serif; background:var(--bg); color:var(--text); height:100vh; overflow:hidden; display:flex; flex-direction:column; }
.header { background:linear-gradient(135deg,#1a2332 0%,#0d1117 100%); border-bottom:1px solid var(--border); padding:0 20px; height:52px; display:flex; align-items:center; justify-content:space-between; flex-shrink:0; gap:12px; }
.header-left { display:flex; align-items:center; gap:12px; min-width:0; }
.header-title { font-family:'Bebas Neue',sans-serif; font-size:20px; letter-spacing:2px; color:var(--accent); white-space:nowrap; }
.header-wilayah { font-size:12px; color:var(--text); opacity:0.75; border-left:1px solid var(--border); padding-left:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:300px; }
.header-wilayah b { color:var(--accent); }
.header-right { display:flex; align-items:center; gap:8px; flex-shrink:0; }
.header-badge { background:var(--accent); color:#000; font-size:10px; font-weight:700; padding:3px 10px; border-radius:20px; letter-spacing:1px; }
.btn-export { display:flex; align-items:center; gap:5px; background:rgba(240,165,0,0.12); border:1px solid rgba(240,165,0,0.3); color:var(--accent); padding:5px 12px; border-radius:6px; font-size:11px; font-weight:600; cursor:pointer; transition:all 0.15s; text-decoration:none; }
.btn-export:hover { background:rgba(240,165,0,0.22); }
.main { display:flex; flex:1; overflow:hidden; }
.sidebar { width:300px; background:var(--surface); border-right:1px solid var(--border); display:flex; flex-direction:column; overflow:hidden; flex-shrink:0; }
.sidebar-scroll { overflow-y:auto; flex:1; padding:14px; }
.sidebar-scroll::-webkit-scrollbar { width:3px; }
.sidebar-scroll::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }
.upload-zone { border:2px dashed var(--border); border-radius:10px; padding:16px; text-align:center; cursor:pointer; transition:all 0.2s; margin-bottom:10px; position:relative; }
.upload-zone:hover { border-color:var(--accent); background:rgba(240,165,0,0.04); }
.upload-zone input[type=file] { position:absolute; inset:0; opacity:0; cursor:pointer; width:100%; height:100%; }
.upload-icon { font-size:24px; margin-bottom:4px; }
.upload-label { font-size:12px; color:var(--muted); }
.upload-label b { color:var(--accent); }
.btn-upload { width:100%; padding:9px; background:var(--accent); color:#000; border:none; border-radius:8px; font-family:'DM Sans',sans-serif; font-weight:700; font-size:13px; cursor:pointer; transition:opacity 0.2s; margin-top:6px; }
.btn-upload:hover { opacity:0.85; }
.error-msg { background:rgba(231,76,60,0.15); border:1px solid rgba(231,76,60,0.4); color:#ff8080; padding:8px 12px; border-radius:6px; font-size:11px; margin-top:8px; }
.section-label { font-size:10px; font-weight:700; letter-spacing:2px; color:var(--muted); text-transform:uppercase; margin:14px 0 8px; display:flex; align-items:center; gap:6px; }
.section-label::after { content:''; flex:1; height:1px; background:var(--border); }
.stats-grid { display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-bottom:10px; }
.stat-card { background:var(--bg); border:1px solid var(--border); border-radius:8px; padding:10px; }
.stat-val { font-family:'JetBrains Mono',monospace; font-size:18px; font-weight:600; color:var(--accent); line-height:1; }
.stat-lbl { font-size:9px; color:var(--muted); margin-top:3px; text-transform:uppercase; letter-spacing:0.5px; }
.legend-item { display:flex; align-items:center; gap:8px; padding:7px 9px; border-radius:6px; margin-bottom:3px; background:var(--bg); border:1px solid var(--border); transition:all 0.15s; cursor:pointer; user-select:none; }
.legend-item:hover { border-color:var(--accent); }
.legend-item.filtered-out { opacity:0.35; }
.legend-item.active-filter { border-color:var(--accent); background:rgba(240,165,0,0.08); }
.legend-swatch { width:10px; height:32px; border-radius:3px; flex-shrink:0; }
.legend-text { flex:1; }
.legend-kelas { font-size:11px; font-weight:500; }
.legend-range { font-size:10px; color:var(--muted); font-family:'JetBrains Mono',monospace; }
.legend-code { font-family:'JetBrains Mono',monospace; font-size:10px; font-weight:600; background:rgba(255,255,255,0.06); padding:2px 6px; border-radius:4px; }
.filter-hint { font-size:10px; color:var(--muted); text-align:center; margin-bottom:6px; }
.btn-reset-filter { width:100%; padding:6px; background:transparent; border:1px solid var(--border); color:var(--muted); border-radius:6px; font-size:11px; cursor:pointer; margin-top:4px; transition:all 0.15s; }
.btn-reset-filter:hover { border-color:var(--accent); color:var(--accent); }
.dom-item { margin-bottom:7px; }
.dom-header { display:flex; justify-content:space-between; font-size:11px; margin-bottom:3px; }
.dom-bar { height:5px; border-radius:3px; background:var(--border); overflow:hidden; }
.dom-fill { height:100%; border-radius:3px; }
.risk-bar { display:flex; height:18px; border-radius:6px; overflow:hidden; gap:2px; margin:6px 0; }
.risk-seg { transition:flex 0.5s ease; }
.alert-box { border-radius:8px; padding:10px 12px; font-size:11px; line-height:1.5; margin-top:6px; }
.alert-danger { background:rgba(231,76,60,0.12); border:1px solid rgba(231,76,60,0.3); }
.alert-warn   { background:rgba(230,126,34,0.12); border:1px solid rgba(230,126,34,0.3); }
.alert-ok     { background:rgba(39,174,96,0.12);  border:1px solid rgba(39,174,96,0.3); }
.content { flex:1; display:flex; flex-direction:column; overflow:hidden; background:var(--bg); }
.tab-bar { display:flex; align-items:center; background:var(--surface); border-bottom:1px solid var(--border); padding:0 14px; gap:2px; height:42px; flex-shrink:0; }
.tab-btn { background:none; border:none; color:var(--muted); font-family:'DM Sans',sans-serif; font-size:12px; font-weight:500; padding:0 12px; height:100%; cursor:pointer; border-bottom:2px solid transparent; transition:all 0.15s; display:flex; align-items:center; gap:5px; }
.tab-btn:hover { color:var(--text); }
.tab-btn.active { color:var(--accent); border-bottom-color:var(--accent); }
.tab-spacer { flex:1; }
.tab-pane { display:none; flex:1; overflow:auto; }
.tab-pane.active { display:flex; flex-direction:column; flex:1; }
#pane-map { display:none; flex:1; overflow:hidden; }
#pane-map.active { display:block; }
#pane-map iframe, #pane-map > div { width:100% !important; height:100% !important; }
.search-bar { padding:12px 16px 8px; display:flex; gap:8px; align-items:center; background:var(--surface); border-bottom:1px solid var(--border); }
.search-input { flex:1; background:var(--bg); border:1px solid var(--border); color:var(--text); padding:6px 10px; border-radius:6px; font-size:12px; font-family:'DM Sans',sans-serif; outline:none; }
.search-input:focus { border-color:var(--accent); }
.btn-csv { display:flex; align-items:center; gap:5px; background:rgba(39,174,96,0.12); border:1px solid rgba(39,174,96,0.3); color:#27ae60; padding:6px 12px; border-radius:6px; font-size:11px; font-weight:600; cursor:pointer; transition:all 0.15s; }
.btn-csv:hover { background:rgba(39,174,96,0.22); }
.data-table { width:100%; border-collapse:collapse; font-size:12px; }
.data-table th { background:var(--surface); border:1px solid var(--border); padding:8px 10px; text-align:left; font-size:10px; letter-spacing:1px; text-transform:uppercase; color:var(--muted); position:sticky; top:0; cursor:pointer; }
.data-table th:hover { color:var(--accent); }
.data-table td { border:1px solid var(--border); padding:7px 10px; color:var(--text); }
.data-table tr:nth-child(even) td { background:rgba(255,255,255,0.02); }
.data-table tr:hover td { background:rgba(240,165,0,0.05); }
.data-table tr.hidden-row { display:none; }
.badge-kelas { display:inline-block; padding:2px 8px; border-radius:4px; font-size:10px; font-weight:700; color:#fff; }
#pane-chart { padding:20px; overflow:auto; }
#pane-chart.active { display:flex; flex-direction:column; }
.chart-grid { display:grid; grid-template-columns:1fr 1fr; gap:24px; }
.chart-card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px; }
.chart-title { font-family:'Bebas Neue',sans-serif; letter-spacing:2px; color:var(--accent); font-size:14px; margin-bottom:14px; }
#pane-analisis { padding:20px; overflow:auto; }
#pane-analisis.active { display:block; }
.risk-header { border-radius:10px; padding:18px 20px; margin-bottom:20px; display:flex; align-items:flex-start; gap:16px; }
.risk-icon { font-size:40px; flex-shrink:0; }
.risk-level-text { font-family:'Bebas Neue',sans-serif; font-size:28px; letter-spacing:3px; }
.risk-summary { font-size:12px; color:var(--muted); margin-top:4px; line-height:1.5; }
.rek-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:20px; }
.rek-card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:14px; border-left:3px solid; }
.rek-header { display:flex; align-items:center; gap:8px; margin-bottom:8px; }
.rek-badge { font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:600; padding:2px 8px; border-radius:4px; color:#fff; }
.rek-kelas { font-size:13px; font-weight:600; }
.rek-pct { font-size:11px; color:var(--muted); margin-left:auto; }
.rek-field { margin-top:6px; }
.rek-label { font-size:9px; letter-spacing:1.5px; text-transform:uppercase; color:var(--muted); }
.rek-value { font-size:11px; color:var(--text); margin-top:1px; line-height:1.4; }
.priority-pill { display:inline-block; font-size:9px; font-weight:700; padding:1px 7px; border-radius:10px; letter-spacing:1px; margin-top:4px; }
.kritis-section { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px; }
.kritis-title { font-family:'Bebas Neue',sans-serif; letter-spacing:2px; color:var(--accent); font-size:14px; margin-bottom:12px; }
.kritis-item { display:flex; align-items:center; gap:10px; padding:8px 0; border-bottom:1px solid var(--border); }
.kritis-item:last-child { border-bottom:none; }
.kritis-rank { font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--muted); width:20px; }
.kritis-nama { flex:1; font-size:12px; }
.kritis-slope { font-family:'JetBrains Mono',monospace; font-size:11px; }
.kritis-kelas { font-size:10px; font-weight:700; padding:1px 7px; border-radius:3px; color:#fff; }
.empty-state { display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; color:var(--muted); font-size:14px; gap:12px; }
@media print {
  .sidebar, .tab-bar, .header-right, .search-bar { display:none !important; }
  .content { width:100% !important; }
  .tab-pane, #pane-analisis { display:block !important; }
}
@media (max-width: 900px) {
  .rek-grid { grid-template-columns:1fr; }
  .chart-grid { grid-template-columns:1fr; }
}
</style>
</head>
<body>
<div class="header">
  <div class="header-left">
    <div class="header-title">⛰ WebGIS Lereng</div>
    {% if wilayah_nama %}
    <div class="header-wilayah">📍 <b>{{ wilayah_nama }}</b></div>
    {% else %}
    <div class="header-wilayah" style="opacity:0.4">Upload shapefile untuk memulai</div>
    {% endif %}
  </div>
  <div class="header-right">
    {% if tabel_rows %}
    <button class="btn-export" onclick="exportJSON()" title="Download analisis JSON">⬇ JSON</button>
    <button class="btn-export" onclick="window.print()" title="Cetak laporan">🖨 Print</button>
    {% endif %}
    <div class="header-badge">SNI / VAN ZUIDAM</div>
  </div>
</div>
<div class="main">
  <div class="sidebar">
  <div class="sidebar-scroll">
    {% if error %}<div class="error-msg">⚠ {{ error }}</div>{% endif %}
    {% if wilayah_nama %}
    <div class="section-label">📍 Wilayah</div>
    <div style="background:rgba(240,165,0,0.08);border:1px solid rgba(240,165,0,0.25);border-radius:8px;padding:10px 12px;font-size:12px">
      <div style="font-weight:600;color:var(--accent)">{{ wilayah_nama }}</div>
      <div style="font-size:10px;color:var(--muted);margin-top:3px">CRS: EPSG:4326 (WGS84)</div>
    </div>
    {% endif %}
    <div class="section-label">📊 Statistik</div>
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-val">{{ jumlah }}</div>
        <div class="stat-lbl">Total Poligon</div>
      </div>
      <div class="stat-card">
        <div class="stat-val">{{ luas_total }}</div>
        <div class="stat-lbl">Luas (km²)</div>
      </div>
      <div class="stat-card">
        <div class="stat-val">{{ slope_avg }}°</div>
        <div class="stat-lbl">Slope Rata-rata</div>
      </div>
      <div class="stat-card">
        <div class="stat-val" style="color:var(--danger)">{{ pct_rawan }}%</div>
        <div class="stat-lbl">Area Rawan</div>
      </div>
      {% if analisis %}
      <div class="stat-card">
        <div class="stat-val" style="font-size:14px">{{ analisis.slope_min }}°</div>
        <div class="stat-lbl">Slope Min</div>
      </div>
      <div class="stat-card">
        <div class="stat-val" style="font-size:14px">{{ analisis.slope_max }}°</div>
        <div class="stat-lbl">Slope Maks</div>
      </div>
      {% endif %}
    </div>
    <div class="section-label">⚠ Distribusi Risiko</div>
    <div class="risk-bar">{{ risk_bar|safe }}</div>
    {{ risk_alert|safe }}
    <div class="section-label">🎨 Filter Kelas Lereng</div>
    <p class="filter-hint">Klik kelas untuk filter peta &amp; tabel</p>
    {% for k in kelas_list %}
    <div class="legend-item" id="legend-{{ k.kode }}" onclick="toggleFilter('{{ k.kode }}')" title="{{ k.deskripsi }}">
      <div class="legend-swatch" style="background:{{ k.warna }}"></div>
      <div class="legend-text">
        <div class="legend-kelas">{{ k.kelas }}</div>
        <div class="legend-range">{{ k.min }}°–{% if k.max == 9999 %}&gt;45°{% else %}{{ k.max }}°{% endif %}</div>
      </div>
      <div class="legend-code">{{ k.kode }}</div>
    </div>
    {% endfor %}
    <button class="btn-reset-filter" onclick="resetFilter()">↺ Tampilkan Semua</button>
    {% if dom_data %}
    <div class="section-label">📈 Proporsi Luas</div>
    {% for d in dom_data %}
    <div class="dom-item">
      <div class="dom-header">
        <span>{{ d.kelas }}</span>
        <span style="font-family:'JetBrains Mono',monospace;color:var(--accent)">{{ d.pct }}%</span>
      </div>
      <div class="dom-bar"><div class="dom-fill" style="width:{{ d.pct }}%;background:{{ d.warna }}"></div></div>
    </div>
    {% endfor %}
    {% endif %}
  </div>
  </div>
  <div class="content">
    <div class="tab-bar">
      <button class="tab-btn active" onclick="switchTab('map',this)">🗺 Peta</button>
      <button class="tab-btn" onclick="switchTab('table',this)">📋 Tabel</button>
      <button class="tab-btn" onclick="switchTab('chart',this)">📊 Grafik</button>
      <button class="tab-btn" onclick="switchTab('analisis',this)">🔍 Analisis</button>
      <div class="tab-spacer"></div>
    </div>
    <div id="pane-map" class="active">
      {% if peta %}
        {{ peta|safe }}
      {% else %}
        <div class="empty-state">
          <div style="font-size:48px">⛰</div>
          <div>Upload shapefile kemiringan lereng untuk memulai</div>
          <div style="font-size:12px;opacity:0.5">Format: ZIP berisi .shp, .shx, .dbf</div>
        </div>
      {% endif %}
    </div>
    <div id="pane-table" class="tab-pane">
      {% if tabel_rows %}
      <div class="search-bar">
        <input class="search-input" id="searchInput" type="text" placeholder="🔍 Cari nama / kelas / status..." oninput="filterTable()">
        <select id="filterKelasSelect" class="search-input" style="max-width:140px" onchange="filterTable()">
          <option value="">Semua Kelas</option>
          {% for k in kelas_list %}<option value="{{ k.kelas }}">{{ k.kelas }}</option>{% endfor %}
        </select>
        <button class="btn-csv" onclick="exportCSV()">⬇ CSV</button>
      </div>
      <div style="overflow:auto;flex:1">
      <table class="data-table" id="dataTable">
        <thead>
          <tr>
            <th>#</th>
            <th>Nama / ID</th>
            <th>Slope (°)</th>
            <th>Luas (km²)</th>
            <th>Kelas</th>
            <th>Risiko</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody id="tableBody">
        {% for r in tabel_rows %}
        <tr data-kelas="{{ r.kelas }}" data-kode="{{ r.kode }}">
          <td>{{ loop.index }}</td>
          <td>{{ r.nama }}</td>
          <td style="font-family:'JetBrains Mono',monospace">{{ r.slope }}</td>
          <td style="font-family:'JetBrains Mono',monospace">{{ r.luas }}</td>
          <td><span class="badge-kelas" style="background:{{ r.warna }}">{{ r.kelas }} ({{ r.kode }})</span></td>
          <td>
            <div style="display:flex;align-items:center;gap:6px">
              <div style="flex:1;background:var(--border);border-radius:3px;height:6px;overflow:hidden">
                <div style="width:{{ r.risk_score }}%;height:100%;background:{{ r.risk_color }};border-radius:3px"></div>
              </div>
              <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--muted)">{{ r.risk_score }}</span>
            </div>
          </td>
          <td style="color:{% if r.rawan %}var(--danger){% else %}var(--safe){% endif %}">
            {% if r.rawan %}⚠ Rawan{% else %}✓ Aman{% endif %}
          </td>
        </tr>
        {% endfor %}
        </tbody>
      </table>
      </div>
      <div id="tableInfo" style="padding:8px 16px;font-size:11px;color:var(--muted);background:var(--surface);border-top:1px solid var(--border)">
        Menampilkan <span id="visibleCount">{{ tabel_rows|length }}</span> dari {{ tabel_rows|length }} baris
      </div>
      {% else %}
      <div class="empty-state" style="height:300px"><div>Upload shapefile untuk melihat data tabel.</div></div>
      {% endif %}
    </div>
    <div id="pane-chart" class="tab-pane">
      <div style="padding:20px;overflow:auto">
      {% if chart_data %}
      <div class="chart-grid">
        <div class="chart-card">
          <div class="chart-title">DISTRIBUSI LUAS PER KELAS (km²)</div>
          <canvas id="chartKelas"></canvas>
        </div>
        <div class="chart-card">
          <div class="chart-title">PROPORSI KELAS LERENG (%)</div>
          <canvas id="chartPie"></canvas>
        </div>
        <div class="chart-card" style="grid-column:1/-1">
          <div class="chart-title">SKOR RISIKO — TOP 15 AREA TERTINGGI</div>
          <canvas id="chartRisk"></canvas>
        </div>
      </div>
      {% else %}
      <div class="empty-state" style="height:300px"><div>Upload shapefile untuk melihat grafik.</div></div>
      {% endif %}
      </div>
    </div>
    <div id="pane-analisis" class="tab-pane">
      <div style="padding:20px;max-width:1000px;margin:0 auto;width:100%">
      {% if analisis %}
        <div class="risk-header" style="background:rgba(231,76,60,0.12);border:1px solid {{ analisis.level_color }}40">
          <div class="risk-icon">{{ analisis.level_icon }}</div>
          <div>
            <div style="font-size:10px;text-transform:uppercase;letter-spacing:2px;color:var(--muted)">Tingkat Risiko Wilayah</div>
            <div class="risk-level-text" style="color:{{ analisis.level_color }}">{{ analisis.level }}</div>
            <div class="risk-summary">{{ analisis.summary }}</div>
          </div>
        </div>
        <div class="rek-grid">
        {% for r in analisis.rek_per_kelas %}
          <div class="rek-card" style="border-left-color:{{ r.warna }}">
            <div class="rek-header">
              <span class="rek-badge" style="background:{{ r.warna }}">{{ r.kode }}</span>
              <span class="rek-kelas">{{ r.kelas }}</span>
              <span class="rek-pct">{{ r.pct }}% Luas</span>
            </div>
            <div class="rek-field">
              <div class="rek-label">Rekomendasi Struktur Lahan</div>
              <div class="rek-value">{{ r.penggunaan }}</div>
            </div>
            <div class="rek-field">
              <div class="rek-label">Mitigasi &amp; Konservasi</div>
              <div class="rek-value">{{ r.mitigasi }}</div>
            </div>
            <div class="rek-field">
              <div class="rek-label">Saran Komoditas Vegetasi</div>
              <div class="rek-value">{{ r.vegetasi }}</div>
            </div>
            <span class="priority-pill" style="background:rgba(255,255,255,0.05);color:{{ r.warna_risk }}">
              {{ r.icon }} Prioritas Mitigasi: {{ r.prioritas }}
            </span>
          </div>
        {% endfor %}
        </div>
        <div class="kritis-section">
          <div class="kritis-title">🚨 DAFTAR POLIGON DENGAN KELERENGAN TERTINGGI</div>
          {% for k in analisis.kritis %}
          <div class="kritis-item">
            <span class="kritis-rank">#{{ loop.index }}</span>
            <span class="kritis-nama"><b>{{ k.nama }}</b></span>
            <span class="kritis-slope">Slope: <b>{{ k.slope }}°</b></span>
            <span class="kritis-kelas" style="background:{{ k.warna }}">{{ k.kelas }}</span>
          </div>
          {% endfor %}
        </div>
      {% else %}
        <div class="empty-state" style="height:300px"><div>Upload shapefile untuk melihat dokumen analisis komprehensif.</div></div>
      {% endif %}
      </div>
    </div>
  </div>
</div>
<script>
function updateFileName(input) {
    var txt = input.files.length > 0 ? "📦 <b>" + input.files[0].name + "</b>" : "Drop <b>Shapefile (.zip)</b><br>atau klik untuk memilih";
    document.getElementById("upload-label-text").innerHTML = txt;
}
function switchTab(tabId, btn) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane, #pane-map').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    var target = document.getElementById('pane-' + tabId);
    if(target) target.classList.add('active');
    if(tabId === 'chart') { renderCharts(); }
}
var currentFilter = null;
function toggleFilter(kode) {
    var items = document.querySelectorAll('.legend-item');
    var rows = document.querySelectorAll('#tableBody tr');
    if(currentFilter === kode) { resetFilter(); return; }
    currentFilter = kode;
    items.forEach(it => {
        if(it.id === 'legend-' + kode) { it.classList.add('active-filter'); it.classList.remove('filtered-out'); }
        else { it.classList.remove('active-filter'); it.classList.add('filtered-out'); }
    });
    var visible = 0;
    rows.forEach(r => {
        if(r.getAttribute('data-kode') === kode) { r.classList.remove('hidden-row'); visible++; }
        else { r.classList.add('hidden-row'); }
    });
    document.getElementById("visibleCount").innerText = visible;
}
function resetFilter() {
    currentFilter = null;
    document.querySelectorAll('.legend-item').forEach(it => it.classList.remove('filtered-out', 'active-filter'));
    document.querySelectorAll('#tableBody tr').forEach(r => r.classList.remove('hidden-row'));
    var total = document.querySelectorAll('#tableBody tr').length;
    var el = document.getElementById("visibleCount");
    if(el) el.innerText = total;
}
function filterTable() {
    var q = document.getElementById("searchInput").value.toUpperCase();
    var sel = document.getElementById("filterKelasSelect").value;
    var rows = document.querySelectorAll('#tableBody tr');
    var visible = 0;
    rows.forEach(r => {
        var text = r.textContent.toUpperCase();
        var k = r.getAttribute('data-kelas');
        var matchQ = text.includes(q);
        var matchSel = sel === "" || k === sel;
        if(matchQ && matchSel) { r.classList.remove('hidden-row'); visible++; }
        else { r.classList.add('hidden-row'); }
    });
    document.getElementById("visibleCount").innerText = visible;
}
function exportJSON() { window.location.href = "/export/json"; }
function exportCSV() { window.location.href = "/export/csv"; }
var chart1, chart2, chart3;
function renderCharts() {
    if(!document.getElementById("chartKelas")) return;
    if(chart1) return;
    var rows = [];
    document.querySelectorAll('#tableBody tr').forEach(r => {
        rows.push({
            nama: r.cells[1].innerText,
            slope: parseFloat(r.cells[2].innerText),
            luas: parseFloat(r.cells[3].innerText),
            kelas: r.getAttribute('data-kelas'),
            kode: r.getAttribute('data-kode')
        });
    });
    var kData = {};
    rows.forEach(r => {
        if(!kData[r.kelas]) { kData[r.kelas] = { luas: 0, warna: window.getComputedStyle(document.getElementById("legend-"+r.kode).querySelector('.legend-swatch')).backgroundColor }; }
        kData[r.kelas].luas += r.luas;
    });
    var labels = Object.keys(kData);
    var luasVals = labels.map(l => kData[l].luas.toFixed(2));
    var warnaVals = labels.map(l => kData[l].warna);
    chart1 = new Chart(document.getElementById("chartKelas"), {
        type: 'bar',
        data: { labels: labels, datasets: [{ label: 'Luas Wilayah (km²)', data: luasVals, backgroundColor: warnaVals, borderWidth: 1 }]},
        options: { responsive: true, plugins: { legend: { display: false } } }
    });
    chart2 = new Chart(document.getElementById("chartPie"), {
        type: 'doughnut',
        data: { labels: labels, datasets: [{ data: luasVals, backgroundColor: warnaVals }]},
        options: { responsive: true }
    });
    var sortedRisk = [...rows].sort((a,b) => b.slope - a.slope).slice(0, 15);
    chart3 = new Chart(document.getElementById("chartRisk"), {
        type: 'bar',
        data: {
            labels: sortedRisk.map(s => s.nama),
            datasets: [{ label: 'Kemiringan Lereng (Derajat)', data: sortedRisk.map(s => s.slope), backgroundColor: '#e74c3c' }]
        },
        options: { responsive: true, indexAxis: 'y' }
    });
}
</script>
</body>
</html>
"""


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE UTAMA
# ═════════════════════════════════════════════════════════════════════════════
@app.route("/", methods=["GET", "POST"])
def index():
    # Nilai default kosong
    ctx = {
        "error": None, "peta_html": None, "wilayah_nama": None,
        "jumlah_poligon": 0, "luas_total_disp": 0,
        "slope_avg_disp": 0, "pct_rawan_disp": 0,
        "tabel_rows": [], "risk_bar_html": "",
        "risk_alert_html": "", "dom_data_list": [],
        "analisis": None, "chart_data": None,
    }

    # ── AUTO-LOAD GeoJSON Toraja saat pertama buka web (GET request) ──────────
    if request.method == "GET":
        if os.path.exists(GEOJSON_DEFAULT):
            try:
                with open(GEOJSON_DEFAULT, "r") as f:
                    geojson_data = json.load(f)
                gdf = gpd.GeoDataFrame.from_features(
                    geojson_data["features"], crs="EPSG:4326"
                )
                # simpan_supabase=False → tidak kirim ke DB saat auto-load,
                # ubah jadi True kalau mau tetap tersimpan
                ctx.update(proses_gdf(gdf, simpan_supabase=True))
            except Exception as e:
                ctx["error"] = f"Gagal load data default: {str(e)}"

    # ── UPLOAD shapefile baru (POST request) ──────────────────────────────────
    elif request.method == "POST":
        if 'file' not in request.files:
            ctx["error"] = "Tidak ada file yang diunggah."
        else:
            file = request.files['file']
            if file.filename == '':
                ctx["error"] = "Tidak ada file yang dipilih."
            elif file and file.filename.endswith('.zip'):
                temp_dir = tempfile.mkdtemp()
                zip_path = os.path.join(temp_dir, file.filename)
                file.save(zip_path)
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    shp_file = None
                    for root, dirs, files_list in os.walk(temp_dir):
                        for f in files_list:
                            if f.endswith('.shp'):
                                shp_file = os.path.join(root, f)
                                break
                    if not shp_file:
                        raise Exception("File .shp tidak ditemukan di dalam arsip ZIP.")
                    gdf = gpd.read_file(shp_file)
                    # simpan_supabase=True → data upload baru disimpan ke DB
                    ctx.update(proses_gdf(gdf, simpan_supabase=True))
                except Exception as e:
                    ctx["error"] = f"Gagal membaca shapefile: {str(e)}"
                finally:
                    shutil.rmtree(temp_dir)
            else:
                ctx["error"] = "Format tidak valid. Sediakan .zip berisi shapefile."

    return render_template_string(
        HTML,
        error        = ctx["error"],
        peta         = ctx["peta_html"],
        wilayah_nama = ctx["wilayah_nama"],
        jumlah       = ctx["jumlah_poligon"],
        luas_total   = ctx["luas_total_disp"],
        slope_avg    = ctx["slope_avg_disp"],
        pct_rawan    = ctx["pct_rawan_disp"],
        kelas_list   = KELAS_LERENG,
        tabel_rows   = ctx["tabel_rows"],
        risk_bar     = ctx["risk_bar_html"],
        risk_alert   = ctx["risk_alert_html"],
        dom_data     = ctx["dom_data_list"],
        chart_data   = ctx["chart_data"],
        analisis     = ctx["analisis"],
    )


# ── Export routes ─────────────────────────────────────────────────────────────
@app.route("/export/json")
def export_json_route():
    data = _cache.get('last_data', {"error": "Belum ada data dianalisis"})
    return jsonify(data)

@app.route("/export/csv")
def export_csv_route():
    data = _cache.get('last_data')
    if not data:
        return "No data available", 400
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(["No", "Nama_ID", "Slope_Derajat", "Luas_Km2", "Kelas", "Kode_Kelas", "Skor_Risiko", "Status_Rawan"])
    for idx, r in enumerate(data["rows"]):
        cw.writerow([idx+1, r["nama"], r["slope"], r["luas"], r["kelas"], r["kode"], r["risk_score"], "RAWAN" if r["rawan"] else "AMAN"])
    output = Response(si.getvalue(), mimetype="text/csv")
    output.headers["Content-Disposition"] = f"attachment; filename=Laporan_Lereng_{data['wilayah'].replace(' ','_')}.csv"
    return output


if __name__ == "__main__":
    app.run(debug=True, port=5000)