"""One-file Malaysian paddy EDA dashboard. Run normally in PyCharm.

Install once in the PyCharm terminal:
    python -m pip install pandas numpy plotly

This writes ONE interactive HTML dashboard and opens it in your browser.
No Streamlit server, command-line arguments, or separate chart windows needed.
The source CSV is never modified. Internet is needed only if no cached state
boundary file is available. The completed dashboard works offline.

Verified reference: Perangkaan Agromakanan Malaysia 2024:
  Table 2.1.2, printed p.13 (PDF p.27): area ha; production metric tonnes (mt).
  Table 2.1.13, printed p.24 (PDF p.38): yield kg/ha.
For 2017-2018, the 2020 edition agrees: Tables 2.1.2 (p.8) and 2.1.13 (p.19).
CSV production_kg is MISLABELLED: its values match the PDF's metric tonnes.
Yield is calculated as production_mt * 1000 / planted_area_ha. It may differ
slightly from the separately rounded published yield table.
"""

from pathlib import Path
import argparse
import json
import urllib.request
import webbrowser
import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

CSV_PATH = Path(r"C:\Users\fadhlina\PycharmProjects\Data Paddy\malaysia_paddy_state_gee_perangkaan_2017_2024.csv")
OUTPUT_DIR = Path(__file__).resolve().parent / "Paddy_EDA_Output"
# Optional: set to a local Malaysia ADM1/state GeoJSON. Otherwise reuse or download.
BOUNDARY_PATH = None
DEFAULT_START_YEAR = 2018  # 2017 remains selectable in the dashboard.
PRODUCTION_VALUES_UNIT = "mt"  # Verified against the supplied Perangkaan PDF.

ENVIRONMENT = {
    "Median_NDVI": ("ndvi", "Annual median NDVI", "index"),
    "Solar_Radiation_MJ": ("solar", "Mean daily solar radiation", "MJ/m²/day"),
    "Mean_Temp_C": ("temperature", "Mean temperature", "°C"),
    "Annual_Rain_mm": ("rain", "Annual rainfall", "mm/year"),
    "Soil_pH": ("ph", "Soil pH (0–5 cm)", "pH"),
    "Soil_Clay_Pct": ("clay", "Soil clay (0–5 cm)", "%"),
}


def state_key(value):
    key = str(value).strip().casefold()
    return {"penang": "pulau pinang", "malacca": "melaka",
            "negri sembilan": "negeri sembilan", "trengganu": "terengganu"}.get(key, key)


def load_data(path):
    source = pd.read_csv(path)
    source.columns = source.columns.str.strip()
    if "Year" in source and "year" not in source:
        source = source.rename(columns={"Year": "year"})
    area = next((c for c in ["planted_area_ha", "planted_area"] if c in source), None)
    production = next((c for c in ["production_mt", "production_t", "production_kg", "production"] if c in source), None)
    if area is None or production is None or not {"state", "year"}.issubset(source):
        raise ValueError("Use the combined Perangkaan/GEE CSV with state, year, planted area and production.")
    source["state"] = source.state.astype("string").str.strip().replace("", pd.NA)
    for col in ["year", area, production, *[c for c in ENVIRONMENT if c in source]]:
        source[col] = pd.to_numeric(source[col], errors="raise")
        if source[col].isin([np.inf, -np.inf]).any():
            raise ValueError(f"Infinite values in {col}.")
    if source[["state", "year", area, production]].isna().any().any():
        raise ValueError("Missing state, year, area or production. Resolve these before calculating yields.")
    if source.empty or (source.year % 1 != 0).any():
        raise ValueError("Data must be nonempty and years must be whole numbers.")
    if (source[area] <= 0).any() or (source[production] < 0).any():
        raise ValueError("Area must be positive and production nonnegative.")
    if PRODUCTION_VALUES_UNIT not in ("mt", "kg"):
        raise ValueError("PRODUCTION_VALUES_UNIT must be mt or kg.")
    d = pd.DataFrame({"state": source.state, "key": source.state.map(state_key),
                      "year": source.year.astype(int), "area": source[area],
                      "production": source[production] / (1000 if PRODUCTION_VALUES_UNIT == "kg" else 1)})
    if d.duplicated(["key", "year"]).any():
        raise ValueError("Duplicate state-year rows found; they would double-count totals.")
    d["yield"] = d.production * 1000 / d.area
    variables = {"yield": {"name": "Calculated paddy yield", "unit": "kg/ha"},
                 "area": {"name": "Planted area", "unit": "ha"},
                 "production": {"name": "Paddy production", "unit": "mt"}}
    notes = []
    if production == "production_kg" and PRODUCTION_VALUES_UNIT == "mt":
        notes.append("The source header production_kg is incorrect: its values are metric tonnes, as verified against Table 2.1.2. No division by 1,000 is applied to these values.")
    for original, (key, name, unit) in ENVIRONMENT.items():
        if original in source:
            d[key] = source[original]
            variables[key] = {"name": name, "unit": unit}
        else:
            notes.append(f"Missing environmental column: {original}. It is excluded from the visuals.")
    # Check known source anchors, without replacing source values.
    anchors = [("Johor", 2019, 2555, 7704), ("Johor", 2024, 2329, 10353),
               ("Kedah", 2024, 213556, 777021), ("Sarawak", 2024, 45269, 84510)]
    for state, year, a, p in anchors:
        row = d.loc[(d.state == state) & (d.year == year)]
        if len(row) and (not np.isclose(row.area.iloc[0], a, atol=.1, rtol=0)
                         or not np.isclose(row.production.iloc[0], p, atol=.1, rtol=0)):
            notes.append(f"Source check differs from Perangkaan 2024 Table 2.1.2: {state}, {year}. CSV values are retained.")
    for col, factor in [("actual_yield_kg_ha", 1), ("actual_yield_t_ha", 1000)]:
        if col in source:
            recorded = pd.to_numeric(source[col], errors="raise") * factor
            valid = recorded.notna()
            if valid.any():
                difference = (recorded[valid] - d.loc[valid, "yield"]).abs()
                if difference.max() > 2:
                    notes.append(f"Supplied {col} does not agree with the calculated kg/ha yield (maximum difference {difference.max():,.2f} kg/ha). Charts use production × 1,000 / area.")
    d = d.sort_values(["state", "year"])
    return json.loads(d.to_json(orient="records")), variables, notes


def load_boundaries(output, explicit=None):
    project = Path(r"/")
    candidates = [Path(explicit)] if explicit else [
        output / "malaysia_states.geojson",
        Path(__file__).resolve().parent / "paddy_maps" / "malaysia_states.geojson",
        project / "paddy_maps" / "malaysia_states.geojson",
        project / "Output_Malaysia_Yield_Maps" / "malaysia_state_boundaries.geojson",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        if explicit:
            raise FileNotFoundError(f"Boundary file not found: {explicit}")
        print("Downloading Malaysia state boundaries once...")
        def get_json(url):
            request = urllib.request.Request(url, headers={"User-Agent": "PaddyEDA/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        metadata = get_json("https://www.geoboundaries.org/api/current/gbOpen/MYS/ADM1/")
        boundary = get_json(metadata.get("simplifiedGeometryGeoJSON") or metadata["gjDownloadURL"])
        path = output / "malaysia_states.geojson"
        path.write_text(json.dumps(boundary), encoding="utf-8")
        (output / "boundary_source.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    else:
        boundary = json.loads(path.read_text(encoding="utf-8-sig"))
    if not boundary.get("features"):
        raise ValueError("No state geometries in the boundary file.")
    features = []
    for feature in boundary["features"]:
        props = feature.get("properties", {})
        name = props.get("shapeName") or props.get("name") or props.get("NAME_1")
        if not name or feature["geometry"]["type"] not in ("Polygon", "MultiPolygon"):
            raise ValueError("Boundary file needs named Polygon/MultiPolygon state features.")
        features.append({"key": state_key(name), "name": name, "geometry": feature["geometry"]})
    return features


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Malaysia Paddy | EDA dashboard</title>
<style>
:root{--ink:#173d36;--muted:#587169;--green:#176c55;--line:#dce5df;--bg:#f3f6f2}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,sans-serif}
header{padding:30px 4vw 24px;background:#123f34;color:white}header p{margin:5px 0;color:#c4dfcf}h1{font-size:32px;margin:5px 0;font-weight:650}h2{font-size:21px;margin:0 0 5px}h3{font-size:16px;margin:0 0 5px}.eyebrow{font-size:12px;letter-spacing:2px;text-transform:uppercase}
main{max-width:1550px;margin:auto;padding:20px 3vw 50px}.filters,.panel,.card{background:white;border:1px solid var(--line);border-radius:12px}.filters{padding:15px 18px;margin-bottom:18px;display:flex;gap:18px;flex-wrap:wrap;align-items:end}.filters label,.controls label{font-size:12px;font-weight:650;display:block}select,button{font:inherit;border:1px solid #bdcec4;border-radius:6px;padding:7px 10px;background:white;color:var(--ink)}button{cursor:pointer}button:hover{background:#e5f1e9}select:focus,button:focus{outline:2px solid #469378}#stateChecks{display:flex;flex-wrap:wrap;gap:8px 18px;padding-top:10px}#stateChecks label{font-size:13px;font-weight:400}details{flex:1;min-width:270px}summary{cursor:pointer}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:18px}.card{padding:16px 20px}.card small{color:var(--muted)}.value{font-size:29px;font-weight:650;font-variant-numeric:tabular-nums}.sub{font-size:12px;color:var(--muted)}nav{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px}nav button.active{background:var(--green);color:white;border-color:var(--green)}.panel{padding:20px;margin-bottom:18px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.grid>.panel{min-width:0}.muted{color:var(--muted);font-size:13px;margin:5px 0 14px}.controls{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0}.plot{width:100%;height:440px}.tall{height:590px}.page{display:none}.page.active{display:block}.notice{padding:11px 15px;border-left:4px solid #d8a449;background:#fff8e9;margin-bottom:16px;font-size:13px}.mapgrid{display:grid;grid-template-columns:1fr 1.4fr;gap:16px}.mapgrid svg{width:100%;height:340px;background:#f6f9fa;border-radius:8px}.mapgrid path:hover{stroke:#111;stroke-width:2}.maplegend{display:flex;align-items:center;gap:12px;justify-content:center;margin:8px}.gradient{width:220px;height:12px;border-radius:4px;background:linear-gradient(90deg,#ffffcc,#78c679,#006837)}.scroll{overflow:auto;max-height:480px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:right;padding:9px 12px;border-bottom:1px solid var(--line);white-space:nowrap}th{position:sticky;top:0;background:#edf4ee}th:first-child,td:first-child{text-align:left}#tooltip{position:fixed;pointer-events:none;display:none;background:#123f34;color:white;padding:10px 14px;border-radius:6px;z-index:100;font-size:13px;max-width:300px}#status{font-size:13px;color:var(--muted);margin-bottom:12px}a{color:#166751}footer{font-size:12px;color:var(--muted);padding:6px 0}.unit-table td{white-space:normal;text-align:left}.unit-table th{text-align:left}.check{accent-color:var(--green)}
@media(max-width:900px){.grid,.mapgrid{grid-template-columns:1fr}.cards{grid-template-columns:1fr 1fr}.value{font-size:24px}.panel{padding:14px}h1{font-size:26px}}
</style><script>__PLOTLY__</script></head><body>
<header><div class="eyebrow">Malaysia · Paddy and environment</div><h1>Explore state-level paddy performance</h1><p>Yield, distributions and environmental relationships · Perangkaan units: ha / mt / kg per ha</p></header>
<main><div class="filters"><label>From year<br><select id="start"></select></label><label>To year<br><select id="end"></select></label><details><summary>Choose states <span id="stateCount"></span></summary><div id="stateChecks"></div><button id="allStates">Select all</button> <button id="noStates">Clear</button></details><button id="reset">Reset filters</button></div>
<div id="status"></div><div id="warning" class="notice" hidden></div>
<div class="cards"><div class="card"><small>Latest selected year</small><div class="value" id="kYear"></div><div class="sub" id="kCount"></div></div><div class="card"><small>Area-weighted yield</small><div class="value" id="kYield"></div><div class="sub">kg/ha · latest year, selected states</div></div><div class="card"><small>Paddy production</small><div class="value" id="kProduction"></div><div class="sub">metric tonnes (mt) · latest year</div></div><div class="card"><small>Planted area</small><div class="value" id="kArea"></div><div class="sub">hectares (ha) · latest year</div></div></div>
<nav><button class="active" data-tab="overview">Overview</button><button data-tab="geography">Yield map</button><button data-tab="distributions">Boxplots</button><button data-tab="relationships">Correlations</button><button data-tab="quality">Data quality & units</button></nav>
<section id="overview" class="page active"><div class="panel"><h2>Yield by state and year</h2><p class="muted">Calculated kg/ha. Missing state-years remain blank. Values are not rounded before analysis.</p><div id="yieldHeat" class="plot tall"></div></div><div class="grid"><div class="panel"><h2>Annual trend</h2><div class="controls"><select id="trendVariable"><option value="yield">Area-weighted yield (kg/ha)</option><option value="production">Total production (mt)</option><option value="area">Total planted area (ha)</option></select></div><div id="trend" class="plot"></div></div><div class="panel"><h2>State comparison</h2><p class="muted">Production × 1,000 / planted area summed over the selected period. This is an area-weighted yield, not a simple annual average.</p><div id="ranking" class="plot"></div></div></div></section>
<section id="geography" class="page"><div class="panel"><h2>Where is yield higher?</h2><div class="controls"><label>Map year<br><select id="mapYear"></select></label></div><p class="muted">Yield in kg/ha. One fixed colour scale across all years and states in the CSV supports comparison. Grey means excluded or unavailable. Regional panels are enlarged separately and are not at the same map scale.</p><div class="mapgrid"><div><h3>Peninsular Malaysia</h3><svg id="westMap" viewBox="0 0 520 380" role="img" aria-label="Peninsular Malaysia yield map"></svg></div><div><h3>Sabah & Sarawak</h3><svg id="eastMap" viewBox="0 0 720 380" role="img" aria-label="East Malaysia yield map"></svg></div></div><div class="maplegend"><span id="mapLow"></span><span class="gradient"></span><span id="mapHigh"></span><span>kg/ha</span></div><p class="muted">Hover over a state for yield, area and production. State averages are not paddy-field locations. Boundaries: geoBoundaries / OpenStreetMap contributors; ODbL 1.0 for the inspected cached Malaysia layer.</p><div id="mapRanking" class="plot"></div></div></section>
<section id="distributions" class="page"><div class="panel"><h2>Distributions and unusual observations</h2><div class="controls"><label>Variable<br><select id="boxVariable"></select></label><label>Group by<br><select id="boxGroup"><option value="year">Year (states within each box)</option><option value="state">State (years within each box)</option></select></label></div><p class="muted">Boxes show Q1–Q3; middle line is the median; whiskers extend to the furthest observation within 1.5 × IQR. All observations are shown. Outliers are descriptive flags, not errors. Mean and sample SD are reported below.</p><div id="boxplot" class="plot tall"></div><div id="boxTable" class="scroll"></div></div></section>
<section id="relationships" class="page"><div class="panel"><h2>Relationships across variables</h2><div class="controls"><label>Correlation method<br><select id="method"><option value="pearson">Pearson (linear)</option><option value="spearman">Spearman (rank)</option></select></label><label>Comparison<br><select id="corrMode"><option value="pooled">Pooled state-year observations</option><option value="within">Remove each state's mean (Pearson)</option></select></label><label>Variables<br><select id="corrScope"><option value="environment">Yield + environment</option><option value="all">Include area and production</option></select></label></div><p class="muted" id="corrNote"></p><div id="correlation" class="plot tall"></div><details><summary>Paired sample counts</summary><div id="pairTable" class="scroll"></div></details></div><div class="panel"><h2>Inspect a relationship</h2><div class="controls"><label>Horizontal axis<br><select id="scatterVariable"></select></label></div><p class="muted">One point per state-year. Colour identifies state. Hover for year and values; click legend entries to isolate states.</p><div id="scatter" class="plot tall"></div></div></section>
<section id="quality" class="page"><div class="panel"><h2>Units and definitions</h2><p>Perangkaan Agromakanan Malaysia 2024, Table 2.1.2, printed page 13 (PDF page 27): planted area <b>ha</b>, production <b>mt</b>. Table 2.1.13, printed page 24 (PDF page 38): yield <b>kg/ha</b>. Includes wetland and dryland paddy, main and off-seasons.</p><p>Yield here is <b>production (mt) × 1,000 / planted area (ha)</b>. It is calculated from the CSV, not transcribed from the rounded published yield table. The annual area is cropped area across seasons, not necessarily unique land area.</p><div id="unitNotes"></div><div id="unitTable"></div><p class="muted">GEE definitions follow the supplied harvester and metadata: state-wide means, without a paddy mask. “Median_NDVI” is a legacy name for the spatial mean of annual per-pixel median NDVI. Soil variables are static 0–5 cm estimates. Repeating them across years does not provide independent soil observations.</p><p class="muted">Correlations are exploratory, not causal. Repeated observations of states are dependent. Removing state means does not remove shared year effects. Area and production are mathematically related to calculated yield. No statistical significance is claimed.</p></div><div class="grid"><div class="panel"><h2>Missing values</h2><div id="missingPlot" class="plot"></div></div><div class="panel"><h2>Descriptive statistics</h2><div id="statsTable" class="scroll"></div></div></div></section>
<footer>Source CSV: <span id="sourceName"></span> · Dashboard generated <span id="generated"></span>. Use plot toolbars to save chart images.</footer></main><div id="tooltip"></div>
<script>
const DATA=__DATA__,VARS=__VARS__,BOUNDARIES=__BOUNDARIES__,NOTES=__NOTES__,META=__META__;
const $=id=>document.getElementById(id), finite=x=>typeof x==='number'&&Number.isFinite(x);
const fmt=(x,d=0)=>finite(x)?x.toLocaleString('en-GB',{maximumFractionDigits:d,minimumFractionDigits:d}):'—';
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sum=(rows,key)=>rows.reduce((a,r)=>a+(finite(r[key])?r[key]:0),0);
const mean=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:null;
const quantile=(a,p)=>{if(!a.length)return null;let b=[...a].sort((x,y)=>x-y),n=(b.length-1)*p,i=Math.floor(n);return b[i]+(b[Math.ceil(n)]-b[i])*(n-i)};
const sd=a=>a.length>1?Math.sqrt(a.reduce((s,x)=>s+(x-mean(a))**2,0)/(a.length-1)):null;
const label=k=>VARS[k].name+' ('+VARS[k].unit+')';
const YEARS=[...new Set(DATA.map(r=>r.year))].sort((a,b)=>a-b),STATES=[...new Set(DATA.map(r=>r.state))].sort();
const PALETTE=['#176c55','#497bc2','#c2763d','#8056a0','#ab4d5e','#268b93','#767333','#4c4f98','#b66b99','#41773a','#9e6326','#6d7791','#6a514b'];
const LAYOUT={paper_bgcolor:'white',plot_bgcolor:'white',font:{family:'Segoe UI, Arial',color:'#23453c',size:12},margin:{t:20,b:70,l:85,r:30},hoverlabel:{bgcolor:'#123f34',font:{color:'white'}},xaxis:{gridcolor:'#edf1ed',automargin:true},yaxis:{gridcolor:'#edf1ed',automargin:true}};
const CONFIG={responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',scale:2}};
function plot(id,traces,layout={}){return Plotly.react(id,traces,{...LAYOUT,...layout},CONFIG)}
function options(id,items,selected){$(id).innerHTML=items.map(([v,t])=>`<option value="${esc(v)}">${esc(t)}</option>`).join('');if(selected!==undefined)$(id).value=String(selected)}
function table(id,headers,rows){$(id).innerHTML='<table><thead><tr>'+headers.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+esc(x??'—')+'</td>').join('')+'</tr>').join('')+'</tbody></table>'}
function selectedRows(){let states=[...document.querySelectorAll('#stateChecks input:checked')].map(x=>x.value);return DATA.filter(r=>r.year>=Number($('start').value)&&r.year<=Number($('end').value)&&states.includes(r.state))}
function selectedYears(){return YEARS.filter(y=>y>=Number($('start').value)&&y<=Number($('end').value))}
function selectedStates(){return [...document.querySelectorAll('#stateChecks input:checked')].map(x=>x.value)}
function aggregate(rows){return {area:rows.length?sum(rows,'area'):null,production:rows.length?sum(rows,'production'):null,yield:rows.length?sum(rows,'production')*1000/sum(rows,'area'):null}}
function ranks(a){let out=Array(a.length),s=a.map((x,i)=>({x,i})).sort((a,b)=>a.x-b.x);for(let i=0;i<s.length;){let j=i+1;while(j<s.length&&s[j].x===s[i].x)j++;for(let k=i;k<j;k++)out[s[k].i]=(i+j-1)/2+1;i=j}return out}
function correlate(a,b,method='pearson'){let pairs=a.map((x,i)=>[x,b[i]]).filter(p=>p.every(finite)),n=pairs.length;if(n<3)return {r:null,n};let x=pairs.map(p=>p[0]),y=pairs.map(p=>p[1]);if(method==='spearman'){x=ranks(x);y=ranks(y)}let mx=mean(x),my=mean(y),xx=0,yy=0,xy=0;for(let i=0;i<n;i++){xx+=(x[i]-mx)**2;yy+=(y[i]-my)**2;xy+=(x[i]-mx)*(y[i]-my)}return {r:xx<1e-20||yy<1e-20?null:Math.max(-1,Math.min(1,xy/Math.sqrt(xx*yy))),n}}
function overview(rows){let ys=selectedYears(),ss=selectedStates();let lookup=new Map(rows.map(r=>[r.state+'|'+r.year,r.yield]));let z=ss.map(s=>ys.map(y=>lookup.get(s+'|'+y)??null));plot('yieldHeat',[{type:'heatmap',x:ys.map(String),y:ss,z,zmin:Math.min(...DATA.map(r=>r.yield)),zmax:Math.max(...DATA.map(r=>r.yield)),colorscale:[[0,'#e8f5e9'],[0.25,'#c8e6c9'],[0.5,'#81c784'],[0.75,'#43a047'],[1,'#1b5e20']],texttemplate:'%{z:,.0f}',hoverongaps:false,hovertemplate:'%{y}<br>%{x}: %{z:,.1f} kg/ha<extra></extra>',colorbar:{title:{text:'kg/ha'}}}],{yaxis:{autorange:'reversed',automargin:true},xaxis:{type:'category'},margin:{t:15,l:135,r:60,b:55}});
let metric=$('trendVariable').value,annual=ys.map(y=>aggregate(rows.filter(r=>r.year===y)));plot('trend',[{type:'scatter',mode:'lines+markers',x:ys,y:annual.map(a=>a[metric]),connectgaps:false,line:{color:'#176c55',width:3},marker:{size:8},hovertemplate:'%{x}<br>%{y:,.2f}<extra></extra>'}],{yaxis:{title:{text:label(metric)},automargin:true},xaxis:{dtick:1}});
let rank=ss.map(s=>({state:s,...aggregate(rows.filter(r=>r.state===s))})).filter(r=>finite(r.yield)).sort((a,b)=>a.yield-b.yield);plot('ranking',[{type:'bar',orientation:'h',y:rank.map(r=>r.state),x:rank.map(r=>r.yield),marker:{color:'#38886a'},hovertemplate:'%{y}: %{x:,.1f} kg/ha<extra></extra>'}],{xaxis:{title:{text:'Area-weighted yield (kg/ha)'}},yaxis:{automargin:true},margin:{t:15,l:135,r:20,b:60}})}
function boxes(rows){let key=$('boxVariable').value,group=$('boxGroup').value,groups=group==='year'?selectedYears():selectedStates(),summaries=[];let traces=groups.map((g,i)=>{let part=rows.filter(r=>r[group]===g&&finite(r[key])),values=part.map(r=>r[key]);let q1=quantile(values,.25),q3=quantile(values,.75),iqr=q3-q1; summaries.push([g,values.length,fmt(mean(values),2),fmt(sd(values),2),fmt(quantile(values,.5),2),fmt(q1,2),fmt(q3,2),values.filter(x=>x<q1-1.5*iqr||x>q3+1.5*iqr).length]);return {type:'box',name:String(g),y:values,boxpoints:'all',jitter:.3,pointpos:0,quartilemethod:'linear',boxmean:true,marker:{color:PALETTE[i%PALETTE.length],size:4,opacity:.7},text:part.map(r=>r.state+' · '+r.year),hovertemplate:'%{text}<br>%{y:,.3f}<extra></extra>'}});plot('boxplot',traces,{showlegend:false,yaxis:{title:{text:label(key)},automargin:true},xaxis:{type:'category',tickangle:group==='state'?-35:0}});table('boxTable',[group==='year'?'Year':'State','N','Mean','Sample SD','Median','Q1','Q3','IQR flags'],summaries)}
function correlations(rows){let keys=Object.keys(VARS).filter(k=>$('corrScope').value==='all'||!['area','production'].includes(k));let within=$('corrMode').value==='within',method=within?'pearson':$('method').value;$('method').disabled=within;let matrixRows=rows.map(r=>({...r}));if(within){for(let s of selectedStates()){let group=matrixRows.filter(r=>r.state===s);for(let k of keys){let m=mean(group.map(r=>r[k]).filter(finite));for(let r of group)r[k]=finite(r[k])?r[k]-m:null}}}let counts=[],z=keys.map((a,i)=>{counts[i]=[];return keys.map(b=>{let result=correlate(matrixRows.map(r=>r[a]),matrixRows.map(r=>r[b]),method);counts[i].push(result.n);return result.r})});plot('correlation',[{type:'heatmap',x:keys.map(k=>VARS[k].name),y:keys.map(k=>VARS[k].name),z,customdata:counts,zmin:-1,zmax:1,colorscale:'RdBu',reversescale:true,texttemplate:'%{z:.2f}',hoverongaps:false,hovertemplate:'%{x}<br>%{y}<br>r = %{z:.3f}<br>paired N = %{customdata}<extra></extra>',colorbar:{title:{text:method==='pearson'?'r':'rho'}}}],{margin:{t:15,b:140,l:185,r:40},xaxis:{tickangle:-35},yaxis:{autorange:'reversed'}});table('pairTable',['Paired N',...keys.map(k=>VARS[k].name)],keys.map((k,i)=>[VARS[k].name,...counts[i]]));$('corrNote').textContent=(within?'Pearson correlations of deviations from each state’s selected-period mean. Static soil variables are undefined. Shared year effects remain.':'Pooled correlations mix between-state and within-state relationships. Repeated state observations are not independent.')+' Pairwise-complete observations; minimum N = 3. Blank cells mean insufficient data or no variation. '+($('corrScope').value==='all'?'Area and production are mathematical components of yield.':'');let x=$('scatterVariable').value;plot('scatter',selectedStates().map((s,i)=>{let d=rows.filter(r=>r.state===s&&finite(r[x]));return {type:'scatter',mode:'markers',name:s,x:d.map(r=>r[x]),y:d.map(r=>r.yield),text:d.map(r=>r.state+' · '+r.year),marker:{size:9,color:PALETTE[STATES.indexOf(s)%PALETTE.length],opacity:.8},hovertemplate:'%{text}<br>x: %{x:,.3f}<br>Yield: %{y:,.1f} kg/ha<extra></extra>'}}),{xaxis:{title:{text:label(x)},automargin:true},yaxis:{title:{text:'Yield (kg/ha)'},automargin:true},legend:{orientation:'h',y:-.25},margin:{t:10,l:80,r:15,b:130}})}
function quality(rows){let keys=Object.keys(VARS);plot('missingPlot',[{type:'bar',orientation:'h',y:keys.map(k=>VARS[k].name),x:keys.map(k=>rows.filter(r=>!finite(r[k])).length),marker:{color:'#bd8551'}}],{xaxis:{title:{text:'Missing observations'},dtick:1},margin:{t:10,l:180,r:15,b:60}});table('statsTable',['Variable','N','Mean','SD','Min','Median','Max'],keys.map(k=>{let a=rows.map(r=>r[k]).filter(finite);return [label(k),a.length,fmt(mean(a),2),fmt(sd(a),2),fmt(a.length?Math.min(...a):null,2),fmt(quantile(a,.5),2),fmt(a.length?Math.max(...a):null,2)]}));table('dataTable',['State','Year',...keys.map(label)],rows.map(r=>[r.state,r.year,...keys.map(k=>fmt(r[k],3))]))}
const SVGNS='http://www.w3.org/2000/svg',LOW=Math.min(...DATA.map(r=>r.yield)),HIGH=Math.max(...DATA.map(r=>r.yield));
function colour(value){if(!finite(value))return '#dce2e1';let t=HIGH===LOW?.5:Math.max(0,Math.min(1,(value-LOW)/(HIGH-LOW))),colors=[[255,255,204],[120,198,121],[0,104,55]],i=t<.5?0:1,f=t<.5?t*2:(t-.5)*2;return 'rgb('+colors[i].map((c,j)=>Math.round(c+(colors[i+1][j]-c)*f)).join(',')+')'}
function drawRegion(id,features,width){const svg=$(id);svg.replaceChildren();if(!features.length)return;const merc=p=>[p[0],Math.log(Math.tan(Math.PI/4+p[1]*Math.PI/360))*180/Math.PI];const polys=f=>f.geometry.type==='Polygon'?[f.geometry.coordinates]:f.geometry.coordinates;const pts=features.flatMap(f=>polys(f).flatMap(p=>p.flatMap(r=>r.map(merc))));let xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),scale=Math.min((width-35)/(xmax-xmin),345/(ymax-ymin));const xy=p=>{let q=merc(p);return [(q[0]-xmin)*scale+(width-(xmax-xmin)*scale)/2,(ymax-q[1])*scale+(380-(ymax-ymin)*scale)/2]};for(let f of features){let path=document.createElementNS(SVGNS,'path');path.setAttribute('d',polys(f).map(p=>p.map(r=>r.map((point,i)=>(i?'L':'M')+xy(point).map(v=>v.toFixed(2)).join(',')).join('')+'Z').join('')).join(''));path.setAttribute('fill-rule','evenodd');path.setAttribute('stroke','#607d71');path.setAttribute('stroke-width','.7');path.dataset.key=f.key;path.dataset.name=f.name;path.setAttribute('tabindex','0');path.addEventListener('mousemove',e=>{const tip=$('tooltip');tip.innerHTML=path.dataset.tip;tip.style.display='block';tip.style.left=Math.min(e.clientX+12,window.innerWidth-310)+'px';tip.style.top=Math.max(5,e.clientY-70)+'px'});path.addEventListener('mouseleave',()=>$('tooltip').style.display='none');svg.appendChild(path)}}
function maps(rows){let old=$('mapYear').value,ys=selectedYears();options('mapYear',ys.map(y=>[y,y]),ys.includes(Number(old))?old:ys.at(-1));let year=Number($('mapYear').value),part=rows.filter(r=>r.year===year),lookup=new Map(part.map(r=>[r.key,r]));for(let path of document.querySelectorAll('.mapgrid path')){let r=lookup.get(path.dataset.key);path.setAttribute('fill',colour(r?.yield));path.dataset.tip='<b>'+esc(path.dataset.name)+'</b><br>'+year+' · '+(r?fmt(r.yield,1)+' kg/ha<br>'+fmt(r.production)+' mt · '+fmt(r.area)+' ha':'No selected data');path.setAttribute('aria-label',path.dataset.name+': '+(r?fmt(r.yield,1)+' kg/ha':'No selected data'));let title=path.querySelector('title')||document.createElementNS(SVGNS,'title');title.textContent=path.getAttribute('aria-label');path.appendChild(title)}$('mapLow').textContent=fmt(LOW);$('mapHigh').textContent=fmt(HIGH);part.sort((a,b)=>a.yield-b.yield);plot('mapRanking',[{type:'bar',orientation:'h',x:part.map(r=>r.yield),y:part.map(r=>r.state),marker:{color:part.map(r=>colour(r.yield))},hovertemplate:'%{y}: %{x:,.1f} kg/ha<extra></extra>'}],{xaxis:{title:{text:'Yield (kg/ha) · '+year}},margin:{l:140,r:20,t:15,b:55}})}
let active='overview';function render(){let rows=selectedRows(),ss=selectedStates(),ys=selectedYears(),last=ys.at(-1),current=rows.filter(r=>r.year===last),a=aggregate(current);$('stateCount').textContent='('+ss.length+'/'+STATES.length+')';$('status').textContent=rows.length+' state-year observations · '+ss.length+' selected states · '+(ys.length?ys[0]+'–'+last:'no years');$('kYear').textContent=last??'—';$('kCount').textContent=current.length+' states reporting';$('kYield').textContent=fmt(a.yield);$('kProduction').textContent=current.length?fmt(a.production):'—';$('kArea').textContent=current.length?fmt(a.area):'—';let gaps=ss.length*ys.length-rows.length;$('warning').hidden=rows.length>0&&gaps===0;$('warning').textContent=!rows.length?'No observations. Select at least one state and a valid year range.':gaps+' selected state-year records are missing. Annual totals may not be directly comparable.';if(active==='overview')overview(rows);if(active==='geography')maps(rows);if(active==='distributions')boxes(rows);if(active==='relationships')correlations(rows);if(active==='quality')quality(rows)}
function reset(){ $('start').value=YEARS.find(y=>y>=META.defaultStart)??YEARS[0];$('end').value=YEARS.at(-1);document.querySelectorAll('#stateChecks input').forEach(x=>x.checked=true);render() }
options('start',YEARS.map(y=>[y,y]));options('end',YEARS.map(y=>[y,y]));$('stateChecks').innerHTML=STATES.map(s=>'<label><input class="check" type="checkbox" checked value="'+esc(s)+'"> '+esc(s)+'</label>').join('');options('boxVariable',Object.keys(VARS).map(k=>[k,label(k)]),'yield');options('scatterVariable',Object.keys(VARS).filter(k=>k!=='yield').map(k=>[k,label(k)]),VARS.ndvi?'ndvi':'area');$('unitNotes').innerHTML=NOTES.map(n=>'<p class="notice">'+esc(n)+'</p>').join('');table('unitTable',['Dashboard variable','Unit'],Object.keys(VARS).map(k=>[VARS[k].name,VARS[k].unit]));$('sourceName').textContent=META.source;$('generated').textContent=META.generated;
drawRegion('westMap',BOUNDARIES.filter(f=>!['sabah','sarawak','labuan','w.p. labuan','wilayah persekutuan labuan'].includes(f.key)),520);drawRegion('eastMap',BOUNDARIES.filter(f=>['sabah','sarawak','labuan','w.p. labuan','wilayah persekutuan labuan'].includes(f.key)),720);
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{active=b.dataset.tab;document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('.page').forEach(p=>p.classList.toggle('active',p.id===active));render()});document.querySelectorAll('select').forEach(s=>s.onchange=()=>{if(Number($('start').value)>Number($('end').value)){if(s.id==='start')$('end').value=$('start').value;else $('start').value=$('end').value}render()});$('stateChecks').onchange=render;$('reset').onclick=reset;$('allStates').onclick=()=>{document.querySelectorAll('#stateChecks input').forEach(x=>x.checked=true);render()};$('noStates').onclick=()=>{document.querySelectorAll('#stateChecks input').forEach(x=>x.checked=false);render()};
window.addEventListener('error',e=>{$('warning').hidden=false;$('warning').textContent='Dashboard error: '+e.message});reset();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--boundaries", type=Path, default=BOUNDARY_PATH)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data, variables, notes = load_data(args.csv)
    boundaries = load_boundaries(args.output, args.boundaries)
    unmatched = {r["key"] for r in data} - {f["key"] for f in boundaries}
    if unmatched:
        raise ValueError(f"States not matched to boundaries: {sorted(unmatched)}. Add aliases in state_key().")
    from datetime import datetime
    replacements = {"__DATA__": data, "__VARS__": variables, "__BOUNDARIES__": boundaries,
                    "__NOTES__": notes, "__META__": {"source": args.csv.name,
                    "generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "defaultStart": DEFAULT_START_YEAR}}
    html = HTML
    for token, value in replacements.items():
        html = html.replace(token, json.dumps(value, ensure_ascii=False, allow_nan=False).replace("</", "<\\/"))
    html = html.replace("__PLOTLY__", get_plotlyjs())
    destination = args.output / "index.html"
    destination.write_text(html, encoding="utf-8")
    print(f"Loaded {len(data)} records. Units: area ha; production mt; yield kg/ha.")
    for note in notes:
        print(note)
    print(f"\nDASHBOARD READY:\n{destination.resolve()}")
    if not args.no_open:
        webbrowser.open(destination.resolve().as_uri())


if __name__ == "__main__":
    main()
