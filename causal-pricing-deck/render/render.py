"""Approximate local renderer for the Slides-type deck: wraps each slide <section> in a page with
the deck's fonts (local woff2), maps /_blob/<id> to local PNGs, draws <x-connector> as SVG, and
screenshots each slide at 1920x1080. Build steps (data-build-in) can be rendered as separate frames.
Usage: python3 render/render.py <deck_dir> <out_dir> [--builds]"""
import json, os, re, sys, base64, pathlib
from playwright.sync_api import sync_playwright

BASE = pathlib.Path(__file__).resolve().parent.parent
deck_dir, out_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
BUILDS = "--builds" in sys.argv
out_dir.mkdir(parents=True, exist_ok=True)
blobmap = json.load(open(BASE / "render/blobmap.json"))
deck = json.load(open(deck_dir / "project/deck.json"))

def uri(p):
    return "data:image/png;base64," + base64.b64encode(open(BASE / p, "rb").read()).decode()

faces = []
for fam, pre, specs in [("IBM Plex Sans", "ibm-plex-sans", [("400", "normal"), ("500", "normal"), ("600", "normal")]),
                        ("Source Serif 4", "source-serif-4", [("400", "normal"), ("600", "normal"), ("400", "italic")])]:
    for w, st in specs:
        for sub in ["latin", "greek"]:
            f = BASE / f"render/fonts/{pre}-{sub}-{w}-{st}.woff2"
            if f.exists():
                b = base64.b64encode(open(f, "rb").read()).decode()
                faces.append(f"@font-face{{font-family:'{fam}';font-weight:{w};font-style:{st};src:url(data:font/woff2;base64,{b}) format('woff2')}}")
CSS = "\n".join(faces) + """
html,body{margin:0;padding:0;background:#888}
section{width:1920px;height:1080px;position:relative;box-sizing:border-box;overflow:hidden}
section *{box-sizing:border-box}
h1,h2,h3,p,ol,ul,table{margin:0}
h1{font-size:96px;font-weight:600;line-height:1.1} h2{font-size:64px;font-weight:600;line-height:1.15}
h3{font-size:44px;font-weight:600;line-height:1.2} p{font-size:32px;line-height:1.4}
ol,ul{padding-left:1.3em} li{margin:0}
table{border-collapse:collapse} td,th{border:1px solid #d9d9d2;padding:.35em .6em;text-align:left;vertical-align:top}
img{display:block}
aside{display:none}
x-connector{display:none}
"""
JS = """
() => {
  document.querySelectorAll('x-connector').forEach(c => { if (c.style.visibility==='hidden') return;
    const host = c.parentElement; const ns='http://www.w3.org/2000/svg';
    const svg=document.createElementNS(ns,'svg'); svg.style.position='absolute'; svg.style.left='0'; svg.style.top='0';
    svg.setAttribute('width', host.offsetWidth); svg.setAttribute('height', host.offsetHeight); svg.style.overflow='visible';
    const col = c.style.color || '#333'; const w = parseFloat(c.style.borderWidth||'2');
    const x1=+c.getAttribute('x1'), y1=+c.getAttribute('y1'), x2=+c.getAttribute('x2'), y2=+c.getAttribute('y2');
    const l=document.createElementNS(ns,'line'); l.setAttribute('x1',x1); l.setAttribute('y1',y1); l.setAttribute('x2',x2); l.setAttribute('y2',y2);
    l.setAttribute('stroke',col); l.setAttribute('stroke-width',w); l.setAttribute('stroke-linecap','round'); svg.appendChild(l);
    if ((c.getAttribute('head')||'end')!=='none'){ const a=Math.atan2(y2-y1,x2-x1), s=4*w;
      const p=document.createElementNS(ns,'path');
      p.setAttribute('d',`M${x2},${y2} L${x2-s*Math.cos(a-0.45)},${y2-s*Math.sin(a-0.45)} M${x2},${y2} L${x2-s*Math.cos(a+0.45)},${y2-s*Math.sin(a+0.45)}`);
      p.setAttribute('stroke',col); p.setAttribute('stroke-width',w); p.setAttribute('stroke-linecap','round'); p.setAttribute('fill','none'); svg.appendChild(p);}
    host.appendChild(svg);
  });
}
"""
def build_orders(html):
    return sorted({int(m) for m in re.findall(r'data-build-in="[a-z]+\s+(\d+)', html)})

with sync_playwright() as pw:
    br = pw.chromium.launch()
    pg = br.new_page(viewport={"width": 1920, "height": 1080})
    for i, sid in enumerate(deck["order"], 1):
        html = open(deck_dir / f"project/slides/{sid}.html").read()
        html = re.sub(r'/_blob/([0-9a-f]{32})', lambda m: uri(blobmap[m.group(1)]), html)
        steps = [None]
        if BUILDS:
            steps = build_orders(html) or [None]
            steps = [0] + steps if steps != [None] else [None]
        for st in steps:
            h = html
            if st is not None:
                # hide build elements whose order is greater than the current step
                def hide(m):
                    tag, order = m.group(0), int(m.group(1))
                    return tag.replace('style="', 'style="visibility:hidden; ', 1) if order > st else tag
                h = re.sub(r'<[a-z0-9-]+[^>]*data-build-in="[a-z]+\s+(\d+)[^>]*>', hide, h)
            pg.set_content(f"<html><head><style>{CSS}</style></head><body>{h}</body></html>")
            pg.evaluate("document.fonts.ready")
            pg.evaluate(JS)
            name = f"slide{i}_{sid}" + ("" if st is None else f"_step{st}")
            pg.locator("section").screenshot(path=str(out_dir / f"{name}.png"))
            # overflow report: elements extending beyond section or below y=920 (except pinned footer)
            rep = pg.evaluate("""() => { const s=document.querySelector('section').getBoundingClientRect(); const out=[];
              document.querySelectorAll('section *').forEach(e=>{ if(e.tagName==='ASIDE'||e.closest('aside')) return;
                const r=e.getBoundingClientRect(); if(!r.width) return;
                const pinnedFooter = getComputedStyle(e).position==='absolute' && e.style.bottom==='64px';
                if(!pinnedFooter && (r.bottom-s.top>922 || r.right-s.left>1794)) out.push(e.tagName+':'+Math.round(r.bottom-s.top)+'/'+Math.round(r.right-s.left)+':'+(e.textContent||'').trim().slice(0,40));
                if(e.scrollHeight>e.clientHeight+8 && ['P','H1','H2','H3','LI','TD'].includes(e.tagName)) out.push('overflow-y '+e.tagName+':'+(e.textContent||'').trim().slice(0,40)); });
              return out; }""")
            print(name, "OK" if not rep else rep[:6])
    br.close()
