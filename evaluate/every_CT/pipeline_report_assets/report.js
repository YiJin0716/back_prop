const R = DATA.report;
const number = (v, digits=2) => v === null || v === undefined ? '—' : Number(v).toFixed(digits);
const percent = v => v === null || v === undefined ? '—' : v>0 && v<.001 ? '<0.1%' : (100*v).toFixed(1)+'%';
const label = v => v === null || v === undefined ? 'Indeterminate' : v ? 'Positive' : 'Negative';
const cap = s => s.charAt(0).toUpperCase()+s.slice(1);
function node(tag, text, cls='') { const n=document.createElement(tag); n.textContent=text; n.className=cls; return n; }
function table(headers, rows) {
  const t=node('table','','report-table'), head=document.createElement('thead'), tr=document.createElement('tr');
  headers.forEach(h=>tr.appendChild(node('th',h))); head.appendChild(tr); t.appendChild(head);
  const body=document.createElement('tbody');
  rows.forEach(values=>{const r=document.createElement('tr'); values.forEach(value=>{
    const td=document.createElement('td'); if(value instanceof Node)td.appendChild(value); else td.textContent=value;
    r.appendChild(td); }); body.appendChild(r); }); t.appendChild(body); return t;
}
function nameOf(row) { return row.nodule_id === null ? `Query ${row.query_id}` : `Nodule ${row.nodule_id}`; }
function badge(row) {return node('span',row.status,'badge '+(row.status==='Missed nodule'?'missed':row.status==='Unmatched prediction'?'extra':''));}
function reportNavigation() {
  $('report-title').textContent=R.case_id;
  const add=(id,title,subtitle)=>{
    const card=node('div','','row'); card.dataset.id=id; card.tabIndex=0; card.setAttribute('role','button');
    card.append(node('b',title),node('small',subtitle));
    card.onclick=()=>show(id); card.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();show(id);}};
    $('list').appendChild(card); return card;
  };
  add('scan','Whole CT',`${R.predicted} predicted · ${R.official} official nodules`);
  DATA.rows.forEach(row=>{
    const card=add(row.id,nameOf(row),row.status+(row.query_id===null?'':` · query ${row.query_id}`));
    if(!row.has_scene)card.onclick=()=>renderReport(row.id);
  });
}
function renderReport(id) {
  const panel=$('report-panel'); panel.replaceChildren(); panel.scrollTop=0;
  const row=DATA.rows.find(r=>r.id===id);
  if(!row) {renderOverview(panel); return;}
  const heading=node('div','','panel-heading');heading.append(node('h2',nameOf(row)),badge(row));panel.append(heading);
  if(row.center_mm){const b=node('button','Locate in whole CT','jump');b.onclick=()=>show('scan',row.center_mm);panel.append(b);}
  const risk=node('div','','risk-pair');
  const pred=node('div','','risk-card');pred.append(node('span','MODEL RISK'),node('b',percent(row.risk)));
  const gt=node('div','','risk-card official');gt.append(node('span','OFFICIAL MALIGNANCY'),node('b',row.official_mean?number(row.official_mean.malignancy)+' / 5':'—'));
  risk.append(pred,gt);panel.append(risk);
  panel.append(node('h3','Semantic features'));
  panel.append(table(['Feature','Model','Reader mean'],R.semantic_names.map(n=>[
    cap(n),number(row.semantic?.[n]),number(row.official_mean?.[n])])));
  panel.append(node('h3','Segmentation agreement'));
  panel.append(table(['Reference mask','Dice','IoU'],[
    [row.closest_annotation_id===null?'Closest reader':`Closest reader · ann. ${row.closest_annotation_id}`,
      number(row.overlap_closest?.dice,3),number(row.overlap_closest?.iou,3)],
    ['Official consensus',number(row.overlap_consensus?.dice,3),number(row.overlap_consensus?.iou,3)]]));
}
function renderOverview(panel) {
  panel.append(node('h2','Whole CT'));
  const counts=['Matched','Missed nodule','Unmatched prediction'].map(s=>DATA.rows.filter(r=>r.status===s).length);
  const stats=node('div','','stats');counts.forEach((v,k)=>{const d=node('div','','stat');d.append(node('b',v),node('small',['Matched','Missed','Unmatched'][k]));stats.append(d);});panel.append(stats);
  const risk=node('div','','risk-pair');const d=node('div','','risk-card');d.append(node('span','MODEL · SCAN RISK'),node('b',percent(R.scan_risk)));
  const g=node('div','','risk-card official');g.append(node('span','OFFICIAL · SCAN LABEL'),node('b',label(R.official_scan_labels.exclude_3)));
  risk.append(d,g);panel.append(risk);
  panel.append(node('h3','All nodules and predictions'));
  panel.append(table(['Nodule / query','Status','Risk','GT malignancy'],DATA.rows.map(row=>{
    const b=node('button',nameOf(row));b.onclick=()=>row.has_scene?show(row.id):renderReport(row.id);
    return [b,row.status==='Unmatched prediction'?'Unmatched':row.status==='Missed nodule'?'Missed':'Matched',percent(row.risk),row.official_mean?number(row.official_mean.malignancy):'—'];
  })));
}
