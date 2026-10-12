/* Offline unit tests: execute dashboard logic against a minimal DOM and Plotly stub.
   This does not launch or control a browser and does not test visual rendering. */
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(path.join(__dirname,'../reports/dashboard.html'),'utf8');
const payload = html.split('<script id="project-data" type="application/json">')[1].split('</script>')[0];
const script = html.slice(html.lastIndexOf('<script>')+8,html.lastIndexOf('</script>'));
const nodes = {}, elements = [], graphs = {}, downloads = [];
class Element {
  constructor(tag='div'){this.tagName=tag;this.children=[];this.dataset={};this.style={};this.handlers={};this.hidden=false;this.checked=false;this.textContent='';this.value='';elements.push(this);}
  append(...items){this.children.push(...items);} appendChild(item){this.children.push(item);return item;}
  replaceChildren(...items){this.children=items;} setAttribute(name,value){this[name]=value;}
  addEventListener(name,handler){this.handlers[name]=handler;}
  click(){if(this.handlers.click)this.handlers.click();}
  set innerHTML(value){this.children=[];} get innerHTML(){return '';}
}
const el=id=>nodes[id]||(nodes[id]=Object.assign(new Element(),{id}));
el('project-data').textContent=payload;
Object.entries({source:'GCAG',phase:'test',horizon:'12',metric:'MAE','future-months':'12','interval-level':'0.9'}).forEach(([id,value])=>el(id).value=value);
el('show-interval').checked=true;
for(const name of ['explore','models','forecast','methods']){el(name).className='panel';el('tab-'+name).dataset.tab=name;}
const context=vm.createContext({console,Blob,Set,Map,JSON,Math,Promise,Array,Object,String,document:{
  getElementById:el,createElement:tag=>new Element(tag),createTextNode:text=>({textContent:text}),
  querySelectorAll:selector=>elements.filter(e=>selector==='[data-model]'?e.dataset.model:selector==='[data-tab]'?e.dataset.tab:selector==='.panel'?e.className==='panel':false)
},Plotly:{react:(id,data,layout)=>{graphs[id]={data:JSON.parse(JSON.stringify(data)),layout};return Promise.resolve();}},
URL:{createObjectURL:blob=>{downloads.push(blob);return 'blob:offline-test';},revokeObjectURL:()=>{}},setTimeout:fn=>fn()});
const run=s=>vm.runInContext(s,context);
let checks=0;
function check(condition,message){assert.ok(condition,message);checks++;}
(async()=>{
  run(script);await run('renderActive()');
  check(Object.keys(graphs).filter(k=>k.startsWith('eda-')).length===11,'All eleven EDA figures render through plotting API');
  check(el('render-status').textContent.startsWith('Ready'),'EDA reaches ready state');
  run("active='models'");await run('renderActive()');
  check(graphs['comparison-plot'].data.length===4,'Default overlay = observations + 3 selected models');
  check(el('selection-count').textContent==='3 of 10 models selected','Visible selection count');
  run('selected.clear();candidates.forEach(m=>selected.add(m));updateSelection()');await run('renderActive()');
  check(graphs['comparison-plot'].data.length===11,'All 10 models overlay');
  check(graphs['ranking-plot'].data[0].y.length===10,'Ranking includes all ten');
  let rows=run('metricsRows');const data=JSON.parse(payload);
  for(const r of rows){const original=data.predictions.filter(p=>p.source==='GCAG'&&p.phase==='test'&&p.model===r.model);const mae=original.reduce((s,p)=>s+Math.abs(p.actual-p.prediction),0)/original.length;check(Math.abs(mae-r.MAE)<1e-12,'Metric recomputed correctly for '+r.model);}
  el('source').value='GISTEMP';el('horizon').value='3';await run('renderActive()');
  check(run('metricsRows.every(r=>r.n===15)'),'Horizon filter includes 5 origins × 3 horizons');
  check(graphs['horizon-plot'].data.every(t=>t.x.length===3),'Horizon plot applies filter');
  check(run('selected.size')===10,'Source change retains selected models');
  el('phase').value='selection';await run('renderActive()');
  check(run('metricsRows.every(r=>r.n===45)'),'Selection phase includes 15 origins × 3 horizons');
  el('metric').value='RMSE';await run('renderActive()');
  const ranks=graphs['ranking-plot'].data[0].x;check(ranks.every((x,i)=>i===0||x>=ranks[i-1]),'RMSE rank is sorted');
  el('download-metrics').click();const csv=await downloads.at(-1).text();check(csv.includes('"GISTEMP","selection","3"'),'Export reflects source/phase/horizon filters');
  run('selected.clear();updateSelection()');await run('renderActive()');
  check(!el('no-models').hidden,'Empty selection has explicit guidance');
  check(graphs['comparison-plot'].data.length===1,'Actual series remains for empty selection');
  run("active='forecast';selected.add('ridge_1');selected.add('arima_111');updateSelection()");await run('renderActive()');
  check(graphs['future-plot'].data.length===5,'Future overlay = actual + band pair + two models');
  el('future-months').value='3';el('interval-level').value='0.8';await run('renderActive()');
  check(run('futureRows.length')===6,'Future horizon applies to both models');
  check(graphs['future-plot'].data.some(t=>t.name?.startsWith('80% interval')),'80% interval control updates band');
  el('future-months').value='1';await run('renderActive()');
  check(graphs['future-plot'].data.find(t=>t.name==='Ridge α=1').mode==='lines+markers','Single-month forecast is a visible point');
  check(graphs['future-plot'].data.find(t=>t.name==='Ridge α=1').error_y.visible,'Single-month interval has an error bar');
  el('future-months').value='3';await run('renderActive()');
  run("selected.delete('ridge_1');updateSelection()");await run('renderActive()');
  check(graphs['future-plot'].data.length===2,'No winner means no fabricated interval for other models');
  check(el('forecast-context').textContent.includes('point forecasts only'),'Interval limitation explained');
  el('download-forecast').click();const out=await downloads.at(-1).text();check(out.split('\r\n').length===4,'Forecast CSV contains header plus 3 selected rows');
  el('default-models').click();await run('renderActive()');check(run('selected.size')===3,'Suggested comparison button restores 3 models');
  el('all-models').click();await run('renderActive()');check(run('selected.size')===10,'Select all button works');
  el('clear-models').click();await run('renderActive()');check(run('selected.size')===0,'Clear selection button works');
  console.log(`${checks} offline dashboard assertions passed (logic/data only, not browser rendering).`);
})().catch(error=>{console.error(error);process.exitCode=1;});
