"use strict";

// Recorded scenario data are the sole source of demonstration portfolio impacts.
const MARKET_URL = "./market_scenarios_20261025.json";
const PRIMARY_EXAMPLE_ID = "technology_and_treasury";
const SAMPLE = [
  {symbol:"NVDA",weight:.20},
  {symbol:"QQQ",weight:.50},
  {symbol:"TLT",weight:.30}
];
const appRoot = document.getElementById("app");
const portfolioDialog = document.getElementById("portfolio-dialog");
const toast = document.getElementById("toast");
const state = {
  snapshots:null, events:[], holdings:SAMPLE.map(h=>({...h})), notional:250000,
  activeExample:PRIMARY_EXAMPLE_ID, sample:true, selected:null, details:false, showOther:false,
  selectedContract:null, selectedOutcomeId:null, jobStatus:"", jobError:"",
  draft:SAMPLE.map(h=>({symbol:h.symbol,value:h.weight*100})),
  mode:"weight", editorTab:"manual", csvName:"", validation:""
};
let toastTimer;

function esc(value){
  return String(value == null ? "" : value).replace(/[&<>"']/g,function(ch){
    return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch];
  });
}
function pct(number,digits){
  if(!Number.isFinite(number))return "—";
  return (number*100).toFixed(digits == null ? 0 : digits)+"%";
}
function money(number,compact){
  if(!Number.isFinite(number))return "No estimate";
  const sign=number<0?"−":number>0?"+":"";
  const amount=Math.abs(number);
  const text=compact&&amount>=1000
    ?(amount/1000).toLocaleString("en-US",{maximumFractionDigits:1})+"k"
    :amount.toLocaleString("en-US",{maximumFractionDigits:0});
  return sign+"$"+text;
}
function moneyPlain(number){
  return Number(number).toLocaleString("en-US",{style:"currency",currency:"USD",maximumFractionDigits:0});
}
function prettyDate(iso){
  const date=new Date(String(iso)+"T12:00:00Z");
  return Number.isNaN(date.getTime())?"Date pending":
    new Intl.DateTimeFormat("en-US",{month:"short",day:"numeric",year:"numeric",timeZone:"UTC"}).format(date);
}
function shortDate(iso){
  const date=new Date(String(iso)+"T12:00:00Z");
  return Number.isNaN(date.getTime())?"Date pending":
    new Intl.DateTimeFormat("en-US",{month:"short",day:"numeric",timeZone:"UTC"}).format(date);
}
function selectedOption(event){
  if(!event || event.kind!=="market")return null;
  return event.outcomes.find(o=>o.id===state.selectedOutcomeId)||
    event.outcomes.find(o=>o.contract_id===state.selectedContract && o.id.endsWith(":YES"))||
    event.outcomes[0]||null;
}
function impactFor(event){
  if(!event)return null;
  if(event.kind==="market"){
    const outcome=state.selected===event.id?selectedOption(event):event.outcomes[0];
    return Number.isFinite(outcome?.pnl)?outcome.pnl:null;
  }
  return Number.isFinite(event.impact)?event.impact*state.notional/100000:null;
}
function eventsFromSnapshot(snapshot,sourceDate){
  const now=new Date().toISOString().slice(0,10);
  return (snapshot.risks||[]).filter(r=>r.date>=now&&r.outcomes?.some(o=>Number.isFinite(o.pnl))).map(r=>({
    id:r.id,name:r.title,deadline:r.date,condition:r.original_question,
    trigger:r.broad_definition,kind:"market",outcomes:r.outcomes,
    exposure:r.exposure||[],quality:r.quality||{},modelStatus:r.model_status,
    sourceDate:sourceDate||"2026-10-25"
  })).sort((a,b)=>Math.max(...b.outcomes.map(o=>Math.abs(o.pnl||0)))-
    Math.max(...a.outcomes.map(o=>Math.abs(o.pnl||0))));
}
function selectedEvent(){
  return state.events.find(e=>e.id===state.selected)||state.events[0];
}
function adjustedImpact(event){
  return impactFor(event);
}
function selectEvent(id){
  state.selected=id;
  const event=selectedEvent();
  state.selectedOutcomeId=null;
  state.selectedContract=event?.kind==="market"?event.outcomes[0]?.contract_id||null:null;
  state.details=false;
}
function chooseExample(id){
  const sample=state.snapshots?.cases.find(c=>c.id===id);
  if(!sample)return;
  state.activeExample=id;
  state.sample=true;
  state.jobStatus="";
  state.jobError="";
  state.notional=sample.notional;
  state.holdings=sample.positions.map(p=>({symbol:p.symbol,weight:p.weight}));
  state.events=eventsFromSnapshot(sample,state.snapshots.snapshot_date);
  selectEvent(state.events[0]?.id);
  state.showOther=false;
  render();
}
function samePortfolio(rows){
  if(rows.length!==SAMPLE.length)return false;
  return SAMPLE.every(function(h){
    const row=rows.find(r=>r.symbol===h.symbol);
    return !!row&&Math.abs(row.weight-h.weight)<0.00001;
  });
}
function showToast(message){
  clearTimeout(toastTimer);
  toast.textContent=message;
  toast.classList.add("visible");
  toastTimer=setTimeout(()=>toast.classList.remove("visible"),3700);
}
function htmlPortfolio(){
  return '<section class="portfolio-strip" aria-label="Your portfolio">'+
    '<div class="portfolio-brief"><span>Your portfolio</span><strong>'+moneyPlain(state.notional)+'</strong></div>'+
    '<div class="portfolio-pills">'+state.holdings.map(function(h){
      return '<span class="holding-pill"><strong>'+esc(h.symbol)+'</strong> '+pct(h.weight,0)+'</span>';
    }).join("")+'</div>'+
    '<div class="example-picker"><label for="example-select">Portfolio</label>'+
      '<select id="example-select" aria-label="Choose a research portfolio">'+
                (state.snapshots?.cases||[]).map(c=>'<option value="'+esc(c.id)+'" '+(state.activeExample===c.id?"selected":"")+'>'+esc(c.label)+'</option>').join("")+
        (state.activeExample==="custom"||state.activeExample==="live"?
          '<option value="'+esc(state.activeExample)+'" selected>Your holdings</option>':"")+
      '</select></div>'+
  '</section>';
}
function htmlEventItem(event,index){
  const impact=adjustedImpact(event),isActive=state.selected===event.id;
  const hint=event.kind==="model"?pct(event.probability,0)+" chance":"Scenario impact";
  return '<button class="risk-event '+(isActive?"selected":"")+'" type="button" data-action="event" data-id="'+esc(event.id)+'"'+
    ' aria-pressed="'+String(isActive)+'" aria-label="'+esc(event.name)+'; '+esc(hint)+'; portfolio effect '+esc(money(impact,true))+'">'+
    '<span class="risk-name">'+esc(event.name)+'</span>'+
    '<span class="risk-meta"><span>'+shortDate(event.deadline)+' · '+(event.kind==="model"?'<strong>'+pct(event.probability,0)+'</strong> chance':'Recorded scenario')+'</span>'+
    '<strong class="risk-effect '+(!Number.isFinite(impact)?"pending":impact<0?"negative":impact>0?"positive":"neutral")+'">'+
      (Number.isFinite(impact)?money(impact,true):"Impact pending")+'</strong></span></button>';
}
function htmlEventList(){
  const complete=state.events.filter(e=>Number.isFinite(adjustedImpact(e)));
  const other=state.events.filter(e=>!Number.isFinite(adjustedImpact(e)));
  return '<aside class="event-rail" aria-label="Events to watch">'+
    '<div class="rail-header"><div><h2>Events to watch</h2><p>Select an event to see its effect.</p></div>'+
    '<span class="rail-count">'+complete.length+'</span></div>'+
    '<div class="event-scroll" id="event-scroll" role="group" aria-label="Events ranked by portfolio effect">'+
      complete.map(htmlEventItem).join("")+'</div>'+
    (other.length?'<div class="rail-other">'+
      '<button class="other-toggle" type="button" data-action="other" aria-expanded="'+String(state.showOther)+'">'+
        (state.showOther?'Hide':'See')+' '+other.length+' more event'+(other.length===1?'':'s')+
        ' <span aria-hidden="true">'+(state.showOther?'−':'+')+'</span></button>'+
      (state.showOther?'<div class="other-events">'+other.map(htmlEventItem).join("")+'</div>':'')+
    '</div>':'')+
  '</aside>';
}
function methodRow(label,content){
  return '<div class="method-row"><span class="method-key">'+esc(label)+'</span><p>'+esc(content)+'</p></div>';
}
function htmlMethod(event){
  const names=event.exposure.slice(0,5).map(x=>x.symbol+" "+pct(x.weight,1)).join(" · ");
  const hasContract=event.outcomes.some(o=>Boolean(o.contract_id));
  return '<div class="method-content" id="event-method" aria-label="Definitions and calculation details">'+
    methodRow("Event rule",event.trigger||event.condition)+
    methodRow("Exposed holdings",names||"Not reported for this event.")+
    methodRow("Probability evidence",hasContract
      ?"Recorded prediction-market quote for the selected contract and settlement rule."
      :"Recorded scenario weights for the named economic outcomes.")+
    methodRow("Portfolio model","Conditional portfolio response under the selected outcome and scenario inputs.")+
    '<p class="method-asof">Dated research snapshot · '+esc(event.sourceDate)+'.</p></div>';
}
function htmlMarketControls(event){
  const yesOptions=event.outcomes.filter(o=>o.id.endsWith(":YES")&&o.contract_id);
  if(yesOptions.length){
    const active=event.outcomes.find(o=>o.contract_id===state.selectedContract)||yesOptions[0];
    const activeId=active.contract_id;
    const selected=selectedOption(event);
    return '<div class="scenario-controls">'+
      '<label for="contract-select">Market question</label>'+
      '<select id="contract-select" aria-label="Choose the exact market question">'+yesOptions.map(o=>
        '<option value="'+esc(o.contract_id)+'" '+(o.contract_id===activeId?"selected":"")+'>'+esc(o.question||o.label.replace(/^YES:\s*/,""))+'</option>'
      ).join("")+'</select>'+
      '<div class="outcome-row"><span>If the answer is</span><div class="outcome-switch" role="group" aria-label="Event outcome">'+
        ['YES','NO'].map(side=>{
          const available=event.outcomes.find(o=>o.contract_id===activeId&&o.id.endsWith(":"+side));
          return '<button type="button" data-action="outcome" data-id="'+esc(available?.id||"")+'" '+
            (available?'':'disabled ')+'aria-pressed="'+String(selected?.id===available?.id)+'" class="'+(selected?.id===available?.id?"active ":"")+
            (side==="YES"?"yes-outcome":"no-outcome")+'">'+side+'</button>';
        }).join("")+
      '</div></div></div>';
  }
  return '<div class="scenario-controls"><label for="outcome-select">Choose an outcome</label>'+
    '<select id="outcome-select" aria-label="Choose the modeled event outcome">'+
      event.outcomes.map(o=>'<option value="'+esc(o.id)+'" '+(selectedOption(event)?.id===o.id?"selected":"")+'>'+esc(o.label)+'</option>').join("")+
    '</select></div>';
}
function htmlDollarRange(range,central){
  if(!Array.isArray(range)||range.length!==2||!range.every(Number.isFinite)){
    return '<div class="range-unavailable">No dollar interval published for this estimate.</div>';
  }
  const low=Math.min(range[0],range[1]), high=Math.max(range[0],range[1]);
  const color=central<0?"negative":central>0?"positive":"neutral";
  const pos=high===low?50:Math.max(0,Math.min(100,(central-low)/(high-low)*100));
  return '<div class="dollar-range '+color+'">'+
    '<div class="range-label"><span>Modeled outcome range</span><strong>'+money(low,false)+' to '+money(high,false)+'</strong></div>'+
    '<div class="impact-band-track" role="img" aria-label="Modeled impact from '+esc(money(low,false))+' to '+esc(money(high,false))+' with central estimate '+esc(money(central,false))+'">'+
      '<div class="impact-band-fill"></div><span class="impact-band-dot" style="left:'+pos+'%"></span></div>'+
    '<p>Range across the selected scenario inputs.</p></div>';
}
function htmlImpactPanel(){
  const event=selectedEvent();
  if(!event)return "";
  const market=event.kind==="market";
  const selected=market?selectedOption(event):null;
  const yes=market?event.outcomes.find(o=>o.contract_id===selected?.contract_id&&o.id.endsWith(":YES")):null;
  const no=market?event.outcomes.find(o=>o.contract_id===selected?.contract_id&&o.id.endsWith(":NO")):null;
  const hasContract=Boolean(selected?.contract_id);
  const quoted=hasContract?(yes||selected):selected;
  const impact=adjustedImpact(event);
  const probability=market?quoted?.q_reference:event.probability;
  const range=market?quoted?.q_interval:event.bounds;
  const validRange=Array.isArray(range)&&range.length===2&&range.every(Number.isFinite);
  const left=validRange?Math.max(0,Math.min(100,range[0]*100)):0;
  const right=validRange?Math.max(left,Math.min(100,range[1]*100)):0;
  const center=validRange?Math.max(left,Math.min(right,probability*100)):0;
  const diff=Number.isFinite(impact)?money(impact,false):"No estimate";
  const outcomeLabel=market?(selected?.label?.replace(/^YES:\s*/,"").replace(/^NO:\s*/,"")||"Selected outcome"):"Event occurs vs. does not occur";
  const probLabel=hasContract?"YES contract price":"Scenario weight";
  const impactLabel=market?"Portfolio change":"Portfolio difference";
  const comparison=yes&&no&&Number.isFinite(yes.pnl)&&Number.isFinite(no.pnl)?yes.pnl-no.pnl:null;
  return '<article class="result-pane" aria-labelledby="selected-event-title">'+
    '<div class="result-head"><span class="result-date">Research snapshot · '+esc(event.sourceDate)+'</span>'+
       '<span class="result-source-tag">'+(hasContract?"Contract scenario":"Recorded scenario")+'</span></div>'+
    '<h2 id="selected-event-title" class="result-title">'+esc(event.name)+'</h2>'+
    (market?htmlMarketControls(event):"")+
    '<div class="results-grid">'+
      '<div class="result-metric probability-metric"><span>'+probLabel+'</span><strong>'+pct(probability,market?1:0)+'</strong>'+
        '<small>'+(hasContract?"Recorded contract quote":"Recorded scenario estimate")+'</small></div>'+
      '<div class="result-metric impact-metric '+(impact<0?"is-loss":impact>0?"is-gain":"")+'"><span>'+impactLabel+'</span>'+
        '<strong class="'+(!Number.isFinite(impact)?"result-pending":impact<0?"result-loss":impact>0?"result-gain":"result-flat")+'">'+diff+'</strong>'+
        '<small>'+(Number.isFinite(impact)?esc(outcomeLabel):"Not quantified")+'</small></div>'+
    '</div>'+
    (market&&Number.isFinite(comparison)?'<div class="comparison-line"><span>YES − NO difference</span><strong class="'+(comparison<0?"result-loss":comparison>0?"result-gain":"")+'">'+money(comparison,false)+'</strong></div>':"")+
    '<div class="result-range"><div class="range-label"><span>'+(hasContract?"Market quote range":"Scenario-weight range")+'</span><strong>'+(validRange?pct(range[0],1)+' – '+pct(range[1],1):"Not available")+'</strong></div>'+
      '<div class="range-track" role="img" aria-label="Estimate '+pct(probability,1)+' within range '+(validRange?pct(range[0],1)+' to '+pct(range[1],1):"not available")+'">'+
        '<div class="range-band" style="left:'+left+'%;width:'+(right-left)+'%"></div>'+
        '<div class="range-dot" style="left:'+center+'%"></div></div>'+
      '<p>'+(hasContract?"Range of recorded prices for the selected contract.":"Range across recorded scenario weights.")+'</p></div>'+
    htmlDollarRange(market?selected?.pnl_interval:null,impact)+
    '<div class="result-foot">'+
      '<button class="method-toggle" type="button" data-action="details" aria-expanded="'+String(state.details)+'" aria-controls="event-method">'+
        (state.details?"Close details":"Definitions & sources")+' <span aria-hidden="true">'+(state.details?'−':'+')+'</span></button></div>'+
    (state.details?htmlMethod(event):"")+
    '</article>';
}
function htmlRecorded(){
  return '<section class="analysis-view" aria-labelledby="workspace-heading">'+
    '<div class="view-heading"><h1 id="workspace-heading">What could move your portfolio?</h1></div>'+
    '<div class="analysis-grid">'+htmlEventList()+htmlImpactPanel()+'</div>'+
  '</section>';
}
function htmlCustom(){
  return '<section class="analysis-view" aria-labelledby="workspace-heading">'+
    '<div class="view-heading"><h1 id="workspace-heading">Analyze your holdings</h1></div>'+
    '<div class="custom-placeholder">'+
    '<h2>'+(state.jobStatus?"Analyzing your portfolio":"Your portfolio is ready")+'</h2>'+
    '<p>'+(state.jobStatus?esc(state.jobStatus):"Find relevant macro events, review their probability evidence and estimate the possible portfolio impact.")+'</p>'+
    (state.jobError?'<p class="custom-error" role="alert">'+esc(state.jobError)+'</p>':"")+
    '<button class="solid-button" type="button" data-action="run-analysis" '+(state.jobStatus?"disabled":"")+'>Start analysis</button>'+
    '<button class="custom-secondary" type="button" data-action="restore-sample">Explore an example</button></div>'+
    '</section>';
}
function render(){
  if(!state.snapshots)return;
  const previousScroll=appRoot.querySelector(".event-scroll")?.scrollTop||0;
  appRoot.removeAttribute("role");
  appRoot.innerHTML=htmlPortfolio()+(state.sample?htmlRecorded():htmlCustom());
  const nextScroll=appRoot.querySelector(".event-scroll");
  if(nextScroll)nextScroll.scrollTop=previousScroll;
}
function formatDraftValue(number){return Number.isFinite(number)?String(Math.round(number*100)/100):"";}
function editorRow(row,index){
  return '<div class="holding-form-row">'+
    '<input class="field-input" data-field="ticker" data-index="'+index+'" aria-label="Ticker '+(index+1)+'" autocomplete="off" maxlength="12" value="'+esc(row.symbol)+'" placeholder="e.g. AAPL">'+
    '<input class="field-input" data-field="value" data-index="'+index+'" aria-label="'+(state.mode==="weight"?"Weight in percent":"Value in dollars")+' for holding '+(index+1)+'" type="number" min="0" step="any" value="'+esc(formatDraftValue(row.value))+'">'+
    '<button class="row-delete" type="button" data-action="remove-row" data-index="'+index+'" aria-label="Remove holding '+(index+1)+'">×</button>'+
  '</div>';
}
function validationText(){
  if(state.validation)return '<div class="validation-msg" role="alert">'+esc(state.validation)+'</div>';
  if(state.mode==="weight"){
    const sum=state.draft.reduce((v,r)=>v+(Number(r.value)||0),0);
    return '<div class="validation-msg validation-ok">Total allocation: '+sum.toFixed(1)+'%. Weights must add up to 100%.</div>';
  }
  const sum=state.draft.reduce((v,r)=>v+(Number(r.value)||0),0);
  return '<div class="validation-msg validation-ok">Portfolio value from positions: '+moneyPlain(sum)+'</div>';
}
function htmlManualEditor(){
  return '<div>'+
    '<div class="field-spread"><label for="total-value">Portfolio value (USD)</label>'+
      (state.mode==="weight"
        ?'<input class="field-input" id="total-value" type="number" min="1" step="any" value="'+esc(state.notional)+'">'
        :'<span class="section-note">Calculated from position values</span>')+
    '</div>'+
    '<div class="field-spread"><span class="field-label" style="margin:0">Your holdings</span>'+
      '<button class="toggle-mode" type="button" data-action="toggle-mode">'+
        (state.mode==="weight"?"Use dollar values":"Use percentages")+'</button></div>'+
    '<div class="form-headings"><span>Ticker</span><span>'+(state.mode==="weight"?"Weight (%)":"Value (USD)")+'</span><span></span></div>'+
    '<div class="holding-form-list">'+state.draft.map(editorRow).join("")+'</div>'+
    '<button type="button" class="add-row" data-action="add-row">+ Add another holding</button>'+
    validationText()+
  '</div>';
}
function htmlCsvEditor(){
  return '<div>'+
    '<div class="upload-area" id="upload-area">'+
      '<svg viewBox="0 0 36 36" aria-hidden="true"><path d="M18 24V5m0 0-7 7m7-7 7 7M7 23v8h22v-8"/></svg>'+
      '<h3>Upload a CSV portfolio</h3>'+
      '<p>Drag a file here, or choose one from your computer.</p>'+
      '<input type="file" id="csv-file" accept=".csv,text/csv" aria-label="Choose CSV portfolio">'+
      '<label for="csv-file">Choose CSV</label>'+
    '</div>'+
    '<div class="example-csv">ticker,weight\nNVDA,20\nQQQ,50\nTLT,30</div>'+
    '<p class="inline-note">Accepted columns: ticker + weight (percent), or ticker + value (USD).</p>'+
    (state.csvName?'<div class="validation-msg validation-ok">Loaded: '+esc(state.csvName)+' · '+state.draft.length+' holdings. Review them under Add holdings.</div>':"")+
    (state.validation?'<div class="validation-msg" role="alert">'+esc(state.validation)+'</div>':"")+
  '</div>';
}
function renderEditor(){
  portfolioDialog.innerHTML=
    '<div class="dialog-top"><div><span class="eyebrow">Portfolio setup</span><h2 id="portfolio-dialog-title">Add your investments</h2></div>'+
      '<button class="icon-button" type="button" data-close="portfolio-dialog" aria-label="Close">×</button></div>'+
    '<div class="compose-tabs" role="tablist" aria-label="Portfolio entry method">'+
      '<button class="compose-tab '+(state.editorTab==="manual"?"active":"")+'" type="button" data-action="editor-tab" data-tab="manual" role="tab" aria-selected="'+(state.editorTab==="manual")+'">Add holdings</button>'+
      '<button class="compose-tab '+(state.editorTab==="csv"?"active":"")+'" type="button" data-action="editor-tab" data-tab="csv" role="tab" aria-selected="'+(state.editorTab==="csv")+'">Upload CSV</button></div>'+
    (state.editorTab==="manual"?htmlManualEditor():htmlCsvEditor())+
    '<div class="dialog-actions"><button class="ghost-button" type="button" data-action="reset-editor">Use sample portfolio</button>'+
      '<button class="solid-button" type="button" data-action="save-portfolio">Continue →</button></div>';
}
function openEditor(){
  state.mode="weight";state.editorTab="manual";state.validation="";state.csvName="";
  state.draft=state.holdings.map(h=>({symbol:h.symbol,value:h.weight*100}));
  renderEditor();portfolioDialog.showModal();
}
function resetEditor(){
  state.mode="weight";state.editorTab="manual";state.notional=250000;state.validation="";state.csvName="";
  state.draft=SAMPLE.map(h=>({symbol:h.symbol,value:h.weight*100}));
  renderEditor();
}
function savePortfolio(){
  const rows=state.draft.map(r=>({
    symbol:String(r.symbol||"").trim().toUpperCase(),value:Number(r.value)
  })).filter(r=>r.symbol||r.value);
  if(rows.length===0){state.validation="Add at least one holding.";renderEditor();return;}
  if(rows.length>24){state.validation="Enter up to 24 holdings.";renderEditor();return;}
  const seen=new Set();
  for(const row of rows){
    if(!/^[A-Z][A-Z0-9.\-]{0,11}$/.test(row.symbol)){
      state.validation="Check ticker symbols. Each holding needs a valid ticker.";renderEditor();return;
    }
    if(seen.has(row.symbol)){state.validation="Combine positions with the same ticker.";renderEditor();return;}
    if(!Number.isFinite(row.value)||row.value<=0){state.validation="Enter a positive amount for every holding.";renderEditor();return;}
    seen.add(row.symbol);
  }
  let sum=rows.reduce((v,r)=>v+r.value,0);
  if(state.mode==="weight"&&Math.abs(sum-100)>.05){
    state.validation="Weights currently total "+sum.toFixed(1)+"%. Adjust them to 100%.";
    renderEditor();return;
  }
  if(state.mode==="weight"&&(!Number.isFinite(state.notional)||state.notional<=0)){
    state.validation="Enter a positive portfolio value.";renderEditor();return;
  }
  if(state.mode==="value")state.notional=sum;
  state.holdings=rows.map(r=>({symbol:r.symbol,weight:r.value/sum}));
  const matchesExample=samePortfolio(state.holdings)&&Math.abs(state.notional-250000)<.01;
  state.sample=matchesExample;
  state.activeExample=matchesExample?PRIMARY_EXAMPLE_ID:"custom";
  state.validation="";state.details=false;
  state.jobStatus="";state.jobError="";
  portfolioDialog.close();
  if(matchesExample)chooseExample(PRIMARY_EXAMPLE_ID);
  else render();
  showToast(matchesExample?"Sample portfolio loaded":"Portfolio ready. Select Start analysis.");
}
function parseCsvLine(text){
  const result=[];let current="",quoted=false;
  for(let i=0;i<text.length;i++){
    const char=text[i];
    if(char==='"'){
      if(quoted&&text[i+1]==='"'){current+='"';i++;}
      else quoted=!quoted;
    }else if(char===","&&!quoted){result.push(current.trim());current="";}
    else current+=char;
  }
  if(quoted)throw new Error("A CSV row has an unclosed quotation mark.");
  result.push(current.trim());return result;
}
async function loadCsv(file){
  if(!file)return;
  if(file.size>1024*1024){state.validation="Choose a CSV smaller than 1 MB.";renderEditor();return;}
  try{
    const content=(await file.text()).replace(/^\uFEFF/,"");
    const lines=content.split(/\r?\n/).map(s=>s.trim()).filter(Boolean);
    if(lines.length<2)throw new Error("Add a header and at least one holding.");
    const headers=parseCsvLine(lines[0]).map(s=>s.toLowerCase().replace(/[\s_]+/g,""));
    const tickerCol=headers.findIndex(h=>["ticker","symbol"].includes(h));
    const weightCol=headers.findIndex(h=>["weight","weight%","percent","allocation"].includes(h));
    const valueCol=headers.findIndex(h=>["value","marketvalue","amount","notional"].includes(h));
    if(tickerCol<0||weightCol<0&&valueCol<0)
      throw new Error("The CSV needs ticker and weight, or ticker and value columns.");
    if(lines.length>25)throw new Error("Import up to 24 holdings.");
    const useValue=weightCol<0;
    const valueColIndex=useValue?valueCol:weightCol;
    const rows=lines.slice(1).map(function(line){
      const cells=parseCsvLine(line);
      return {symbol:(cells[tickerCol]||"").toUpperCase(),
        value:Number(String(cells[valueColIndex]||"").replace(/[%$,]/g,""))};
    });
    if(rows.some(r=>!r.symbol||!Number.isFinite(r.value)))
      throw new Error("Check the ticker and amount in each CSV row.");
    state.draft=rows;state.mode=useValue?"value":"weight";
    state.csvName=file.name;state.validation="";
    state.editorTab="manual";
    if(useValue)state.notional=rows.reduce((v,r)=>v+r.value,0);
    renderEditor();
    showToast("CSV loaded. Review your holdings, then continue.");
  }catch(error){
    state.validation=error.message||"The CSV file could not be read.";
    renderEditor();
  }
}
async function runAnalysis(){
  if(state.jobStatus)return;
  state.jobError="";
  state.jobStatus="Checking the analysis service…";
  render();
  try{
    const statusReply=await fetch("/api/status",{credentials:"same-origin",cache:"no-store"});
    if(!statusReply.ok)throw new Error("Live analysis is not connected to this preview. The research portfolios remain available.");
    const config=await statusReply.json();
    if(!config.available||!config.csrf_token)throw new Error("The live analysis service is not ready.");
    if(!window.confirm("Live research uses external data and model APIs and may incur usage charges. Start this analysis?")){
      state.jobStatus="";
      render();
      return;
    }
    const positions=state.holdings.map(h=>({symbol:h.symbol,type:"auto",amount:Math.round(state.notional*h.weight*100)/100}));
    const amountGap=state.notional-positions.reduce((s,r)=>s+r.amount,0);
    positions[positions.length-1].amount+=Math.round(amountGap*100)/100;
    const started=await fetch("/api/analyze",{
      method:"POST",credentials:"same-origin",
      headers:{"Content-Type":"application/json","X-Portfolio-Impact":"1","X-CSRF-Token":config.csrf_token},
      body:JSON.stringify({positions})
    });
    const first=await started.json().catch(()=>({}));
    if(!started.ok||!first.job_id)throw new Error(first.error||"The analysis could not start.");
    const startTime=Date.now();
    while(Date.now()-startTime<40*60*1000){
      await new Promise(resolve=>setTimeout(resolve,2500));
      const poll=await fetch("/api/jobs/"+encodeURIComponent(first.job_id),{credentials:"same-origin",cache:"no-store"});
      if(!poll.ok)throw new Error("Could not check the analysis status.");
      const job=await poll.json();
      if(job.status==="complete"){
        if(!job.case||!Array.isArray(job.case.risks))throw new Error("The service returned an incomplete report.");
        state.events=eventsFromSnapshot(job.case,new Date().toISOString().slice(0,10));
        if(!state.events.length)throw new Error("No quantified event scenarios were returned for these holdings.");
        state.sample=true;
        state.activeExample="live";
        state.jobStatus="";
        selectEvent(state.events[0].id);
        render();
        showToast("Your analysis is ready.");
        return;
      }
      if(job.status==="failed")throw new Error(job.error||"The analysis stopped.");
      state.jobStatus=job.progress||"Checking events and market evidence…";
      render();
    }
    throw new Error("The analysis exceeded the waiting window. Refresh before retrying.");
  }catch(error){
    state.jobStatus="";
    state.jobError=error.message||"The analysis service is unavailable.";
    render();
  }
}
document.addEventListener("click",function(event){
  const close=event.target.closest("[data-close]");
  if(close){document.getElementById(close.dataset.close)?.close();return;}
  const action=event.target.closest("[data-action]");
  if(!action)return;
  switch(action.dataset.action){
    case "edit":openEditor();break;
    case "event":
      selectEvent(action.dataset.id);render();break;
    case "outcome":
      if(action.dataset.id){state.selectedOutcomeId=action.dataset.id;render();}
      break;
    case "other":
      state.showOther=!state.showOther;render();break;
    case "details":
      state.details=!state.details;render();break;
    case "restore-sample":
      chooseExample(PRIMARY_EXAMPLE_ID);showToast("Sample portfolio loaded.");break;
    case "run-analysis":runAnalysis();break;
    case "editor-tab":
      state.editorTab=action.dataset.tab;state.validation="";renderEditor();break;
    case "toggle-mode":{
      const total=state.draft.reduce((v,r)=>v+(Number(r.value)||0),0);
      if(state.mode==="weight"){
        state.draft=state.draft.map(r=>({...r,value:Number(r.value)*state.notional/100}));
        state.mode="value";
      }else{
        state.draft=state.draft.map(r=>({...r,value:total>0?Number(r.value)*100/total:0}));
        state.mode="weight";
      }
      state.validation="";renderEditor();break;
    }
    case "add-row":
      state.draft.push({symbol:"",value:""});state.validation="";renderEditor();break;
    case "remove-row":
      state.draft.splice(Number(action.dataset.index),1);state.validation="";renderEditor();break;
    case "save-portfolio":savePortfolio();break;
    case "reset-editor":resetEditor();break;
  }
});
document.addEventListener("input",function(event){
  const field=event.target.dataset.field;
  if(field){
    const index=Number(event.target.dataset.index);
    if(state.draft[index])state.draft[index][field==="ticker"?"symbol":"value"]=event.target.value;
  }
  if(event.target.id==="total-value")state.notional=Number(event.target.value);
});
document.addEventListener("change",function(event){
  if(event.target.id==="csv-file")loadCsv(event.target.files?.[0]);
  if(event.target.id==="example-select"&&event.target.value!=="custom"&&event.target.value!=="live"){
    chooseExample(event.target.value);
  }
  if(event.target.id==="contract-select"){
    state.selectedContract=event.target.value;
    const option=selectedEvent()?.outcomes.find(o=>o.contract_id===state.selectedContract&&o.id.endsWith(":YES"));
    state.selectedOutcomeId=option?.id||null;
    render();
  }
  if(event.target.id==="outcome-select"){state.selectedOutcomeId=event.target.value;render();}
});
portfolioDialog.addEventListener("dragover",function(event){
  if(state.editorTab!=="csv")return;
  event.preventDefault();
  document.getElementById("upload-area")?.classList.add("dragover");
});
portfolioDialog.addEventListener("dragleave",function(event){
  document.getElementById("upload-area")?.classList.remove("dragover");
});
portfolioDialog.addEventListener("drop",function(event){
  if(state.editorTab!=="csv")return;
  event.preventDefault();
  document.getElementById("upload-area")?.classList.remove("dragover");
  loadCsv(event.dataTransfer?.files?.[0]);
});
document.getElementById("edit-button").addEventListener("click",openEditor);

fetch(MARKET_URL,{cache:"no-store"}).then(function(response){
  if(!response.ok)throw new Error("Could not open the research examples.");
  return response.json();
}).then(function(snapshots){
  if(snapshots.schema!=="portfolio-impact.ui.market-scenarios.v1"||!Array.isArray(snapshots.cases))
    throw new Error("The recorded scenarios did not pass the data check.");
  const example=snapshots.cases.find(c=>c.id===PRIMARY_EXAMPLE_ID);
  if(!example||!example.risks?.some(r=>r.outcomes?.some(o=>Number.isFinite(o.pnl))))
    throw new Error("The starting portfolio is missing quantified scenarios.");
  state.snapshots=snapshots;
  chooseExample(PRIMARY_EXAMPLE_ID);
}).catch(function(error){
  appRoot.innerHTML='<div class="custom-placeholder"><h2>Could not load the analysis</h2>'+
    '<p>'+esc(error.message)+'</p></div>';
});
