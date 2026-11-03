"use strict";

const assert=require("node:assert/strict");
const vm=require("node:vm");
const fs=require("node:fs");
const path=require("node:path");

// Synthetic reports exercise browser state only. They are never product data.
const report=(id,notional,positions,pnl)=>({
  id,notional,positions,capture:"2026-10-25T12:00:00Z",
  risks:[{id:"event-"+id,title:"Test event "+id,date:"2027-01-23",
    outcomes:[{id:"yes",pnl,q_reference:.25,q_interval:[.2,.3],pnl_interval:[pnl-1,pnl+1]}]}]
});
const A=report("technology_and_treasury",250000,[
  {symbol:"NVDA",weight:.2},{symbol:"QQQ",weight:.5},{symbol:"TLT",weight:.3}
],-100);
const B=report("bonds",100000,[{symbol:"BND",weight:.6},{symbol:"IEF",weight:.4}],-200);
const copy=value=>JSON.parse(JSON.stringify(value));
const reply=(value,status=200)=>({ok:status>=200&&status<300,status,json:async()=>copy(value)});
const deferred=()=>{let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no;});return {promise,resolve,reject};};

function harness(appSource,recoverySource,options={}){
  const listeners={};
  const requests=[];
  const storage=options.storage||new Map();
  let confirmations=0;
  let api=options.api||(()=>{throw new Error("Unexpected API request");});
  const blank=()=>({addEventListener(){},classList:{add(){},remove(){}},
    showModal(){this.open=true;},close(){this.open=false;},innerHTML:"",dataset:{}});
  const elements=new Map();
  const document={
    getElementById(id){if(!elements.has(id))elements.set(id,blank());return elements.get(id);},
    addEventListener(type,callback){(listeners[type]??=[]).push(callback);}
  };
  const NativeDate=Date;
  class FixedDate extends NativeDate{
    constructor(...args){super(...(args.length?args:["2026-11-03T22:00:00Z"]));}
    static now(){return new NativeDate("2026-11-03T22:00:00Z").getTime();}
  }
  const context=vm.createContext({
    document,Date:FixedDate,Intl,console,
    window:{confirm(){confirmations++;return options.confirm===true;}},
    sessionStorage:{
      getItem(key){if(options.storageThrows)throw new Error("Storage unavailable");return storage.get(key)||null;},
      setItem(key,value){if(options.storageThrows)throw new Error("Storage unavailable");storage.set(key,value);},
      removeItem(key){if(options.storageThrows)throw new Error("Storage unavailable");storage.delete(key);}
    },
    setTimeout(callback){queueMicrotask(callback);return 1;},clearTimeout(){},
    fetch:async(url,config)=>{
      if(url.startsWith("./"))return new Promise(()=>{});
      requests.push({url,config});
      if(requests.length>30)throw new Error("Unexpected polling loop");
      return api(url,config);
    }
  });
  vm.runInContext(appSource,context,{filename:"app.js"});
  if(options.recovery!==false)vm.runInContext(recoverySource,context,{filename:"recovery.js"});
  vm.runInContext("render=()=>{};showToast=()=>{};",context);
  const run=source=>vm.runInContext(source,context);
  run("state.snapshots="+JSON.stringify({snapshot_date:"2026-10-25",cases:[A,B]})+";chooseExample('technology_and_treasury');");
  return {
    run,requests,storage,elements,
    setApi(value){api=value;},
    state(){return copy(run("state"));},
    confirmations(){return confirmations;},
    choose(id){run("chooseExample("+JSON.stringify(id)+")");},
    input(id,value){for(const handler of listeners.input||[])handler({target:{id,value,dataset:{}}});},
    click(action){
      const target={closest(selector){return selector==="[data-action]"?{dataset:{action}}:null;}};
      for(const handler of listeners.click||[])handler({target});
    },
    upload(content){
      context.testFile={name:"positions.csv",size:content.length,text:async()=>content};
      return run("loadCsv(testFile)");
    }
  };
}
function connect(h,completedReport,jobId="saved-job"){
  h.setApi(async url=>{
    if(url==="/api/status")return reply({available:true,csrf_token:"test"});
    if(url==="/api/jobs/"+jobId)return reply({status:"complete",created_at:"2026-10-25T12:00:00Z",case:completedReport});
    throw new Error("Unexpected request "+url);
  });
}
function remember(h,id="saved-job"){
  h.run("rememberAnalysisJob("+JSON.stringify(id)+",analysisPortfolioKey(state.holdings,state.notional))");
}
function committed(h){
  const state=h.state();
  return {holdings:state.holdings,notional:state.notional,events:state.events};
}

async function runTests(appSource,recoverySource){
  const tests=[];
  const test=(name,fn)=>tests.push([name,fn]);

  test("editing the portfolio value does not commit before Save",()=>{
    const h=harness(appSource,recoverySource);h.choose("bonds");
    const before=committed(h);
    h.run("openEditor()");h.input("total-value","150000");h.run("portfolioDialog.close()");
    assert.deepEqual(committed(h),before);
    h.run("openEditor()");
    assert.equal(h.state().draftNotional,100000);
  });

  test("reset then cancel keeps holdings, notional and results",()=>{
    const h=harness(appSource,recoverySource);h.choose("bonds");
    const before=committed(h);
    h.run("openEditor();resetEditor();portfolioDialog.close()");
    assert.deepEqual(committed(h),before);
  });

  test("invalid Save leaves the committed portfolio intact",()=>{
    const h=harness(appSource,recoverySource);h.choose("bonds");
    const before=committed(h);
    h.run("openEditor();state.draft=[{symbol:'AAPL',value:50}];state.draftNotional=900000;savePortfolio()");
    assert.deepEqual(committed(h),before);
    assert.match(h.state().validation,/total/);
  });

  test("successful Save commits the draft atomically and clears stale results",()=>{
    const h=harness(appSource,recoverySource);
    h.run("openEditor();state.draft=[{symbol:'AAPL',value:100}];state.draftNotional=123456;savePortfolio()");
    assert.equal(h.state().notional,123456);
    assert.deepEqual(h.state().holdings,[{symbol:"AAPL",weight:1}]);
    assert.equal(h.state().sample,false);
    assert.deepEqual(h.state().events,[]);
  });

  test("CSV dollar import is only a draft until Save",async()=>{
    const h=harness(appSource,recoverySource);const before=committed(h);
    h.run("openEditor()");
    await h.upload("ticker,value\nAAPL,60000\nJPM,40000\n");
    assert.deepEqual(committed(h),before);
    assert.equal(h.state().draftNotional,100000);
    h.run("portfolioDialog.close();openEditor()");
    assert.equal(h.state().draftNotional,250000);
  });

  test("saving a dollar CSV commits the correct total",async()=>{
    const h=harness(appSource,recoverySource);h.run("openEditor()");
    await h.upload("ticker,value\nAAPL,60000\nJPM,40000\n");
    h.run("savePortfolio()");
    assert.equal(h.state().notional,100000);
    assert.deepEqual(h.state().holdings,[{symbol:"AAPL",weight:.6},{symbol:"JPM",weight:.4}]);
  });

  test("dollar to percent conversion preserves the edited draft total",()=>{
    const h=harness(appSource,recoverySource);h.run("openEditor();state.mode='value';state.draft=[{symbol:'BND',value:30000},{symbol:'IEF',value:20000}]");
    h.click("toggle-mode");
    assert.equal(h.state().draftNotional,50000);
    assert.equal(h.state().notional,250000);
    h.click("toggle-mode");
    assert.deepEqual(h.state().draft,[{symbol:"BND",value:30000},{symbol:"IEF",value:20000}]);
  });

  test("portfolio identity is independent of holding order",()=>{
    const h=harness(appSource,recoverySource);
    assert.equal(h.run("analysisPortfolioKey([{symbol:'AAPL',weight:.3},{symbol:'JPM',weight:.7}],100.01)"),
      h.run("analysisPortfolioKey([{symbol:'JPM',weight:.7},{symbol:'AAPL',weight:.3}],100.01)"));
    const positions=copy(h.run("analysisPositions([{symbol:'AAPL',weight:.3},{symbol:'JPM',weight:.7}],100.01)"));
    assert.equal(positions.reduce((sum,row)=>sum+Math.round(row.amount*100),0),10001);
  });

  test("portfolio identity distinguishes dollar amounts",()=>{
    const h=harness(appSource,recoverySource);
    assert.notEqual(h.run("analysisPortfolioKey(state.holdings,250000)"),
      h.run("analysisPortfolioKey(state.holdings,100000)"));
  });

  test("a matching saved job resumes with no new POST or cost confirmation",async()=>{
    const h=harness(appSource,recoverySource);remember(h);connect(h,A);
    await h.run("runAnalysis()");
    assert.equal(h.state().activeExample,"live");
    assert.equal(h.state().notional,250000);
    assert.equal(h.state().events[0].sourceDate,"2026-10-25");
    assert.equal(h.confirmations(),0);
    assert.deepEqual(h.requests.map(x=>x.url),["/api/status","/api/jobs/saved-job"]);
  });

  test("legacy mismatched report cannot replace the visible portfolio",async()=>{
    const h=harness(appSource,recoverySource);
    h.storage.set("castle_portfolio_v4_active_job","saved-job");
    const before=committed(h);connect(h,B);
    await h.run("runAnalysis()");
    assert.deepEqual(committed(h),before);
    assert.match(h.state().jobError,/different holdings or amounts/);
    assert.equal(h.confirmations(),0);
    const saved=JSON.parse(h.storage.get("castle_portfolio_v4_active_job"));
    assert.equal(saved.jobs.length,1);
    h.choose("bonds");
    assert.equal(h.run("rememberedAnalysisJob(analysisPortfolioKey(state.holdings,state.notional))"),"saved-job");
  });

  test("a matching legacy report still resumes without spending",async()=>{
    const h=harness(appSource,recoverySource);
    h.storage.set("castle_portfolio_v4_active_job","saved-job");connect(h,A);
    await h.run("runAnalysis()");
    assert.equal(h.state().activeExample,"live");
    assert.equal(h.confirmations(),0);
    assert.equal(h.state().jobError,"");
  });

  test("the same tickers with a different notional are rejected",async()=>{
    const h=harness(appSource,recoverySource);
    remember(h);connect(h,{...copy(A),notional:100000});
    const before=committed(h);await h.run("runAnalysis()");
    assert.deepEqual(committed(h),before);
    assert.match(h.state().jobError,/different holdings or amounts/);
  });

  test("saved jobs for other portfolios are retained but not resumed",async()=>{
    const h=harness(appSource,recoverySource);remember(h,"tech-job");
    h.choose("bonds");
    h.setApi(async url=>{
      assert.equal(url,"/api/status");return reply({available:true,csrf_token:"test"});
    });
    await h.run("runAnalysis()");
    assert.equal(h.confirmations(),1);
    assert.deepEqual(h.requests.map(x=>x.url),["/api/status"]);
    h.choose("technology_and_treasury");
    assert.equal(h.run("rememberedAnalysisJob(analysisPortfolioKey(state.holdings,state.notional))"),"tech-job");
  });

  test("legacy running jobs are polled and validated before acceptance",async()=>{
    const h=harness(appSource,recoverySource);
    h.storage.set("castle_portfolio_v4_active_job","saved-job");
    let polls=0;
    h.setApi(async url=>{
      if(url==="/api/status")return reply({available:true,csrf_token:"test"});
      assert.equal(url,"/api/jobs/saved-job");
      return reply(++polls===1?{status:"running"}:{status:"complete",case:B});
    });
    const before=committed(h);await h.run("runAnalysis()");
    assert.equal(polls,2);
    assert.deepEqual(committed(h),before);
    assert.equal(h.confirmations(),0);
    assert.match(h.state().jobError,/different holdings or amounts/);
  });

  test("transient recovery errors never launch a duplicate job",async()=>{
    const h=harness(appSource,recoverySource,{confirm:true});remember(h);
    h.setApi(async url=>url==="/api/status"
      ?reply({available:true,csrf_token:"test"}):reply({},503));
    await h.run("runAnalysis()");
    assert.equal(h.confirmations(),0);
    assert.equal(h.run("rememberedAnalysisJob(analysisPortfolioKey(state.holdings,state.notional))"),"saved-job");
    assert.equal(h.requests.some(x=>x.config?.method==="POST"),false);
  });

  test("switching portfolio during status fetch cancels a stale launch",async()=>{
    const h=harness(appSource,recoverySource,{confirm:true});const wait=deferred();
    h.setApi(()=>wait.promise);
    const pending=h.run("runAnalysis()");
    h.choose("bonds");wait.resolve(reply({available:true,csrf_token:"test"}));
    await pending;
    assert.equal(h.state().notional,100000);
    assert.equal(h.confirmations(),0);
    assert.equal(h.requests.length,1);
    assert.equal(h.state().jobStatus,"");
  });

  test("a delayed POST response preserves its original portfolio identity",async()=>{
    const h=harness(appSource,recoverySource,{confirm:true});const wait=deferred();const entered=deferred();
    h.setApi(async(url,config)=>{
      if(url==="/api/status")return reply({available:true,csrf_token:"test"});
      assert.equal(url,"/api/analyze");assert.equal(config.method,"POST");entered.resolve();return wait.promise;
    });
    const pending=h.run("runAnalysis()");await entered.promise;
    h.choose("bonds");wait.resolve(reply({job_id:"tech-job"},202));await pending;
    assert.equal(h.state().notional,100000);
    assert.equal(h.state().activeExample,"bonds");
    assert.equal(h.run("rememberedAnalysisJob(analysisPortfolioKey(state.holdings,state.notional))"),"");
    h.choose("technology_and_treasury");
    assert.equal(h.run("rememberedAnalysisJob(analysisPortfolioKey(state.holdings,state.notional))"),"tech-job");
  });

  test("a late completed report cannot overwrite a newer selected example",async()=>{
    const h=harness(appSource,recoverySource);remember(h);const wait=deferred();const entered=deferred();
    h.setApi(async url=>{
      if(url==="/api/status")return reply({available:true,csrf_token:"test"});
      entered.resolve();return wait.promise;
    });
    const pending=h.run("runAnalysis()");await entered.promise;
    h.choose("bonds");const before=committed(h);
    wait.resolve(reply({status:"complete",case:A}));await pending;
    assert.deepEqual(committed(h),before);
    assert.equal(h.state().activeExample,"bonds");
    assert.equal(h.state().jobError,"");
  });

  test("saving an edited portfolio cancels the old pending view",async()=>{
    const h=harness(appSource,recoverySource);remember(h);const wait=deferred();const entered=deferred();
    h.setApi(async url=>{
      if(url==="/api/status")return reply({available:true,csrf_token:"test"});
      entered.resolve();return wait.promise;
    });
    const pending=h.run("runAnalysis()");await entered.promise;
    h.run("openEditor();state.draft=[{symbol:'AAPL',value:100}];state.draftNotional=75000;savePortfolio()");
    const before=committed(h);wait.resolve(reply({status:"complete",case:A}));await pending;
    assert.deepEqual(committed(h),before);
    assert.equal(h.state().notional,75000);
  });

  test("cancelling an editor draft does not cancel a valid pending analysis",async()=>{
    const h=harness(appSource,recoverySource);remember(h);const wait=deferred();const entered=deferred();
    h.setApi(async url=>{
      if(url==="/api/status")return reply({available:true,csrf_token:"test"});
      entered.resolve();return wait.promise;
    });
    const pending=h.run("runAnalysis()");await entered.promise;
    h.run("openEditor();resetEditor();portfolioDialog.close()");
    wait.resolve(reply({status:"complete",case:A}));await pending;
    assert.equal(h.state().activeExample,"live");
    assert.equal(h.state().jobError,"");
  });

  test("a stale request failure cannot overwrite a newer view",async()=>{
    const h=harness(appSource,recoverySource);const wait=deferred();h.setApi(()=>wait.promise);
    const pending=h.run("runAnalysis()");h.choose("bonds");
    h.run("state.jobStatus='New request';state.jobError=''");
    wait.reject(new Error("Old request failed"));await pending;
    assert.equal(h.state().jobStatus,"New request");
    assert.equal(h.state().jobError,"");
  });

  test("malformed completed reports do not partially replace results",async()=>{
    const h=harness(appSource,recoverySource);remember(h);
    connect(h,{...copy(A),positions:undefined});
    const before=committed(h);await h.run("runAnalysis()");
    assert.deepEqual(committed(h),before);
    assert.notEqual(h.state().jobError,"");
  });

  test("an empty quantified result does not partially replace results",async()=>{
    const h=harness(appSource,recoverySource);remember(h);
    connect(h,{...copy(A),risks:[]});
    const before=committed(h);await h.run("runAnalysis()");
    assert.deepEqual(committed(h),before);
    assert.match(h.state().jobError,/No quantified/);
  });

  test("a valid fresh result works when session storage is unavailable",async()=>{
    const h=harness(appSource,recoverySource,{confirm:true,storageThrows:true});
    h.setApi(async url=>{
      if(url==="/api/status")return reply({available:true,csrf_token:"test"});
      if(url==="/api/analyze")return reply({job_id:"new-job"},202);
      if(url==="/api/jobs/new-job")return reply({status:"complete",case:A});
      throw new Error("Unexpected "+url);
    });
    await h.run("runAnalysis()");
    assert.equal(h.state().activeExample,"live");
    assert.equal(h.state().jobError,"");
  });

  test("the base app also rejects mismatched completed results",async()=>{
    const h=harness(appSource,recoverySource,{confirm:true,recovery:false});
    h.setApi(async url=>{
      if(url==="/api/status")return reply({available:true,csrf_token:"test"});
      if(url==="/api/analyze")return reply({job_id:"new-job"},202);
      return reply({status:"complete",case:B});
    });
    const before=committed(h);await h.run("runAnalysis()");
    assert.deepEqual(committed(h),before);
    assert.match(h.state().jobError,/different portfolio/);
  });

  test("the base app ignores an obsolete status response",async()=>{
    const h=harness(appSource,recoverySource,{confirm:true,recovery:false});
    const wait=deferred();h.setApi(()=>wait.promise);
    const pending=h.run("runAnalysis()");h.choose("bonds");
    wait.resolve(reply({available:true,csrf_token:"test"}));await pending;
    assert.equal(h.state().activeExample,"bonds");
    assert.equal(h.confirmations(),0);
  });

  test("invalid report weights cannot pass the portfolio identity check",()=>{
    const h=harness(appSource,recoverySource);
    assert.throws(()=>h.run("analysisPortfolioKey([{symbol:'AAPL',weight:.5}],100000)"),/100%/);
    assert.throws(()=>h.run("analysisPortfolioKey([{symbol:'AAPL',weight:.5},{symbol:'AAPL',weight:.5}],100000)"),/invalid/);
  });

  const results=[];
  for(const [name,fn] of tests){
    await fn();
    results.push(name);
  }
  return {passed:results.length,tests:results};
}

module.exports={runTests};
if(require.main===module){
  const root=path.resolve(__dirname,"..");
  runTests(
    fs.readFileSync(path.join(root,"web/portfolio_impact_v4/app.js"),"utf8"),
    fs.readFileSync(path.join(root,"web/portfolio_impact_v4/recovery.js"),"utf8")
  ).then(result=>console.log(JSON.stringify(result,null,2)))
    .catch(error=>{console.error(error);process.exitCode=1;});
}
