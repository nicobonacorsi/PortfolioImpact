"use strict";

// Additive restart handling; the approved v4 app.js is unchanged.
const ACTIVE_JOB_STORAGE = "castle_portfolio_v4_active_job";
function rememberAnalysisJob(id){
  try{sessionStorage.setItem(ACTIVE_JOB_STORAGE,id);}catch(_ignored){}
}
function forgetAnalysisJob(){
  try{sessionStorage.removeItem(ACTIVE_JOB_STORAGE);}catch(_ignored){}
}
function rememberedAnalysisJob(){
  try{return sessionStorage.getItem(ACTIVE_JOB_STORAGE)||"";}catch(_ignored){return "";}
}
function acceptLiveAnalysis(job){
  if(!job.case||!Array.isArray(job.case.risks))throw new Error("The service returned an incomplete report.");
  state.events=eventsFromSnapshot(job.case,new Date().toISOString().slice(0,10));
  if(!state.events.length)throw new Error("No quantified event scenarios were returned for these holdings.");
  state.sample=true;
  state.activeExample="live";
  state.jobStatus="";
  forgetAnalysisJob();
  selectEvent(state.events[0].id);
  render();
  showToast("Your analysis is ready.");
}
runAnalysis = async function runAnalysisWithRecovery(){
  if(state.jobStatus)return;
  state.jobError="";
  state.jobStatus="Checking the analysis service…";
  render();
  try{
    const statusReply=await fetch("/api/status",{credentials:"same-origin",cache:"no-store"});
    if(!statusReply.ok)throw new Error("Cannot reach the analysis server. Restart the Codespace, then try again. Your saved provider receipts are not deleted.");
    const config=await statusReply.json();
    if(!config.available||!config.csrf_token)throw new Error("The live analysis service is not ready.");
    let jobId=rememberedAnalysisJob();
    let firstStatus=null;
    if(jobId){
      const prior=await fetch("/api/jobs/"+encodeURIComponent(jobId),{credentials:"same-origin",cache:"no-store"});
      if(prior.ok){
        firstStatus=await prior.json();
        if(firstStatus.status==="complete"){
          acceptLiveAnalysis(firstStatus);
          return;
        }
        if(firstStatus.status!=="running"){
          forgetAnalysisJob();
          jobId="";
          firstStatus=null;
        }
      }else{
        forgetAnalysisJob();
        jobId="";
      }
    }
    if(!jobId){
      if(!window.confirm("Live research uses external data and model APIs and may incur usage charges. If this portfolio was interrupted, saved verified responses will be reused. Continue?")){
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
      jobId=first.job_id;
      rememberAnalysisJob(jobId);
    }
    const startTime=Date.now();
    while(Date.now()-startTime<40*60*1000){
      let job=firstStatus;
      firstStatus=null;
      if(!job){
        await new Promise(resolve=>setTimeout(resolve,2500));
        const poll=await fetch("/api/jobs/"+encodeURIComponent(jobId),{credentials:"same-origin",cache:"no-store"});
        if(!poll.ok)throw new Error("The server stopped responding. Restart the Codespace and press Start analysis to reconnect to this job.");
        job=await poll.json();
      }
      if(job.status==="complete"){
        acceptLiveAnalysis(job);
        return;
      }
      if(job.status==="interrupted"){
        forgetAnalysisJob();
        throw new Error("The Codespace interrupted this job. Its saved provider responses remain available. Reconnect providers and press Start analysis to resume.");
      }
      if(job.status==="failed"){
        forgetAnalysisJob();
        throw new Error(job.error||"The analysis stopped. Verified saved provider responses can be reused on retry.");
      }
      state.jobStatus=job.progress||"Checking events and market evidence…";
      render();
    }
    throw new Error("The analysis exceeded the waiting window. The job may still be running; press Start analysis to reconnect without starting another job.");
  }catch(error){
    state.jobStatus="";
    state.jobError=error.message||"The analysis service is unavailable. The saved job can be resumed.";
    render();
  }
};
