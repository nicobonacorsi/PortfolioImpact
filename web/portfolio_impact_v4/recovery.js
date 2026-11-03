"use strict";

// Recover jobs by their submitted holdings and dollar amounts.
const ACTIVE_JOB_STORAGE = "castle_portfolio_v4_active_job";
function savedAnalysisJobs(){
  try{
    const raw=sessionStorage.getItem(ACTIVE_JOB_STORAGE);
    if(!raw)return {version:2,jobs:[],legacyId:""};
    try{
      const saved=JSON.parse(raw);
      if(saved&&saved.version===2&&Array.isArray(saved.jobs))
        return {version:2,jobs:saved.jobs.filter(job=>job&&typeof job.id==="string"&&typeof job.key==="string"),
          legacyId:typeof saved.legacyId==="string"?saved.legacyId:""};
    }catch(_ignored){}
    return {version:2,jobs:[],legacyId:/^[A-Za-z0-9_-]{1,200}$/.test(raw)?raw:""};
  }catch(_ignored){return {version:2,jobs:[],legacyId:""};}
}
function writeAnalysisJobs(saved){
  try{
    if(saved.jobs.length||saved.legacyId)sessionStorage.setItem(ACTIVE_JOB_STORAGE,JSON.stringify(saved));
    else sessionStorage.removeItem(ACTIVE_JOB_STORAGE);
  }catch(_ignored){}
}
function rememberAnalysisJob(id,key){
  const saved=savedAnalysisJobs();
  saved.jobs=saved.jobs.filter(job=>job.key!==key&&job.id!==id);
  saved.jobs.push({id,key});
  if(saved.legacyId===id)saved.legacyId="";
  writeAnalysisJobs(saved);
}
function forgetAnalysisJob(id){
  const saved=savedAnalysisJobs();
  saved.jobs=saved.jobs.filter(job=>job.id!==id);
  if(saved.legacyId===id)saved.legacyId="";
  writeAnalysisJobs(saved);
}
function rememberedAnalysisJob(key){
  const saved=savedAnalysisJobs();
  return saved.jobs.find(job=>job.key===key)?.id||saved.legacyId||"";
}
function acceptLiveAnalysis(job,request,jobId){
  if(!analysisIsCurrent(request))return false;
  // A completed legacy job can be assigned to its actual portfolio safely.
  if(job.case&&Array.isArray(job.case.positions)){
    const actualKey=analysisPortfolioKey(job.case.positions,job.case.notional);
    if(actualKey!==request.key){
      rememberAnalysisJob(jobId,actualKey);
      throw new Error("The saved analysis is for different holdings or amounts. It has been kept for that portfolio. Select Start analysis again to analyze the current portfolio.");
    }
  }
  if(!applyLiveAnalysis(job,request))return false;
  forgetAnalysisJob(jobId);
  return true;
}
runAnalysis = async function runAnalysisWithRecovery(){
  if(state.jobStatus)return;
  const request=analysisRequest();
  state.jobError="";
  state.jobStatus="Checking the analysis service…";
  render();
  try{
    request.positions=analysisPositions(request.holdings,request.notional);
    request.key=analysisPortfolioKey(request.holdings,request.notional);
    const statusReply=await fetch("/api/status",{credentials:"same-origin",cache:"no-store"});
    if(!analysisIsCurrent(request))return;
    if(!statusReply.ok)throw new Error("Cannot reach the analysis server. Restart the Codespace, then try again. Your saved provider receipts are not deleted.");
    const config=await statusReply.json();
    if(!analysisIsCurrent(request))return;
    if(!config.available||!config.csrf_token)throw new Error("The live analysis service is not ready.");
    let jobId=rememberedAnalysisJob(request.key);
    let firstStatus=null;
    if(jobId){
      const prior=await fetch("/api/jobs/"+encodeURIComponent(jobId),{credentials:"same-origin",cache:"no-store"});
      if(!analysisIsCurrent(request))return;
      if(prior.ok){
        firstStatus=await prior.json();
        if(!analysisIsCurrent(request))return;
        if(firstStatus.status==="complete"){
          acceptLiveAnalysis(firstStatus,request,jobId);
          return;
        }
        if(firstStatus.status!=="running"){
          forgetAnalysisJob(jobId);
          jobId="";
          firstStatus=null;
        }
      }else{
        if(prior.status!==404)throw new Error("Could not reconnect to the saved analysis. Try again without starting a duplicate job.");
        forgetAnalysisJob(jobId);
        jobId="";
      }
    }
    if(!jobId){
      if(!window.confirm("Live research uses external data and model APIs and may incur usage charges. If this portfolio was interrupted, saved verified responses will be reused. Continue?")){
        state.jobStatus="";
        render();
        return;
      }
      const started=await fetch("/api/analyze",{
        method:"POST",credentials:"same-origin",
        headers:{"Content-Type":"application/json","X-Portfolio-Impact":"1","X-CSRF-Token":config.csrf_token},
        body:JSON.stringify({positions:request.positions})
      });
      const first=await started.json().catch(()=>({}));
      // Retain the original portfolio identity even if the user changed views.
      if(started.ok&&first.job_id)rememberAnalysisJob(first.job_id,request.key);
      if(!analysisIsCurrent(request))return;
      if(!started.ok||!first.job_id)throw new Error(first.error||"The analysis could not start.");
      jobId=first.job_id;
    }
    const startTime=Date.now();
    while(Date.now()-startTime<40*60*1000){
      if(!analysisIsCurrent(request))return;
      let job=firstStatus;
      firstStatus=null;
      if(!job){
        await new Promise(resolve=>setTimeout(resolve,2500));
        if(!analysisIsCurrent(request))return;
        const poll=await fetch("/api/jobs/"+encodeURIComponent(jobId),{credentials:"same-origin",cache:"no-store"});
        if(!analysisIsCurrent(request))return;
        if(!poll.ok)throw new Error("The server stopped responding. Restart the Codespace and press Start analysis to reconnect to this job.");
        job=await poll.json();
        if(!analysisIsCurrent(request))return;
      }
      if(job.status==="complete"){
        acceptLiveAnalysis(job,request,jobId);
        return;
      }
      if(job.status==="interrupted"){
        forgetAnalysisJob(jobId);
        throw new Error("The Codespace interrupted this job. Its saved provider responses remain available. Reconnect providers and press Start analysis to resume.");
      }
      if(job.status==="failed"){
        forgetAnalysisJob(jobId);
        throw new Error(job.error||"The analysis stopped. Verified saved provider responses can be reused on retry.");
      }
      state.jobStatus=job.progress||"Checking events and market evidence…";
      render();
    }
    throw new Error("The analysis exceeded the waiting window. The job may still be running; press Start analysis to reconnect without starting another job.");
  }catch(error){
    if(!analysisIsCurrent(request))return;
    state.jobStatus="";
    state.jobError=error.message||"The analysis service is unavailable. The saved job can be resumed.";
    render();
  }
};
