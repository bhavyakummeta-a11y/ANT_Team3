(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const text = value => {
    if (value === null || value === undefined) return 'Not reported';
    if (Array.isArray(value)) return value.map(text).join('; ');
    if (typeof value === 'object') return Object.entries(value).map(([key,val]) => `${human(key)}: ${text(val)}`).join(' · ');
    return String(value);
  };
  const human = value => String(value).replace(/_/g, ' ').replace(/^./, s => s.toUpperCase());
  const number = (value, digits=1) => Number.isFinite(Number(value)) && value !== null ? Number(value).toFixed(digits) : '—';
  const pct = value => typeof value === 'number' && Number.isFinite(value) ? `${(100*value).toFixed(1)}%` : '—';
  function node(tag, value, className) { const n = document.createElement(tag); if (value !== undefined) n.textContent = value; if (className) n.className = className; return n; }
  function list(target, values, empty='No finding reported.') {
    target.replaceChildren();
    if (!values?.length) { target.append(node('p', empty, 'empty')); return; }
    const ul = node('ul'); for (const value of values) ul.append(node('li', text(value))); target.append(ul);
  }
  function details(target, values, heading) {
    if (heading) target.append(node('h3', heading));
    const entries = typeof values === 'object' && values !== null && !Array.isArray(values) ? Object.entries(values) : [['Details', values]];
    const dl = node('dl',undefined,'method-list');
    for (const [key,value] of entries) { dl.append(node('dt',human(key)), node('dd',text(value))); }
    target.append(dl);
  }
  function metrics(target, items) {
    target.replaceChildren(...items.map(([label,value,detail]) => {
      const card=node('div',undefined,'metric');
      card.append(node('div',label,'label'),node('div',value,'value'),node('div',detail,'detail')); return card;
    }));
  }
  function renderFigures(target, figures) {
    target.replaceChildren();
    for (const f of figures || []) {
      if (!f.path) continue;
      const url = new URL(f.path, location.href);
      if (url.origin !== location.origin || !url.pathname.startsWith('/research/outputs/')) continue;
      const figure=node('figure',undefined,'figure'), image=node('img');
      image.src=url.pathname; image.alt=f.description || f.title || 'EEG analysis figure'; image.loading='lazy';
      const caption=node('figcaption'); caption.append(node('b',f.title || 'Measured analysis'),node('span',f.description || ''));
      const link=node('a','Open full-size figure ↗','download'); link.href=url.pathname; link.target='_blank'; link.rel='noopener'; caption.append(node('br'),link);
      image.onerror=()=>{ image.replaceWith(node('p','Figure unavailable. Regenerate the analysis report.','error')); };
      figure.append(image,caption); target.append(figure);
    }
  }
  function dataTable(target, rows, fields) {
    const wrap=node('div',undefined,'table-scroll'), table=node('table'), head=node('thead'), header=node('tr'), body=node('tbody');
    for (const [key,label] of fields) header.append(node('th',label)); head.append(header);
    for (const row of rows) {
      const tr=node('tr'); for (const [key] of fields) tr.append(node('td',text(row[key]))); body.append(tr);
    }
    table.append(head,body);wrap.append(table);target.append(wrap);
  }
  function renderAnt(report) {
    const records=report.recordings || [], events=report.events || [];
    const valid=events.filter(e=>e.valid===true).length;
    metrics($('ant-metrics'),[
      ['Recording files',String(records.length),'ANT DATA'],
      ['Recorded duration',`${number(records.reduce((n,r)=>n+(Number(r.duration_seconds)||0),0)/60)} min`,'ALL INVENTORIED FILES'],
      ['Annotated task intervals',String(events.length),'EXPERIMENTAL LABELS'],
      ['Eligible analysis intervals',String(valid),'AFTER WINDOW CHECKS']
    ]);
    list($('ant-findings'),report.findings);
    list($('ant-limitations'),report.limitations,'Limitations were not supplied; inspect the source report before interpreting results.');
    renderFigures($('ant-figures'),report.figures);
    $('ant-recordings').replaceChildren();
    for (const r of records) {
      const tr=node('tr');
      for (const value of [r.file || r.id, `${number(r.duration_seconds)} s`,`${number(r.sampling_hz,0)} Hz`, Array.isArray(r.channels)?r.channels.length:(r.channels ?? r.channel_count ?? '—'),text(r.role)]) tr.append(node('td',value));
      $('ant-recordings').append(tr);
    }
    const picker=$('recording-picker');picker.replaceChildren(...records.map((r,i)=>new Option(r.file||r.id,String(i))));
    const showRecord=()=>{
      const r=records[Number(picker.value)]; if (!r) return;
      const target=$('recording-detail');target.replaceChildren();
      if (r.warnings?.length) for(const warning of r.warnings) target.append(node('p',text(warning),'recording-warning'));
      details(target,r.quality || {},'Signal-quality descriptors');
      if(r.segmentation) details(target,r.segmentation,'Acquisition segments');
      if(r.annotations?.length) dataTable(target,r.annotations,[['onset','Onset (s)'],['duration','Duration (s)'],['label','Recorded label']]);
      else target.append(node('p','No annotations found for this recording.','empty'));
    };
    picker.onchange=showRecord;showRecord();
    details($('ant-methods'),report.methods,'Reproducible method');
    dataTable($('ant-events'),events.map(e=>({...e,valid:e.valid?'Included':'Excluded',quality_flags:e.quality_flags?.length?e.quality_flags:'None flagged'})),[['recording','Recording'],['task','Annotated task'],['condition','Condition'],['onset','Onset (s)'],['duration','Duration (s)'],['valid','Window eligibility'],['reason','Reason'],['quality_flags','Quality flags']]);
    if(report.timing_audit?.length) {
      $('ant-timing').append(node('h3','Legacy script timing audit'));
      const keys=Object.keys(report.timing_audit[0]).filter(k=>!['file_hash','sha256'].includes(k));
      dataTable($('ant-timing'),report.timing_audit,keys.map(k=>[k,human(k)]));
    }
  }
  function renderExternal(report) {
    const count=report.counts||{},m=report.metrics||{};
    metrics($('external-metrics'),[
      ['Public participants',String(count.subjects ?? '—'),'SELECTION IN REPORT'],
      ['Evaluated epochs',String(count.epochs ?? '—'),'PUBLIC DATA ONLY'],
      ['Balanced accuracy',pct(m.balanced_accuracy),'HELD-OUT PARTICIPANTS'],
      ['Dummy baseline',pct(m.dummy_balanced_accuracy),'SAME EVALUATION SPLITS']
    ]);
    list($('external-findings'),report.findings);
    list($('external-limitations'),report.limitations);
    renderFigures($('external-figures'),report.figures);
    details($('external-methods'),report.selection,'Dataset selection');
    details($('external-methods'),report.methods,'Evaluation method');
    details($('external-source'),report.source,'Dataset provenance');
    if(report.subject_scores?.length) dataTable($('external-methods'),report.subject_scores.map(s=>({...s,balanced_accuracy:pct(s.balanced_accuracy),dummy_balanced_accuracy:pct(s.dummy_balanced_accuracy)})),[['subject','Held-out subject'],['epochs','Epochs'],['balanced_accuracy','Balanced accuracy'],['dummy_balanced_accuracy','Dummy baseline']]);
  }
  function renderComparison(report) {
    list($('comparison-findings'),report.findings);
    list($('comparison-limitations'),report.limitations);
    list($('next-experiment'),report.next_experiment);
    renderFigures($('comparison-figures'),report.figures);
    dataTable($('comparison-results'),(report.channel_comparisons||[]).map(r=>({...r,ant_difference_db:number(r.ant_difference_db,3),external_difference_db:number(r.external_difference_db,3),direction_agrees:r.direction_agrees?'Same direction':'Different directions'})),[['band','Band'],['channel','Electrode'],['ant_difference_db','ANT difference (dB)'],['external_difference_db','Public difference (dB)'],['direction_agrees','Descriptive direction only']]);
  }
  async function loadReport(name, render) {
    try {
      const response=await fetch(`research/outputs/${name}_report.json`,{cache:'no-store'});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const report=await response.json();
      render(report);
      return {name,status:report.status || 'loaded'};
    } catch(error) {
      const box=$(name+'-findings');box.replaceChildren(node('p',`The ${name==='ant'?'ANT':name==='external'?'external':'comparison'} report is unavailable. Run the analysis and serve this page through the project server; no result is being substituted.`,'error'));
      return {name,status:'unavailable'};
    }
  }
  Promise.all([loadReport('ant',renderAnt),loadReport('external',renderExternal),loadReport('comparison',renderComparison)]).then(results=>{
    $('load-status').textContent=results.map(r=>`${r.name==='ant'?'ANT analysis':r.name==='external'?'External benchmark':'Comparison'}: ${r.status}`).join(' · ');
  });

  const tasks=[...document.querySelectorAll('[data-task]')],names=['Read a note','Take a pause','Adjust the view'];
  let highlighted=0,scanTimer=null,breakTimer=null,breakRemaining=0,log=[];
  function highlight(index,announce=true) {
    highlighted=(index+tasks.length)%tasks.length;
    tasks.forEach((task,i)=>{task.classList.toggle('focused',i===highlighted);task.setAttribute('aria-pressed',String(i===highlighted));});
    if(announce) $('task-status').textContent=`${names[highlighted]} is highlighted. ${scanTimer?'Scanning is active.':'Press Select highlighted to open it.'}`;
  }
  function record(command,source) {
    log.unshift(`${new Date().toLocaleTimeString()} · ${source} · ${command}`);log=log.slice(0,8);
    $('command-log').replaceChildren(...log.map(l=>node('li',l)));
  }
  function stopScan(){clearInterval(scanTimer);scanTimer=null;$('scan').textContent='Start scanning';$('scan').setAttribute('aria-pressed','false');}
  function stopBreak(){clearInterval(breakTimer);breakTimer=null;breakRemaining=0;}
  function selectTask(source) {
    stopScan();stopBreak();
    const out=$('task-output');out.replaceChildren();out.hidden=false;
    if(highlighted===0) {
      out.append(node('h4','Experiment reminder'),node('p','Keep task labels and timestamps accurate. Compare performed and imagined actions using real recordings. A visible EEG pattern alone does not establish a usable command.'));
    } else if(highlighted===1) {
      breakRemaining=30;const value=node('p','30 seconds remaining');out.append(node('h4','A short pause'),value);
      breakTimer=setInterval(()=>{breakRemaining--;value.textContent=breakRemaining?`${breakRemaining} seconds remaining`:'Pause complete.';if(!breakRemaining){stopBreak();$('task-status').textContent='Your pause is complete.';}},1000);
    } else {
      document.body.classList.toggle('large-text');
      out.append(node('h4','View adjusted'),node('p',document.body.classList.contains('large-text')?'Larger interface text is enabled. Select this task again to restore the original size.':'Original text size restored.'));
    }
    $('task-status').textContent=`${names[highlighted]} selected using ${source.toLowerCase()}.`;
  }
  function dispatch(command,source='Conventional control') {
    if(command==='next') highlight(highlighted+1);
    else if(command==='previous') highlight(highlighted-1);
    else if(command==='select') selectTask(source);
    else if(command==='cancel') {stopScan();stopBreak();$('task-output').hidden=true;$('task-output').replaceChildren();$('task-status').textContent='Task cancelled. Scanning is stopped.';}
    else if(command==='none') $('task-status').textContent='Simulated uncertain input: no task was executed.';
    else return;
    record(command,source);
  }
  tasks.forEach((task,i)=>{
    task.onclick=()=>{highlight(i,false);dispatch('select');};
    task.onkeydown=e=>{
      if(e.key==='ArrowRight'||e.key==='ArrowDown'){e.preventDefault();highlight(i+1);tasks[highlighted].focus();}
      if(e.key==='ArrowLeft'||e.key==='ArrowUp'){e.preventDefault();highlight(i-1);tasks[highlighted].focus();}
      if(e.key==='Escape'){e.preventDefault();dispatch('cancel');}
    };
  });
  for(const cmd of ['previous','next','select','cancel']) $(cmd).onclick=()=>dispatch(cmd);
  $('simulate').onclick=()=>dispatch($('sim-command').value,'Manual simulation');
  function startScan(){stopScan();scanTimer=setInterval(()=>highlight(highlighted+1),Number($('scan-speed').value));$('scan').textContent='Stop scanning';$('scan').setAttribute('aria-pressed','true');$('task-status').textContent=`Scanning started. ${names[highlighted]} is highlighted.`;}
  $('scan').onclick=()=>{if(scanTimer){stopScan();$('task-status').textContent='Scanning stopped.';}else startScan();};
  $('scan-speed').onchange=()=>{if(scanTimer)startScan();};
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&scanTimer){stopScan();$('task-status').textContent='Scanning paused because the page was hidden.';}});
  window.addEventListener('pagehide',()=>{stopScan();stopBreak();});
  for(const link of document.querySelectorAll('nav a[href^="#"]')) link.addEventListener('click',()=>{
    document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('current',a===link));
  });
})();
