(() => {
  const panel = document.createElement('section');
  panel.style.cssText = 'margin:20px;padding:20px;border:1px solid #348ba1;border-radius:8px;background:#101a23;color:#e2edf5;font:14px system-ui;';
  panel.innerHTML = `
    <h2 style="margin:0 0 10px">Live EEG connection</h2>
    <p>Find the eego stream, select it, and connect. This panel plots received samples; existing demo charts remain simulated.</p>
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin:14px 0">
      <button class="btn" id="eeg-find">Find EEG streams</button>
      <label>EEG stream <select id="eeg-stream" style="max-width:420px"><option value="">Find streams first</option></select></label>
      <button class="btn primary" id="eeg-connect" disabled>Connect</button>
      <button class="btn" id="eeg-disconnect" disabled>Disconnect</button>
    </div>
    <p id="eeg-status" role="status" aria-live="polite">Disconnected. Open this page through the Python bridge on port 8765.</p>
    <p id="eeg-meta"></p>
    <label>Display channel <select id="eeg-channel" style="max-width:220px" disabled></select></label>
    <canvas id="eeg-canvas" height="180" style="display:block;width:100%;height:180px;margin-top:12px" aria-label="Received EEG samples, auto-scaled"></canvas>
    <p id="eeg-values">No samples received.</p>
    <small>Last 5 seconds · auto-scaled amplitude · source units · experimental amplitude trigger available below</small>`;
  (document.querySelector('.main') || document.body).prepend(panel);
  const el = id => panel.querySelector('#eeg-' + id);
  const trigger = new window.EEGAmplitudeTrigger();
  const tabs = [...document.querySelectorAll('nav a[href^="#"]')].filter(a=>document.querySelector(a.getAttribute('href')));
  if(!tabs.length) tabs.push(...document.querySelectorAll('.nav[data-page]'));
  const control=document.createElement('div');
  control.style.cssText='margin-top:18px;padding-top:16px;border-top:1px solid #344a59';
  control.innerHTML=`<h3>Experimental EEG tab clicks</h3>
    <p>A sustained rise in the selected channel’s centered RMS amplitude can click a website tab. Blinks, movement and noise can also trigger this; it is not thought recognition.</p>
    <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:end">
      <label>On activation<select id="eeg-target"><option value="next">Click next website tab</option></select></label>
      <label>Trigger level<select id="eeg-threshold"><option value="1.2">1.2× baseline (sensitive)</option><option value="1.5">1.5× baseline</option><option value="1.8" selected>1.8× baseline</option><option value="2.5">2.5× baseline</option><option value="4">4× baseline</option></select></label>
      <button class="btn" id="eeg-calibrate" disabled>Calibrate · 6 seconds</button>
      <button class="btn primary" id="eeg-arm" disabled>Arm tab clicks</button>
      <button class="btn" id="eeg-stop">Stop / Escape</button>
    </div>
    <p id="eeg-trigger-status" role="status" aria-live="polite">Connect a live EEG stream to calibrate.</p>
    <progress id="eeg-level" max="4" value="0" style="width:100%;height:18px" aria-label="Amplitude relative to calibrated baseline"></progress>
    <p id="eeg-level-text">No calibration yet.</p>
    <p id="eeg-click-log">No EEG-triggered clicks.</p>
    <small>0.5 s signal window · 0.5 s sustained rise · 2.5 s cooldown · return to baseline between clicks · only this website’s tabs</small>`;
  panel.append(control);
  tabs.forEach((tab,i)=>el('target').add(new Option(tab.textContent.trim(),String(i))));
  const emergency=document.createElement('button');
  emergency.textContent='EEG tab clicks ARMED — STOP';emergency.hidden=true;
  emergency.style.cssText='position:fixed;right:18px;bottom:18px;z-index:10000;padding:14px;background:#82373b;color:white;border:2px solid white;border-radius:7px;max-width:calc(100vw - 36px)';
  document.body.append(emergency);
  let clickCount=0;
  function refreshTrigger(){
    const now=Date.now();trigger.tick(now);
    if(el('trigger-status').textContent!==trigger.message)el('trigger-status').textContent=trigger.message;
    el('calibrate').disabled=!socket||socket.readyState!==WebSocket.OPEN||!trigger.rate||!lastArrival||now-lastArrival>1500||trigger.calibrating;
    el('arm').disabled=!trigger.baseline||trigger.calibrating||trigger.armed||!tabs.length;
    el('level').value=Math.min(4,trigger.ratio||0);
    el('level-text').textContent=trigger.baseline?`Live amplitude: ${(trigger.ratio||0).toFixed(2)}× baseline · trigger: ${trigger.factor.toFixed(1)}× · ${channelInfo[channel]?.label || 'selected channel'}`:'Calibrate the selected channel before enabling clicks.';
    emergency.hidden=!trigger.armed;
  }
  function stopTrigger(){trigger.stop();refreshTrigger();}
  el('calibrate').onclick=()=>{trigger.calibrate(Date.now());refreshTrigger();};
  el('arm').onclick=()=>{trigger.arm(Date.now());refreshTrigger();};
  el('stop').onclick=stopTrigger;emergency.onclick=stopTrigger;
  el('threshold').onchange=()=>{trigger.setFactor(Number(el('threshold').value));refreshTrigger();};
  el('target').onchange=()=>{trigger.stop('Target changed. Arm again when ready.');refreshTrigger();};
  document.addEventListener('keydown',e=>{if(e.key==='Escape')stopTrigger();});
  document.addEventListener('visibilitychange',()=>{if(document.hidden){trigger.stop('Page hidden: automatic clicks stopped.');refreshTrigger();}});
  function clickTab(){
    if(document.hidden||!tabs.length)return;
    const current=tabs.findIndex(tab=>tab.classList.contains('current')||tab.classList.contains('active'));
    const target=el('target').value==='next'?(current+1)%tabs.length:Number(el('target').value);
    const tab=tabs[target];if(!tab)return;
    tab.click();clickCount++;
    el('click-log').textContent=`${clickCount} EEG-triggered click(s) · ${tab.textContent.trim()} · amplitude ${trigger.ratio.toFixed(2)}× baseline · ${new Date().toLocaleTimeString()}`;
  }
  let socket = null, points = [], channel = 0, channelInfo = [], count = 0, lastArrival = 0, lastStamp = null;
  let terminalMessage = '', rate = 0;
  const status = text => { if(el('status').textContent!==text)el('status').textContent = text; };
  function buttons(active) {
    el('find').disabled = active;
    el('stream').disabled = active;
    el('connect').disabled = active || !el('stream').value;
    el('disconnect').disabled = !active;
  }
  el('stream').onchange = () => buttons(false);
  el('find').onclick = async () => {
    el('find').disabled = true;
    el('connect').disabled = true;
    el('stream').replaceChildren(new Option('Searching…', ''));
    status('Searching for EEG streams (about 3 seconds)…');
    try {
      const response = await fetch('/api/streams', {signal: AbortSignal.timeout(12000)});
      if (!response.ok) throw new Error('Bridge unavailable');
      const data = await response.json();
      if (!Array.isArray(data.streams)) throw new Error('Invalid bridge response');
      el('stream').replaceChildren();
      for (const s of data.streams) {
        el('stream').add(new Option(`${s.name} — ${s.channels} channels, ${s.sample_rate} Hz (${s.hostname})`, s.id));
      }
      if (!data.streams.length) el('stream').add(new Option('No EEG streams found', ''));
      status(data.streams.length ? 'Select the eego stream, then Connect.' : 'No EEG streams found. Check eego acquisition and LSL streaming. Try running the bridge on the Windows eego computer.');
    } catch (_) {
      el('stream').replaceChildren(new Option('Bridge unavailable', ''));
      status('Cannot reach the Python bridge. Start bridge/server.py and open http://localhost:8765 on that computer, or http://WINDOWS-IP:8765 from your Mac.');
    } finally { buttons(false); }
  };
  el('connect').onclick = () => {
    trigger.reset();
    points = []; count = 0; lastArrival = 0; lastStamp = null; terminalMessage = ''; channelInfo = [];
    el('meta').textContent = ''; el('channel').replaceChildren(); el('channel').disabled = true;
    buttons(true); status('Connecting to the selected LSL stream…');
    const current = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/eeg?id=${encodeURIComponent(el('stream').value)}`);
    socket = current;
    current.onmessage = event => {
      if (socket !== current) return;
      const data = JSON.parse(event.data);
      if (data.type === 'connected') {
        rate = data.stream.sample_rate;
        channelInfo = data.channels;
        el('channel').replaceChildren(...channelInfo.map((c,i) => new Option(`${c.label} (${c.unit})`, String(i))));
        channel = Math.max(0, channelInfo.findIndex(c => c.label === 'C3'));
        el('channel').value = String(channel); el('channel').disabled = false;
        el('meta').textContent = `${data.stream.name} · ${data.stream.channels} channels · ${rate} Hz`;
        trigger.configure(rate,channel);
        status('Stream opened. Waiting for samples…');
      } else if (data.type === 'samples') {
        lastArrival = Date.now(); count += data.samples.length;
        data.samples.forEach((row,i) => points.push({row, t:data.timestamps[i]}));
        lastStamp = data.timestamps.at(-1);
        while (points.length && (points[0].t < lastStamp - 5 || points.length > 100000)) points.shift();
        if(trigger.feed(data.samples,data.timestamps,Date.now()))clickTab();
        refreshTrigger();
        status('Receiving live samples from the selected LSL source.');
      } else if (data.type === 'waiting') {
        status(data.received
          ? `LSL delivery paused for ${data.seconds}s; ${data.received} samples received.`
          : `LSL stream opened; waiting for the first EEG samples (${data.seconds}/${data.timeout}s).`);
      } else if (data.type === 'error') {
        trigger.invalidate('Stream error. Reconnect and recalibrate.');refreshTrigger();
        terminalMessage = data.message; status(terminalMessage);
      }
    };
    current.onerror = () => { trigger.invalidate('Connection error. Reconnect and recalibrate.');refreshTrigger(); terminalMessage = 'Connection failed. Check the bridge, then find streams and reconnect.'; status(terminalMessage); };
    current.onclose = () => {
      if (socket !== current) return;
      trigger.invalidate('Disconnected. Reconnect and recalibrate.');refreshTrigger();
      socket = null; buttons(false);
      status(terminalMessage || 'Disconnected. Last received samples remain visible.');
    };
  };
  el('disconnect').onclick = () => {
    trigger.invalidate('Disconnected. Reconnect and recalibrate.');refreshTrigger();
    terminalMessage = 'Disconnected. Last received samples remain visible.';
    if (socket) socket.close();
  };
  el('channel').onchange = () => { channel = Number(el('channel').value); trigger.configure(rate,channel);refreshTrigger(); };
  window.addEventListener('pagehide', () => { if (socket) socket.close(); });
  setInterval(() => {
    refreshTrigger();
    if (socket && !terminalMessage && lastArrival && Date.now() - lastArrival > 2000) status('Signal stale: the LSL connection has not delivered recent samples.');
    const canvas = el('canvas'), w = Math.max(1, canvas.clientWidth), h = 180;
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d'); ctx.strokeStyle = '#273b4a';
    ctx.beginPath(); ctx.moveTo(0,h/2); ctx.lineTo(w,h/2); ctx.stroke();
    const values = points.map(p => p.row[channel]).filter(Number.isFinite);
    if (!values.length) { el('values').textContent = 'No samples received for this channel.'; return; }
    let lo = Infinity, hi = -Infinity;
    for (const v of values) { lo = Math.min(lo,v); hi = Math.max(hi,v); }
    const span = hi-lo || 1;
    ctx.strokeStyle = '#50c9df'; ctx.beginPath(); let started = false;
    for (const p of points) {
      const v = p.row[channel];
      if (!Number.isFinite(v)) { started = false; continue; }
      const x = (p.t - (lastStamp-5))/5*w, y = h-15-(v-lo)/span*(h-30);
      if (!started) ctx.moveTo(x,y); else ctx.lineTo(x,y); started = true;
    }
    ctx.stroke();
    el('values').textContent = `${count.toLocaleString()} samples received · ${channelInfo[channel]?.label || 'Channel'} · range ${lo.toPrecision(4)} to ${hi.toPrecision(4)} ${channelInfo[channel]?.unit || ''} · LSL timestamp ${lastStamp?.toFixed(3)}`;
  }, 100);
})();
