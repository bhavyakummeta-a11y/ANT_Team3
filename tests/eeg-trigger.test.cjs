const test=require('node:test');
const assert=require('node:assert/strict');
const Trigger=require('../eeg-trigger.js');
function source(){
 const engine=new Trigger();engine.configure(100,0);
 let now=10000,t=100;
 const push=(amp=1e-5,duration=100,offset=0)=>{
  const rows=[],stamps=[];
  for(let i=0;i<duration/10;i++){t+=.01;rows.push([offset+amp*Math.sin(2*Math.PI*10*t)]);stamps.push(t);}
  now+=duration;return engine.feed(rows,stamps,now);
 };
 const run=(n,amp)=>Array.from({length:n},()=>push(amp)).filter(Boolean).length;
 const calibrate=()=>{run(10,1e-5);assert.ok(engine.calibrate(now));run(70,1e-5);assert.ok(engine.baseline);};
 return {engine,push,run,calibrate,get now(){return now},get timestamp(){return t}};
}
test('uncalibrated or unarmed data cannot cause clicks',()=>{
 const s=source();assert.equal(s.engine.arm(s.now),false);assert.equal(s.run(20,3e-5),0);s.calibrate();assert.equal(s.run(20,3e-5),0);
});
test('sustained rise triggers once and requires release plus cooldown',()=>{
 const s=source();s.calibrate();assert.ok(s.engine.arm(s.now));s.run(10,1e-5);
 assert.equal(s.run(20,3e-5),1);assert.equal(s.run(50,3e-5),0);
 s.run(30,1e-5);assert.equal(s.run(20,3e-5),1);
});
test('brief peak does not trigger',()=>{
 const s=source();s.calibrate();s.engine.arm(s.now);s.run(10,1e-5);
 assert.equal(s.push(3e-5,10),false);assert.equal(s.run(15,1e-5),0);
});
test('stale data, channel changes, and malformed data invalidate calibration',()=>{
 for(const fault of ['stale','channel','invalid','timestamps']){
  const s=source();s.calibrate();s.engine.arm(s.now);
  if(fault==='stale')s.engine.tick(s.now+1600);
  if(fault==='channel')s.engine.configure(100,1);
  if(fault==='invalid')s.engine.feed([[null]],[s.timestamp+.01],s.now+10);
  if(fault==='timestamps')s.engine.feed([[1e-5]],[s.timestamp-1],s.now+10);
  assert.equal(s.engine.armed,false,fault);assert.equal(s.engine.baseline,null,fault);
 }
});
test('DC offset does not inflate centered RMS',()=>{
 const s=source();s.calibrate();s.engine.arm(s.now);s.run(10,1e-5);
 // Settle a DC offset while stopped, then verify the centered steady level.
 s.engine.stop();for(let i=0;i<10;i++)s.push(1e-5,100,.01);
 assert.ok(Math.abs(s.engine.ratio-1)<.01);
});
test('threshold changes and stop require explicit rearming',()=>{
 const s=source();s.calibrate();s.engine.arm(s.now);s.engine.setFactor(1.5);
 assert.equal(s.engine.armed,false);s.engine.arm(s.now);s.engine.stop();assert.equal(s.run(20,3e-5),0);
});
test('flat data cannot calibrate',()=>{const s=source();s.run(10,1e-5);s.engine.calibrate(s.now);s.run(70,0);assert.equal(s.engine.baseline,null);});
