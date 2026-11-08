/* An experimental amplitude detector, not a mental-task classifier. */
(function(root) {
  class EEGAmplitudeTrigger {
    constructor() { this.factor=1.8; this.windowSeconds=.5; this.holdMs=500; this.cooldownMs=2500; this.calibrationMs=6000; this.reset(); }
    reset() {
      this.rate=0;this.channel=0;this.buffer=[];this.lastTimestamp=null;this.lastArrival=null;
      this.baseline=null;this.rms=null;this.ratio=null;this.armed=false;this.calibrating=false;
      this.ready=false;this.aboveSince=null;this.belowSince=null;this.lastFire=-Infinity;
      this.calibration=[];this.calibrationStart=null;this.calibrationSourceStart=null;
      this.message='Connect a live EEG stream to calibrate.';
    }
    configure(rate,channel) { this.reset();this.rate=Number.isFinite(rate)&&rate>=20&&rate<=20000?rate:0;this.channel=channel;this.message=this.rate?'Receiving setup. Wait for samples, then calibrate.':'Unsupported sample rate for amplitude control (requires 20–20000 Hz).'; }
    invalidate(message) {
      this.baseline=null;this.armed=false;this.calibrating=false;this.ready=false;
      this.aboveSince=null;this.belowSince=null;this.buffer=[];this.rms=null;this.ratio=null;
      this.calibration=[];this.calibrationStart=null;this.calibrationSourceStart=null;
      this.message=message;
    }
    calibrate(now) {
      if(!this.rate || this.lastArrival===null || now-this.lastArrival>1500) return false;
      this.invalidate('Calibrating: stay still for 6 seconds.');
      this.calibrating=true;this.calibrationStart=now;return true;
    }
    arm(now) {
      if(!this.baseline || this.lastArrival===null || now-this.lastArrival>1500 || this.calibrating) return false;
      this.armed=true;this.ready=false;this.aboveSince=null;this.belowSince=null;
      this.message='Armed. Return near baseline briefly before a signal can trigger.';return true;
    }
    stop(message='Stopped. No automatic clicks.') { this.armed=false;this.calibrating=false;this.ready=false;this.aboveSince=null;this.belowSince=null;this.message=message; }
    setFactor(value) { if(Number.isFinite(value)&&value>=1.2&&value<=4) { this.factor=value;this.stop('Threshold changed. Arm again when ready.'); } }
    tick(now) {
      if(this.lastArrival!==null && now-this.lastArrival>1500 && (this.armed||this.calibrating||this.baseline)) this.invalidate('Signal stale. Recalibrate after samples resume.');
    }
    feed(rows,stamps,now) {
      if(!Array.isArray(rows)||!Array.isArray(stamps)||rows.length!==stamps.length||!rows.length||!(this.rate>=20&&this.rate<=20000)) return false;
      if(this.lastArrival!==null&&now-this.lastArrival>1500) this.invalidate('Signal gap. Recalibrate before arming.');
      this.lastArrival=now;
      for(let i=0;i<rows.length;i++) {
        const value=rows[i]?.[this.channel], t=stamps[i];
        if(!Number.isFinite(value)||!Number.isFinite(t)|| (this.lastTimestamp!==null && (t<=this.lastTimestamp || t-this.lastTimestamp>Math.max(.1,3/this.rate)))) {
          this.invalidate('Invalid samples or timestamp gap. Recalibrate.');this.lastTimestamp=null;return false;
        }
        this.lastTimestamp=t;this.buffer.push({value,t});
      }
      const end=this.lastTimestamp;
      this.buffer=this.buffer.filter(x=>x.t>end-this.windowSeconds);
      if(this.buffer.length<Math.floor(this.rate*this.windowSeconds*.9)) return false;
      // Stable centered RMS removes DC offset; no units conversion is needed for a ratio.
      let mean=0,m2=0,n=0;
      for(const {value} of this.buffer){n++;const d=value-mean;mean+=d/n;m2+=d*(value-mean);}
      this.rms=Math.sqrt(Math.max(0,m2/n));
      if(this.rms<=1e-15 || !Number.isFinite(this.rms)) {this.invalidate('Flat or invalid signal. Check channel and recalibrate.');return false;}
      if(this.calibrating) {
        if(this.calibrationSourceStart===null)this.calibrationSourceStart=end;
        if(!this.calibration.length || end-this.calibration.at(-1).t>=.09)this.calibration.push({t:end,value:this.rms});
        const elapsed=now-this.calibrationStart;
        this.message=`Calibrating: ${Math.min(6,Math.floor(elapsed/1000))}/6 seconds. Stay still.`;
        if(elapsed>=this.calibrationMs && end-this.calibrationSourceStart>=5.5 && this.calibration.length>=30) {
          const values=this.calibration.map(x=>x.value).sort((a,b)=>a-b);
          this.baseline=values[Math.floor(values.length/2)];this.calibrating=false;this.ratio=this.rms/this.baseline;
          this.message='Calibration complete. Review the live level, then Arm tab clicks.';
        }
        return false;
      }
      if(!this.baseline)return false;
      this.ratio=this.rms/this.baseline;
      if(!this.armed)return false;
      const release=1+(this.factor-1)*.4;
      if(this.ratio<=release) {
        this.aboveSince=null;
        if(this.belowSince===null)this.belowSince=now;
        if(now-this.belowSince>=500 && now-this.lastFire>=this.cooldownMs)this.ready=true;
        this.message=this.ready?'Armed: waiting for a sustained amplitude rise.':'Return near baseline; waiting for cooldown.';
      } else {
        this.belowSince=null;
        if(this.ready&&this.ratio>=this.factor) {
          if(this.aboveSince===null)this.aboveSince=now;
          this.message='Amplitude above threshold; checking sustained rise…';
          if(now-this.aboveSince>=this.holdMs && now-this.lastFire>=this.cooldownMs) {
            this.ready=false;this.aboveSince=null;this.lastFire=now;
            this.message='Activation detected. Return near baseline before another click.';return true;
          }
        } else {this.aboveSince=null;this.message=this.ready?'Armed: below trigger threshold.':'Return near baseline before another click.';}
      }
      return false;
    }
  }
  root.EEGAmplitudeTrigger=EEGAmplitudeTrigger;
  if(typeof module!=='undefined'&&module.exports)module.exports=EEGAmplitudeTrigger;
})(typeof window!=='undefined'?window:globalThis);
