/* SeCien 2025 — Regresión Estatura–Peso (GitHub Pages compatible)
   MUESTREO CONTROLADO POR MANO: sólo captura cuando hay 5 dedos levantados,
   una sola muestra por cada “levantada” (debounce por estabilidad temporal).
*/

"use strict";

// ==== Utilidad DOM ====
const $ = (id) => document.getElementById(id);

// Elementos UI
const video   = $("video");
const canvas  = $("overlay");
const ctx     = canvas.getContext("2d", { alpha: true });
const btnStart= $("btnStart"), btnReset=$("btnReset"), btnCsv=$("btnCsv");
const statusEl= document.querySelector(".camera-header .status");

// KPIs
const kNSamples=$("nSamples"), kHLast=$("hLast"), kWLast=$("wLast"),
      kBmiLast=$("bmiLast"), kPxcm=$("pxcm"), kFps=$("fps"), r2Span=$("r2val");

// ==== Parámetros ====
const MAX_HISTORY        = 1000;
const CAL_HAND_CM        = 15.0;   // índice–meñique ≈ 15 cm
const CAL_SPAN_MIN_PX    = 30;
const HEIGHT_RANGE       = [120, 200]; // cm
const WEIGHT_RANGE       = [25, 150];  // kg
const MP_VERSION         = "0.10.7";

// Sólo cuando hay 5 dedos visibles:
const HOLD_MIN_MS        = 600;    // estabilidad mínima de la mano con 5 dedos
const SAMPLE_GAP_MS      = 400;    // separación mínima adicional entre muestras (seguridad)

let FilesetResolver=null, HandLandmarker=null, PoseLandmarker=null;
let handLandmarker=null, poseLandmarker=null;

// Estado
let running=false, frameCount=0, lastTime=performance.now(), fps=0;
let scalePxPerCm = null;
let heightCalib=1.0, weightCalib=1.0;
let mediaStream = null;

// Datos
const samples = []; // {t,h,w,bmi,pxcm}

// Puerta de muestreo por “levantada”
let fiveUp=false;            // ¿se detecta 5 dedos en este instante?
let fiveUpSince=0;           // instante desde el que se detectan 5 dedos de forma continua
let sampledThisHold=false;   // ¿ya se tomó muestra en esta “levantada”?
let lastSampleAt=0;          // última muestra (ms)

// ==== Utilidades ====
function setStatus(msg, cls=""){ statusEl.textContent=msg; statusEl.className=`status ${cls}`; }
function clip(v,[a,b]){ return Math.max(a, Math.min(b, v)); }
function updateFps(){
  frameCount++; const now=performance.now();
  if(now-lastTime>=1000){ fps=frameCount; frameCount=0; lastTime=now; kFps.textContent=String(fps); }
}

// ==== Cámara / Layout ====
async function startCamera(){
  setStatus("Solicitando cámara…","warn");
  mediaStream = await navigator.mediaDevices.getUserMedia({
    video:{width:{ideal:1280},height:{ideal:720},facingMode:{ideal:"user"}}, audio:false
  });
  video.srcObject=mediaStream;
  await video.play();
  resizeCanvas();
  setStatus("Cámara iniciada","ok");
}
function stopCamera(){
  if(mediaStream){
    mediaStream.getTracks().forEach(t=>t.stop());
    mediaStream = null;
  }
}
function resizeCanvas(){
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(640, Math.round(rect.width));
  canvas.height= Math.max(360, Math.round(rect.height));
}
new ResizeObserver(resizeCanvas).observe(document.querySelector(".camera-stage"));

// ==== Carga MediaPipe (ESM con fallback y ?module en unpkg) ====
async function importTasksVision(){
  setStatus("Importando @mediapipe/tasks-vision…","warn");
  let mod=null, err;
  const tries = [
    `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`,
    `https://unpkg.com/@mediapipe/tasks-vision@${MP_VERSION}?module`
  ];
  for (const u of tries) {
    try { mod = await import(u); setStatus(`Importado desde ${new URL(u).host}`,"ok"); break; }
    catch (e) { err = e; }
  }
  if (!mod) { setStatus("No se pudo importar tasks-vision.","err"); throw err || new Error("ImportFailed"); }
  ({ FilesetResolver, HandLandmarker, PoseLandmarker } = mod);
}

async function initModels(){
  setStatus("Cargando modelos y WASM…","warn");
  const vision=await FilesetResolver.forVisionTasks(
    `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision,{
    baseOptions:{ modelAssetPath:"https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" },
    runningMode:"VIDEO", numHands:2,
    minHandDetectionConfidence:.5, minHandPresenceConfidence:.5, minTrackingConfidence:.5
  });
  // CPU por robustez en Pages
  poseLandmarker = await PoseLandmarker.createFromOptions(vision,{
    baseOptions:{ modelAssetPath:"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" },
    runningMode:"VIDEO", numPoses:1,
    minPoseDetectionConfidence:.5, minPosePresenceConfidence:.5, minTrackingConfidence:.5
  });
  setStatus("Modelos listos","ok");
}

// ==== Dibujo ====
const POSE_CONNECTIONS=[[11,12],[11,23],[12,24],[23,24],[11,13],[13,15],[12,14],[14,16],[23,25],[25,27],[27,29],[29,31],[24,26],[26,28],[28,30],[30,32]];
function drawHand(points){
  ctx.fillStyle="#2ecc71";
  for(const p of points){ ctx.beginPath(); ctx.arc(p.x*canvas.width,p.y*canvas.height,3,0,Math.PI*2); ctx.fill(); }
}
function drawPoseSkeleton(landmarks){
  ctx.lineWidth=3; ctx.strokeStyle="#9ad1ff"; ctx.beginPath();
  for(const [a,b] of POSE_CONNECTIONS){
    const p=landmarks[a], q=landmarks[b]; if(!p||!q) continue;
    ctx.moveTo(p.x*canvas.width,p.y*canvas.height);
    ctx.lineTo(q.x*canvas.width,q.y*canvas.height);
  }
  ctx.stroke();
}

// ==== Medición / Calibración ====
function countFingersForHand(lmk,handed="Unknown"){
  if(!lmk||lmk.length!==21) return 0;
  const TIP={thumb:4,index:8,middle:12,ring:16,pinky:20};
  const PIP={index:6,middle:10,ring:14,pinky:18};
  const MCP={thumb:2,index:5,middle:9,ring:13,pinky:17};
  let count=0;
  for(const f of ['index','middle','ring','pinky']){
    const tip=lmk[TIP[f]], pip=lmk[PIP[f]], mcp=lmk[MCP[f]];
    if(tip.y<pip.y && tip.y<mcp.y) count++;
  }
  const t4=lmk[TIP.thumb], t3=lmk[3], tmcp=lmk[MCP.thumb];
  if(handed==="Right"){ if(t4.x>t3.x && t4.x>tmcp.x) count++; }
  else if(handed==="Left"){ if(t4.x<t3.x && t4.x<tmcp.x) count++; }
  else { if(Math.abs(t4.x-t3.x)>0.05) count++; }
  return count;
}
function handSpanPx(lmk){
  const i = lmk[8], p = lmk[20];
  const dx = (i.x - p.x) * canvas.width;
  const dy = (i.y - p.y) * canvas.height;
  return Math.hypot(dx, dy);
}
function calcHeightCm(poseLmk){
  const headIdx=[0,1,2,3,4,5,6,7,8], footIdx=[27,28,29,30,31,32];
  const ysHead = headIdx.map(i => poseLmk[i]?.y).filter(v => v!==undefined);
  const ysFoot = footIdx.map(i => poseLmk[i]?.y).filter(v => v!==undefined);
  if(!ysHead.length || !ysFoot.length || !scalePxPerCm) return null;
  const headY = Math.min(...ysHead), footY = Math.max(...ysFoot);
  const px = (footY - headY) * canvas.height;
  return (px / scalePxPerCm) * heightCalib;
}
function calcWidthCm(poseLmk){
  const L11=poseLmk[11], R12=poseLmk[12], L23=poseLmk[23], R24=poseLmk[24];
  if(!(L11&&R12&&L23&&R24) || !scalePxPerCm) return null;
  const sh = Math.abs((L11.x - R12.x) * canvas.width);
  const wa = Math.abs((L23.x - R24.x) * canvas.width);
  return ((sh + wa) / 2) / scalePxPerCm;
}
function estimateWeightKg(heightCm, widthCm){
  return (1.8 * widthCm + 0.25 * heightCm - 30) * weightCalib;
}

// ==== Gráficos (Chart.js) ====
const hwChart = new Chart($("hwChart"), {
  type: 'scatter',
  data: {
    datasets: [
      {label:'Muestras', data: [], pointRadius:4, backgroundColor:'#f9a826'},
      {label:'Regresión', data: [], type:'line', borderWidth:2, borderColor:'#ff6b6b', pointRadius:0, fill:false}
    ]
  },
  options: {
    responsive:true, maintainAspectRatio:false,
    scales:{
      x:{title:{display:true,text:'Estatura (cm)',color:'#f8fafc'}, ticks:{color:'#f8fafc'}, grid:{color:'rgba(255,255,255,.1)'}},
      y:{title:{display:true,text:'Peso (kg)',color:'#f8fafc'}, ticks:{color:'#f8fafc'}, grid:{color:'rgba(255,255,255,.1)'}}
    },
    plugins:{legend:{labels:{color:'#f8fafc'}}}
  }
});
const hBins = Array.from({length:8},(_,k)=>120+10*k); // 120..190
const wBins = Array.from({length:11},(_,k)=>25+10*k); // 25..135
const hHist = new Chart($("hHist"),{
  type:'bar',
  data:{labels:hBins.map(v=>`${v}–${v+10}`),datasets:[{label:'Frecuencia',data:Array(hBins.length).fill(0),backgroundColor:'#4ea1ff'}]},
  options:{responsive:true,maintainAspectRatio:false,scales:{x:{ticks:{color:'#f8fafc'},grid:{display:false}},y:{beginAtZero:true,ticks:{color:'#f8fafc'},grid:{color:'rgba(255,255,255,.1)'}}},plugins:{legend:{display:false}}}
});
const wHist = new Chart($("wHist"),{
  type:'bar',
  data:{labels:wBins.map(v=>`${v}–${v+10}`),datasets:[{label:'Frecuencia',data:Array(wBins.length).fill(0),backgroundColor:'#22c55e'}]},
  options:{responsive:true,maintainAspectRatio:false,scales:{x:{ticks:{color:'#f8fafc'},grid:{display:false}},y:{beginAtZero:true,ticks:{color:'#f8fafc'},grid:{color:'rgba(255,255,255,.1)'}}},plugins:{legend:{display:false}}}
});

function linearRegression(xs, ys){
  const n = xs.length; if(n<2) return null;
  const mx = xs.reduce((s,v)=>s+v,0)/n, my = ys.reduce((s,v)=>s+v,0)/n;
  let num=0, den=0, ssRes=0, ssTot=0;
  for(let i=0;i<n;i++){ const dx=xs[i]-mx; num += dx*(ys[i]-my); den += dx*dx; }
  const b1 = den===0 ? 0 : num/den;
  const b0 = my - b1*mx;
  for(let i=0;i<n;i++){ const yhat=b0+b1*xs[i]; ssRes += (ys[i]-yhat)**2; ssTot += (ys[i]-my)**2; }
  const r2 = ssTot===0 ? 1 : 1 - ssRes/ssTot;
  return {b0,b1,r2};
}
function updateCharts(){
  const pts = samples.map(s => ({x:s.h, y:s.w}));
  hwChart.data.datasets[0].data = pts;
  if(samples.length>=2){
    const xs=samples.map(s=>s.h), ys=samples.map(s=>s.w);
    const lr = linearRegression(xs,ys);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const y1 = lr.b0 + lr.b1 * xMin, y2 = lr.b0 + lr.b1 * xMax;
    hwChart.data.datasets[1].data = [{x:xMin,y:y1},{x:xMax,y:y2}];
    r2Span.textContent = lr.r2.toFixed(2);
  }else{
    hwChart.data.datasets[1].data = [];
    r2Span.textContent = "–";
  }
  hwChart.update('none');

  const hCounts = Array(hBins.length).fill(0);
  const wCounts = Array(wBins.length).fill(0);
  for(const s of samples){
    if(s.h>=120 && s.h<200){
      const hi = Math.min(hBins.length-1, Math.max(0, Math.floor((s.h-120)/10)));
      hCounts[hi]++;
    }
    if(s.w>=25 && s.w<145){
      const wi = Math.min(wBins.length-1, Math.max(0, Math.floor((s.w-25)/10)));
      wCounts[wi]++;
    }
  }
  hHist.data.datasets[0].data = hCounts; hHist.update('none');
  wHist.data.datasets[0].data = wCounts; wHist.update('none');
}

// ==== Bucle principal ====
async function loop(){
  if(!running) return;
  updateFps(); ctx.clearRect(0,0,canvas.width,canvas.height);
  const now = performance.now();

  // --------- Manos: calibración y detección de “5 dedos” ---------
  let fiveDetectedThisFrame = false;
  if(handLandmarker){
    try{
      const rh=handLandmarker.detectForVideo(video,now);
      if(rh){
        const L=rh.landmarks||rh.handLandmarks||[], H=rh.handedness||rh.handednesses||[];
        for(let i=0;i<L.length;i++){
          drawHand(L[i]);
          const handed = (H[i]&&H[i][0]&&H[i][0].categoryName)?H[i][0].categoryName:"Unknown";
          const f = countFingersForHand(L[i], handed);

          // Calibración px/cm con la misma condición (mano con 5 dedos)
          if(f===5){
            fiveDetectedThisFrame = true;
            const span = handSpanPx(L[i]);
            if(span>CAL_SPAN_MIN_PX){
              scalePxPerCm = span / CAL_HAND_CM;
              kPxcm.textContent = scalePxPerCm.toFixed(2);
            }
          }
        }
      }
    }catch(_){}
  }

  // Actualiza puerta de muestreo por levantada
  if(fiveDetectedThisFrame){
    if(!fiveUp){ fiveUp=true; fiveUpSince=now; sampledThisHold=false; }
  }else{
    fiveUp=false; fiveUpSince=0; sampledThisHold=false;
  }

  // --------- Pose → altura/peso (si calibrado) ---------
  let heightCm=null, widthCm=null, weightKg=null, bmi=null;
  if(poseLandmarker){
    try{
      const rp=poseLandmarker.detectForVideo(video,now);
      if(rp?.landmarks?.length){
        const lmk = rp.landmarks[0];
        drawPoseSkeleton(lmk);
        if(scalePxPerCm){
          heightCm = calcHeightCm(lmk);
          widthCm  = calcWidthCm(lmk);
          if(heightCm && widthCm){
            heightCm = clip(heightCm, HEIGHT_RANGE);
            weightKg = clip(estimateWeightKg(heightCm, widthCm), WEIGHT_RANGE);
            bmi = weightKg / ((heightCm/100)**2);
          }
        }
      }
    }catch(_){}
  }

  // --------- Muestreo: sólo si hay 5 dedos y estabilidad temporal ---------
  const stableFive = fiveUp && (now - fiveUpSince >= HOLD_MIN_MS);
  const cooldownOk = (now - lastSampleAt >= SAMPLE_GAP_MS);
  const eligible = scalePxPerCm && heightCm && weightKg && stableFive && !sampledThisHold && cooldownOk;

  if(eligible){
    const rec = { t:new Date().toISOString(), h:heightCm, w:weightKg, bmi:bmi, pxcm:scalePxPerCm };
    samples.push(rec);
    if(samples.length>MAX_HISTORY) samples.splice(0, samples.length - MAX_HISTORY);
    kNSamples.textContent=String(samples.length);
    kHLast.textContent = heightCm.toFixed(1);
    kWLast.textContent = weightKg.toFixed(1);
    kBmiLast.textContent = bmi.toFixed(1);
    updateCharts();
    sampledThisHold = true;
    lastSampleAt = now;
  }

  requestAnimationFrame(loop);
}

// ==== CSV ====
function downloadCsv(){
  if(!samples.length) return;
  const header = "timestamp,estatura_cm,peso_kg,imc,px_por_cm\n";
  const rows = samples.map(s=>[s.t,s.h.toFixed(2),s.w.toFixed(2),s.bmi.toFixed(2),s.pxcm.toFixed(4)].join(",")).join("\n");
  const blob = new Blob([header+rows],{type:"text/csv;charset=utf-8;"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "regresion_estatura_peso.csv";
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ==== Eventos ====
btnStart.addEventListener("click", async ()=>{
  btnStart.disabled=true; btnStart.innerHTML='<i class="fas fa-spinner fa-spin"></i> Iniciando…';
  try{
    await startCamera();
    await importTasksVision();
    await initModels();
    running=true; requestAnimationFrame(loop);
    btnStart.innerHTML='<i class="fas fa-circle-stop"></i> Detener';
    btnStart.disabled=false;

    const stopHandler = ()=>{
      running=false; stopCamera();
      btnStart.removeEventListener("click", stopHandler);
      btnStart.innerHTML='<i class="fas fa-video"></i> Iniciar';
      setStatus("Detenido","warn");
      // Reset de la puerta por si quedaba activa
      fiveUp=false; fiveUpSince=0; sampledThisHold=false;
    };
    btnStart.addEventListener("click", stopHandler, { once:true });
  }catch(e){
    setStatus(`Error: ${e?.message||e}`,"err");
    btnStart.disabled=false; btnStart.innerHTML='<i class="fas fa-video"></i> Iniciar';
    stopCamera();
  }
});

btnReset.addEventListener("click", ()=>{
  samples.length=0;
  kNSamples.textContent="0"; kHLast.textContent=kWLast.textContent=kBmiLast.textContent="–";
  hwChart.data.datasets[0].data=[]; hwChart.data.datasets[1].data=[]; hwChart.update();
  hHist.data.datasets[0].data=Array(hBins.length).fill(0); hHist.update();
  wHist.data.datasets[0].data=Array(wBins.length).fill(0); wHist.update();
  // Reset de control de mano
  fiveUp=false; fiveUpSince=0; sampledThisHold=false;
  setStatus("Datos reiniciados","warn");
});

btnCsv.addEventListener("click", downloadCsv);

// Parar cámara si se oculta la pestaña
document.addEventListener("visibilitychange", ()=>{ if(document.hidden){ running=false; stopCamera(); }});
