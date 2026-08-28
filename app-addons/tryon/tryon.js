'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const ui = { start:$('startCard'), editor:$('editor'), camera:$('cameraInput'), gallery:$('galleryInput'), model:$('modelStatus'), canvas:$('resultCanvas'), busy:$('busy'), color:$('colorInput'), code:$('colorCode'), opacity:$('opacity'), opacityValue:$('opacityValue'), threshold:$('threshold'), thresholdValue:$('thresholdValue'), showMask:$('showMask'), status:$('editorStatus'), toast:$('toast'), compare:$('compareBtn'), palette:$('palette') };
  const colors = ['#F5D0C5','#D98A91','#F04479','#D81B60','#A81748','#8B2F67','#7446B8','#335CC7','#1597A5','#3BAA70','#D6A522','#17171B'];
  let session, sourceBitmap, sourceImage, probabilities, geometry, showingOriginal = false;

  function setStatus(el, text, kind=''){ el.textContent=text; el.className='status '+kind; }
  function toast(text){ ui.toast.textContent=text; ui.toast.classList.remove('hidden'); clearTimeout(toast.timer); toast.timer=setTimeout(()=>ui.toast.classList.add('hidden'),2400); }
  function selectedColor(hex){ ui.color.value=hex; ui.code.textContent=hex.toUpperCase(); document.querySelectorAll('.swatch').forEach(x=>x.classList.toggle('active',x.dataset.color.toLowerCase()===hex.toLowerCase())); render(); }
  colors.forEach((color,i)=>{ const b=document.createElement('button'); b.type='button'; b.className='swatch'+(i===3?' active':''); b.style.background=color; b.dataset.color=color; b.setAttribute('aria-label','Цвет '+color); b.onclick=()=>selectedColor(color); ui.palette.appendChild(b); });

  async function initModel(){
    try {
      if (!window.ort) throw new Error('модуль ONNX Runtime не загрузился');
      ort.env.wasm.wasmPaths='./'; ort.env.wasm.numThreads=1; ort.env.wasm.simd=true;
      session=await ort.InferenceSession.create('./nail-unet.onnx',{executionProviders:['wasm'],graphOptimizationLevel:'all'});
      setStatus(ui.model,'Распознавание готово','ok');
    } catch(e){ console.error(e); setStatus(ui.model,'Не удалось загрузить модель: '+e.message,'error'); }
  }

  function decodeWithImage(file){
    return new Promise((resolve,reject)=>{
      const url=URL.createObjectURL(file),image=new Image();
      image.onload=()=>{URL.revokeObjectURL(url);resolve(image);};
      image.onerror=()=>{URL.revokeObjectURL(url);reject(new Error('формат фотографии не поддерживается'));};
      image.src=url;
    });
  }
  async function decodePhoto(file){
    if(window.createImageBitmap){
      try{return await createImageBitmap(file,{imageOrientation:'from-image'});}catch(e){console.warn('createImageBitmap fallback',e);}
    }
    return decodeWithImage(file);
  }
  async function chooseFile(file){
    if(!file)return;
    try {
      setStatus(ui.model,'Открываю фотографию…');
      if(sourceBitmap&&sourceBitmap.close)sourceBitmap.close();
      sourceBitmap=await decodePhoto(file);
      sourceImage=makeSourceCanvas(sourceBitmap);
      ui.start.classList.add('hidden'); ui.editor.classList.remove('hidden');
      await recognize();
    } catch(e){ console.error(e); setStatus(ui.model,'Не удалось открыть фото: '+e.message,'error'); }
  }
  [ui.camera,ui.gallery].forEach(input=>input.addEventListener('change',()=>{chooseFile(input.files&&input.files[0]);input.value='';}));

  function makeSourceCanvas(bitmap){
    const max=1800, scale=Math.min(1,max/Math.max(bitmap.width,bitmap.height));
    const c=document.createElement('canvas'); c.width=Math.max(1,Math.round(bitmap.width*scale)); c.height=Math.max(1,Math.round(bitmap.height*scale));
    c.getContext('2d').drawImage(bitmap,0,0,c.width,c.height); return c;
  }
  function inputTensor(){
    const w=sourceImage.width,h=sourceImage.height,side=Math.max(w,h),scale=256/side,dw=w*scale,dh=h*scale,ox=(256-dw)/2,oy=(256-dh)/2;
    const c=document.createElement('canvas');c.width=c.height=256;const x=c.getContext('2d',{willReadFrequently:true});x.fillStyle='#000';x.fillRect(0,0,256,256);x.drawImage(sourceImage,ox,oy,dw,dh);
    const p=x.getImageData(0,0,256,256).data,a=new Float32Array(3*256*256),n=256*256;
    for(let i=0;i<n;i++){a[i]=p[4*i]/255;a[n+i]=p[4*i+1]/255;a[2*n+i]=p[4*i+2]/255;}
    geometry={w,h,ox,oy,dw,dh}; return new ort.Tensor('float32',a,[1,3,256,256]);
  }
  async function recognize(){
    if(!session){ setStatus(ui.status,'Модель ещё загружается. Подождите несколько секунд.','error'); await initModel(); if(!session)return; }
    ui.busy.classList.remove('hidden'); setStatus(ui.status,'');
    try {
      await new Promise(requestAnimationFrame); const started=performance.now();
      const result=await session.run({[session.inputNames[0]]:inputTensor()}),logits=result[session.outputNames[0]].data;
      probabilities=new Float32Array(logits.length); for(let i=0;i<logits.length;i++)probabilities[i]=1/(1+Math.exp(-logits[i]));
      render(); setStatus(ui.status,'Готово за '+Math.round(performance.now()-started)+' мс','ok');
    } catch(e){console.error(e);setStatus(ui.status,'Ошибка распознавания: '+e.message,'error');}
    finally{ui.busy.classList.add('hidden');}
  }
  function maskCanvas(){
    const t=+ui.threshold.value,net=document.createElement('canvas');net.width=net.height=256;const x=net.getContext('2d'),im=x.createImageData(256,256);
    for(let i=0;i<probabilities.length;i++){const v=probabilities[i]>t?255:0;im.data[4*i]=im.data[4*i+1]=im.data[4*i+2]=v;im.data[4*i+3]=255;}x.putImageData(im,0,0);
    const m=document.createElement('canvas');m.width=geometry.w;m.height=geometry.h;m.getContext('2d').drawImage(net,geometry.ox,geometry.oy,geometry.dw,geometry.dh,0,0,m.width,m.height);return m;
  }
  function render(){
    if(!sourceImage)return; const w=sourceImage.width,h=sourceImage.height;ui.canvas.width=w;ui.canvas.height=h;const out=ui.canvas.getContext('2d');out.drawImage(sourceImage,0,0);
    if(showingOriginal||!probabilities)return; const mask=maskCanvas(),m=mask.getContext('2d').getImageData(0,0,w,h).data,src=sourceImage.getContext('2d').getImageData(0,0,w,h),dst=out.createImageData(w,h),hex=ui.color.value,r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5),16),targetLum=.299*r+.587*g+.114*b,alpha=+ui.opacity.value/100,debug=ui.showMask.checked;
    dst.data.set(src.data); for(let i=0;i<w*h;i++){const a=(m[4*i]/255)*alpha;if(a<.01)continue;const q=4*i;if(debug){dst.data[q]=255;dst.data[q+1]=45;dst.data[q+2]=130;continue;}const lum=.299*src.data[q]+.587*src.data[q+1]+.114*src.data[q+2],k=Math.max(.38,Math.min(1.65,lum/(targetLum||1)));dst.data[q]=src.data[q]*(1-a)+Math.min(255,r*k)*a;dst.data[q+1]=src.data[q+1]*(1-a)+Math.min(255,g*k)*a;dst.data[q+2]=src.data[q+2]*(1-a)+Math.min(255,b*k)*a;}
    out.putImageData(dst,0,0);
  }
  function resultDataUrl(){showingOriginal=false;render();return ui.canvas.toDataURL('image/jpeg',.94);}
  async function saveOrShare(share){
    if(!probabilities)return; const dataUrl=resultDataUrl(),base64=dataUrl.split(',')[1],name='MANI-Magic-'+Date.now()+'.jpg';
    try{
      const cap=window.Capacitor&&window.Capacitor.Plugins,fs=cap&&cap.Filesystem,sh=cap&&cap.Share,media=cap&&cap.TryOnMedia;
      if(!share&&media){await media.saveImage({data:base64,name});toast('Сохранено в «Фото» → MANI Magic');return;}
      if(fs){
        const saved=await fs.writeFile({path:name,data:base64,directory:'CACHE'});
        if(sh){await sh.share({title:'Мой маникюр MANI Magic',text:'Примерка цвета в MANI Magic',url:saved.uri,dialogTitle:'Поделиться результатом'});return;}
      }
      const a=document.createElement('a');a.href=dataUrl;a.download=name;document.body.appendChild(a);a.click();a.remove();toast('Результат сохранён');
    }catch(e){if(String(e).toLowerCase().includes('cancel'))return;console.error(e);setStatus(ui.status,'Не удалось сохранить: '+e.message,'error');}
  }
  ui.color.oninput=()=>selectedColor(ui.color.value);ui.opacity.oninput=()=>{ui.opacityValue.textContent=ui.opacity.value+'%';render();};ui.threshold.oninput=()=>{ui.thresholdValue.textContent=(+ui.threshold.value).toFixed(2);render();};ui.showMask.onchange=render;
  $('recognizeBtn').onclick=recognize;$('newPhotoBtn').onclick=()=>ui.gallery.click();$('saveBtn').onclick=()=>saveOrShare(false);$('shareBtn').onclick=()=>saveOrShare(true);
  const originalOn=()=>{showingOriginal=true;render();},originalOff=()=>{showingOriginal=false;render();};['pointerdown','touchstart'].forEach(e=>ui.compare.addEventListener(e,originalOn,{passive:true}));['pointerup','pointercancel','pointerleave','touchend'].forEach(e=>ui.compare.addEventListener(e,originalOff,{passive:true}));
  initModel();
})();