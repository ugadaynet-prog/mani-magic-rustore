'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const ui = { start:$('startCard'), editor:$('editor'), camera:$('cameraInput'), gallery:$('galleryInput'), model:$('modelStatus'), canvas:$('resultCanvas'), busy:$('busy'), color:$('colorInput'), code:$('colorCode'), opacity:$('opacity'), opacityValue:$('opacityValue'), threshold:$('threshold'), thresholdValue:$('thresholdValue'), showMask:$('showMask'), status:$('editorStatus'), toast:$('toast'), compare:$('compareBtn'), palette:$('palette') };
  const colors = ['#F5D0C5','#D98A91','#F04479','#D81B60','#A81748','#8B2F67','#7446B8','#335CC7','#1597A5','#3BAA70','#D6A522','#17171B'];
  let sourceBitmap, sourceImage, probabilities, geometry, showingOriginal = false;
  const RUN_TIMEOUT_MS = 30000;

  // Нативный плагин распознавания ногтей (Kotlin + onnxruntime-android).
  // WebView больше не запускает ONNX Runtime Web — никакого WASM и .mjs.
  // Пока плагина нет (Шаг 1), экран честно сообщает об этом и не падает.
  const nailPlugin = (window.Capacitor && window.Capacitor.Plugins && window.Capacitor.Plugins.NailSegmentation) || null;

  function setStatus(el, text, kind=''){ el.textContent=text; el.className='status '+kind; }
  function withTimeout(promise, ms, message){
    let timer;
    return Promise.race([promise, new Promise((_, reject) => { timer=setTimeout(() => reject(new Error(message)), ms); })]).finally(() => clearTimeout(timer));
  }
  function toast(text){ ui.toast.textContent=text; ui.toast.classList.remove('hidden'); clearTimeout(toast.timer); toast.timer=setTimeout(()=>ui.toast.classList.add('hidden'),2400); }
  function selectedColor(hex){ ui.color.value=hex; ui.code.textContent=hex.toUpperCase(); document.querySelectorAll('.swatch').forEach(x=>x.classList.toggle('active',x.dataset.color.toLowerCase()===hex.toLowerCase())); render(); }
  colors.forEach((color,i)=>{ const b=document.createElement('button'); b.type='button'; b.className='swatch'+(i===3?' active':''); b.style.background=color; b.dataset.color=color; b.setAttribute('aria-label','Цвет '+color); b.onclick=()=>selectedColor(color); ui.palette.appendChild(b); });

  // Раньше здесь была загрузка ort.InferenceSession.create('./nail-unet.onnx').
  // Теперь модель загружает нативный плагин один раз внутри себя.
  // На старте экрана достаточно проверить, что плагин вообще доступен.
  function checkPlugin(){
    if(nailPlugin) { setStatus(ui.model,'Распознавание готово','ok'); return true; }
    setStatus(ui.model,'Нативное распознавание недоступно в этой сборке. Обновите приложение.','error');
    return false;
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

  // Готовит квадрат 256×256 с вписанным фото и чёрными полями —
  // тот же preprocessing, что раньше уходил в ort.Tensor.
  // Теперь canvas уходит в нативный плагин как JPEG dataURL.
  function prepareInferenceCanvas(){
    const w=sourceImage.width,h=sourceImage.height,side=Math.max(w,h),scale=256/side,dw=w*scale,dh=h*scale,ox=(256-dw)/2,oy=(256-dh)/2;
    const c=document.createElement('canvas');c.width=c.height=256;const x=c.getContext('2d');x.fillStyle='#000';x.fillRect(0,0,256,256);x.drawImage(sourceImage,ox,oy,dw,dh);
    geometry={w,h,ox,oy,dw,dh};
    return c;
  }

  // Расшифровывает маску 256×256, PNG в base64, в массив вероятностей 0..1.
  // Плагин отдаёт grayscale-картинку: значение пикселя / 255 = вероятность.
  function decodeMaskPng(dataUrl){
    return new Promise((resolve,reject)=>{
      const img=new Image();
      img.onload=()=>{
        const c=document.createElement('canvas');c.width=img.width;c.height=img.height;
        const x=c.getContext('2d',{willReadFrequently:true});x.drawImage(img,0,0);
        const p=x.getImageData(0,0,img.width,img.height).data;
        const probs=new Float32Array(img.width*img.height);
        for(let i=0;i<probs.length;i++) probs[i]=p[4*i]/255;
        resolve({probs,width:img.width,height:img.height});
      };
      img.onerror=()=>reject(new Error('не удалось прочитать маску от нативного плагина'));
      img.src=dataUrl;
    });
  }

  async function recognize(){
    ui.busy.classList.remove('hidden');
    try {
      if(!checkPlugin()) throw new Error('нативный плагин NailSegmentation не зарегистрирован');
      await new Promise(requestAnimationFrame);
      const started=performance.now();
      const inferenceCanvas=prepareInferenceCanvas();
      const response=await withTimeout(
        nailPlugin.segment({ image: inferenceCanvas.toDataURL('image/jpeg',0.9) }),
        RUN_TIMEOUT_MS,
        'распознавание не ответило за 30 секунд'
      );
      const decoded=await decodeMaskPng(response.mask);
      probabilities=decoded.probs;
      render();
      setStatus(ui.status,'Готово за '+(response.elapsedMs||Math.round(performance.now()-started))+' мс','ok');
    } catch(e){
      console.error(e);
      setStatus(ui.status,'Не удалось распознать ногти: '+e.message+'. Попробуйте другое фото или перезапустите экран.','error');
    } finally{ui.busy.classList.add('hidden');}
  }
  function maskCanvas(){
    const t=+ui.threshold.value,net=document.createElement('canvas');net.width=net.height=256;const x=net.getContext('2d'),im=x.createImageData(256,256);
    for(let i=0;i<probabilities.length;i++){const v=probabilities[i]>t?255:0;im.data[4*i]=im.data[4*i+1]=im.data[4*i+2]=v;im.data[4*i+3]=255;}x.putImageData(im,0,0);
    const m=document.createElement('canvas');m.width=geometry.w;m.height=geometry.h;m.getContext('2d').drawImage(net,geometry.ox,geometry.oy,geometry.dw,geometry.dh,0,0,m.width,m.height);return m;
  }
  function render(){
    if(!sourceImage)return; const w=sourceImage.width,h=sourceImage.height;ui.canvas.width=w;ui.canvas.height=h;const out=ui.canvas.getContext('2d');out.drawImage(sourceImage,0,0);
    if(showingOriginal||!probabilities)return; const mask=maskCanvas(),m=mask.getContext('2d').getImageData(0,0,w,h).data,src=sourceImage.getContext('2d').getImageData(0,0,w,h),dst=out.createImageData(w,h),hex=ui.color.value,r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5),16),targetLum=.299*r+.587*g+.114*b,alpha=+ui.opacity.value/100,debug=ui.showMask.checked;
    dst.data.set(src.data); for(let i=0;i<w*h;i++){const a=m[4*i]/255*alpha; if(a<=0)continue; const sR=src.data[4*i],sG=src.data[4*i+1],sB=src.data[4*i+2],sLum=.299*sR+.587*sG+.114*sB;let cR=r,cG=g,cB=b; if(!debug){const scale=sLum/255*1.4+0.4;cR*=scale;cG*=scale;cB*=scale;} dst.data[4*i]=cR+(sR-cR)*a;dst.data[4*i+1]=cG+(sG-cG)*a;dst.data[4*i+2]=cB+(sB-cB)*a;dst.data[4*i+3]=255;} out.putImageData(dst,0,0);
  }
  async function saveOrShare(share){
    if(!sourceImage)return;
    try{
      const dataUrl=ui.canvas.toDataURL('image/jpeg',0.92),base64=dataUrl.split(',')[1],name='MANI-Magic-'+Date.now()+'.jpg';
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
  checkPlugin();
})();
