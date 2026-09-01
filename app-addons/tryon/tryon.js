'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const ui = { start:$('startCard'), editor:$('editor'), camera:$('cameraInput'), gallery:$('galleryInput'), model:$('modelStatus'), canvas:$('resultCanvas'), busy:$('busy'), color:$('colorInput'), code:$('colorCode'), opacity:$('opacity'), opacityValue:$('opacityValue'), threshold:$('threshold'), thresholdValue:$('thresholdValue'), showMask:$('showMask'), status:$('editorStatus'), toast:$('toast'), compare:$('compareBtn'), palette:$('palette'), newPhoto:$('newPhotoBtn'), share:$('shareBtn'), save:$('saveBtn'), recognize:$('recognizeBtn') };
  const colors = ['#F5D0C5','#D98A91','#F04479','#D81B60','#A81748','#8B2F67','#7446B8','#335CC7','#1597A5','#3BAA70','#D6A522','#17171B'];
  let sourceBitmap, sourceImage, probabilities, geometry, showingOriginal = false;

  // Нативный плагин NailSegmentation (Kotlin + onnxruntime-android).
  // В WebView недоступен, поэтому получаем прокси через Capacitor.
  const NailSeg = window.Capacitor && window.Capacitor.Plugins
    ? window.Capacitor.Plugins.NailSegmentation
    : null;

  function toast(text){ ui.toast.textContent=text; ui.toast.classList.remove('hidden'); clearTimeout(toast.timer); toast.timer=setTimeout(()=>ui.toast.classList.add('hidden'),2400); }
  function setStatus(el, text, kind=''){ el.textContent=text; el.className='status '+kind; }
  function selectedColor(hex){ ui.color.value=hex; ui.code.textContent=hex.toUpperCase(); document.querySelectorAll('.swatch').forEach(x=>x.classList.toggle('active',x.dataset.color.toLowerCase()===hex.toLowerCase())); render(); }
  colors.forEach((color,i)=>{ const b=document.createElement('button'); b.type='button'; b.className='swatch'+(i===3?' active':''); b.style.background=color; b.dataset.color=color; b.setAttribute('aria-label','Цвет '+color); b.onclick=()=>selectedColor(color); ui.palette.appendChild(b); });

  // Проверяем доступность нативного плагина при загрузке экрана.
  function checkNativePlugin(){
    console.log('checkNativePlugin: window.Capacitor =', !!window.Capacitor);
    console.log('checkNativePlugin: window.Capacitor.Plugins =', window.Capacitor && window.Capacitor.Plugins);
    console.log('checkNativePlugin: NailSeg =', NailSeg);
    console.log('checkNativePlugin: NailSeg.segment =', NailSeg && typeof NailSeg.segment);
    
    if (!window.Capacitor) {
      setStatus(ui.model, 'DIAG: window.Capacitor отсутствует — Capacitor не инициализирован', 'error');
      return false;
    }
    if (!window.Capacitor.Plugins) {
      setStatus(ui.model, 'DIAG: window.Capacitor.Plugins отсутствует', 'error');
      return false;
    }
    if (!NailSeg) {
      const available = Object.keys(window.Capacitor.Plugins);
      setStatus(ui.model, `DIAG: NailSegmentation не найден. Доступные плагины: [${available.join(', ')}]`, 'error');
      return false;
    }
    if (typeof NailSeg.segment !== 'function') {
      setStatus(ui.model, `DIAG: NailSeg.segment не функция (typeof=${typeof NailSeg.segment})`, 'error');
      return false;
    }
    setStatus(ui.model, 'Плагин NailSegmentation готов ✓', 'ok');
    return true;
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

  // Конвертирует ImageBitmap/Image/Canvas в JPEG dataURL для передачи в нативный плагин.
  function toJpegDataUrl(bitmap){
    const max=1800, scale=Math.min(1,max/Math.max(bitmap.width,bitmap.height));
    const c=document.createElement('canvas');
    c.width=Math.max(1,Math.round(bitmap.width*scale));
    c.height=Math.max(1,Math.round(bitmap.height*scale));
    c.getContext('2d').drawImage(bitmap,0,0,c.width,c.height);
    return { dataUrl: c.toDataURL('image/jpeg', 0.9), w: c.width, h: c.height };
  }

  async function chooseFile(file){
    if(!file)return;
    if(!checkNativePlugin()){
      toast('Нативный плагин недоступен');
      return;
    }
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

  // Вычисляет геометрию letterbox (та же, что в нативном плагине).
  function computeGeometry(w,h){
    const side=Math.max(w,h), scale=384/side;
    return { w, h, dw:w*scale, dh:h*scale, ox:(384-w*scale)/2, oy:(384-h*scale)/2 };
  }

  async function recognize(){
    ui.busy.classList.remove('hidden');
    try {
      setStatus(ui.status,'Распознаю ногти…');
      await new Promise(r=>requestAnimationFrame(r));

      const started=performance.now();
      // Готовим JPEG для нативного плагина.
      const { dataUrl, w, h } = toJpegDataUrl(sourceImage);
      geometry = computeGeometry(w, h);

      // Вызов нативного плагина: передаём JPEG dataURL, получаем PNG-маску 384×384.
      const result = await NailSeg.segment({ image: dataUrl });
      const maskDataUrl = result.mask;

      // Декодируем PNG-маску в probabilities (Float32Array 384×384).
      probabilities = await decodeMaskToProbabilities(maskDataUrl);
      render();
      setStatus(ui.status, `Готово за ${result.elapsedMs || Math.round(performance.now()-started)} мс`, 'ok');
    } catch(e){
      console.error('recognize() error:', e);
      // Диагностический вывод: покажем тип ошибки, сообщение и stack
      const errType = e && e.constructor ? e.constructor.name : typeof e;
      const errMsg = e && e.message ? e.message : String(e);
      const stack = e && e.stack ? e.stack.split('\n').slice(0, 3).join(' | ') : '';
      setStatus(ui.status, `[${errType}] ${errMsg}${stack ? ' || '+stack : ''}`, 'error');
      // Также покажем состояние плагина
      const pluginState = NailSeg ? 'плагин есть' : 'плагин ОТСУТСТВУЕТ';
      setStatus(ui.model, `Диагностика: ${pluginState}. Ошибка: ${errType}: ${errMsg.substring(0, 120)}`, 'error');
    } finally { ui.busy.classList.add('hidden'); }
  }

  // Загружает PNG-маску (grayscale 384×384) и возвращает массив вероятностей 0..1.
  function decodeMaskToProbabilities(maskDataUrl){
    return new Promise((resolve,reject)=>{
      const img=new Image();
      img.onload=()=>{
        const c=document.createElement('canvas');
        c.width=c.height=384;
        const x=c.getContext('2d',{willReadFrequently:true});
        x.drawImage(img,0,0,384,384);
        const p=x.getImageData(0,0,384,384).data;
        const probs=new Float32Array(384*384);
        for(let i=0;i<probs.length;i++) probs[i]=p[4*i]/255;
        resolve(probs);
      };
      img.onerror=()=>reject(new Error('не удалось декодировать маску'));
      img.src=maskDataUrl;
    });
  }

  function maskCanvas(){
    const t=+ui.threshold.value,net=document.createElement('canvas');net.width=net.height=384;const x=net.getContext('2d'),im=x.createImageData(384,384);
    for(let i=0;i<probabilities.length;i++){const v=probabilities[i]>t?255:0;im.data[4*i]=im.data[4*i+1]=im.data[4*i+2]=v;im.data[4*i+3]=255;}x.putImageData(im,0,0);
    const m=document.createElement('canvas');m.width=geometry.w;m.height=geometry.h;m.getContext('2d').drawImage(net,geometry.ox,geometry.oy,geometry.dw,geometry.dh,0,0,m.width,m.height);return m;
  }
  function render(){
    if(!sourceImage)return; const w=sourceImage.width,h=sourceImage.height;ui.canvas.width=w;ui.canvas.height=h;const out=ui.canvas.getContext('2d');out.drawImage(sourceImage,0,0);
    if(showingOriginal||!probabilities)return; const mask=maskCanvas(),m=mask.getContext('2d').getImageData(0,0,w,h).data,src=sourceImage.getContext('2d').getImageData(0,0,w,h),dst=out.createImageData(w,h),hex=ui.color.value,r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5),16),targetLum=.299*r+.587*g+.114*b,alpha=+ui.opacity.value/100,debug=ui.showMask.checked;
    dst.data.set(src.data); for(let i=0;i<w*h;i++){const a=(m[4*i]/255)*alpha;if(a<.01)continue;const q=4*i;if(debug){dst.data[q]=255;dst.data[q+1]=45;dst.data[q+2]=130;continue;}const lum=.299*src.data[q]+.587*src.data[q+1]+.114*src.data[q+2],k=Math.max(.38,Math.min(1.65,lum/(targetLum||1)));dst.data[q]=src.data[q]*(1-a)+Math.min(255,r*k)*a;dst.data[q+1]=src.data[q+1]*(1-a)+Math.min(255,g*k)*a;dst.data[q+2]=src.data[q+2]*(1-a)+Math.min(255,b*k)*a;}
    out.putImageData(dst,0,0);
  }
  function resultDataUrl(){showingOriginal=false;render();return ui.canvas.toDataURL('image/jpeg',.92);}

  ui.color.addEventListener('input',()=>{ui.code.textContent=ui.color.value.toUpperCase();render();});
  ui.opacity.addEventListener('input',()=>{ui.opacityValue.textContent=ui.opacity.value+'%';render();});
  ui.threshold.addEventListener('input',()=>{ui.thresholdValue.textContent=Math.round(ui.threshold.value*100)+'%';render();});
  ui.showMask.addEventListener('change',render);
  ui.compare.addEventListener('mousedown',()=>{showingOriginal=true;render();});
  ui.compare.addEventListener('mouseup',()=>{showingOriginal=false;render();});
  ui.compare.addEventListener('mouseleave',()=>{showingOriginal=false;render();});
  ui.compare.addEventListener('touchstart',e=>{e.preventDefault();showingOriginal=true;render();},{passive:false});
  ui.compare.addEventListener('touchend',()=>{showingOriginal=false;render();});

  // ===== Кнопка «Другое фото» =====
  if(ui.newPhoto) ui.newPhoto.addEventListener('click',()=>{
    if(sourceBitmap&&sourceBitmap.close)sourceBitmap.close();
    sourceBitmap=sourceImage=null; probabilities=null;
    ui.editor.classList.add('hidden'); ui.start.classList.remove('hidden');
    setStatus(ui.status,'');
  });

  // ===== Кнопка «Распознать заново» =====
  if(ui.recognize) ui.recognize.addEventListener('click',()=>{ if(sourceImage) recognize(); });

  // ===== Кнопка «Сохранить результат» =====
  if(ui.save) ui.save.addEventListener('click',()=>{
    try {
      const dataUrl = resultDataUrl();
      const link = document.createElement('a');
      link.href = dataUrl; link.download = 'mani-magic-tryon.jpg';
      link.click();
      toast('Сохранено в загрузки');
    } catch(e){ toast('Не удалось сохранить'); console.error(e); }
  });

  // ===== Кнопка «Поделиться» =====
  if(ui.share) ui.share.addEventListener('click', async ()=>{
    try {
      const dataUrl = resultDataUrl();
      const blob = await (await fetch(dataUrl)).blob();
      const file = new File([blob], 'mani-magic.jpg', { type:'image/jpeg' });
      if(navigator.canShare && navigator.canShare({ files:[file] })) {
        await navigator.share({ files:[file], title:'MANI Magic', text:'Примерка маникюра' });
      } else if(navigator.share) {
        await navigator.share({ title:'MANI Magic', text:'Примерка маникюра', url:dataUrl });
      } else {
        const link = document.createElement('a');
        link.href = dataUrl; link.target='_blank'; link.click();
        toast('Открыто в новой вкладке');
      }
    } catch(e){ if(e.name!=='AbortError') toast('Не удалось поделиться'); console.error(e); }
  });

  document.addEventListener('DOMContentLoaded', checkNativePlugin);
  if(document.readyState!=='loading') checkNativePlugin();
})();
