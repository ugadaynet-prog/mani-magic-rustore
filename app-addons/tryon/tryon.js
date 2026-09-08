'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const ui = { start:$('startCard'), editor:$('editor'), camera:$('cameraInput'), gallery:$('galleryInput'), model:$('modelStatus'), canvas:$('resultCanvas'), busy:$('busy'), color:$('colorInput'), code:$('colorCode'), opacity:$('opacity'), opacityValue:$('opacityValue'), status:$('editorStatus'), toast:$('toast'), compare:$('compareBtn'), palette:$('palette'), newPhoto:$('newPhotoBtn'), share:$('shareBtn'), save:$('saveBtn') };
  const colors = ['#F5D0C5','#D98A91','#F04479','#D81B60','#A81748','#8B2F67','#7446B8','#335CC7','#1597A5','#3BAA70','#D6A522','#17171B'];
  let sourceBitmap, sourceImage, probabilities, geometry, showingOriginal = false;
  // Порог отсечки маски. Был ползунком в разделе «Настроить распознавание»,
  // но клиенту нечего с ним делать: «порог маски» ничего ему не говорит, а
  // сценарий должен быть «сфотографировал — примерил». Значение выбрано
  // замером по ручному эталону (13 отложенных кадров): при 0.40 находится 75
  // ногтей из 81 против 74 при 0.50, лишних пятен столько же.
  const THRESHOLD = 0.40;
  // Ширина мягкой полосы вокруг порога. Внутри неё краска нарастает плавно,
  // а не включается щелчком: край лака на фотографии тоже не бывает резким,
  // и жёсткая граница на увеличенной маске читается ступеньками.
  const SOFT = 0.10;
  // Дырка внутри ногтя площадью до этой доли всей маски закрашивается.
  // Блёстки, стразы и белый рисунок модель нередко считает не-ногтем, и
  // посреди покрашенной пластины остаётся непрокрашенное пятно. Настоящих
  // сквозных отверстий в ногте не бывает, так что закрывать их безопасно.
  const HOLE_MAX_FRAC = 0.04;
  // Сторона маски берётся из самой маски, а не задаётся константой: размер
  // входа модели менялся (384 → 512), и зашитое здесь число разъезжалось бы
  // с плагином молча — маска легла бы на фото со сдвигом и масштабом.
  let maskSide = 512;

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
  function computeGeometry(w,h,side){
    const long=Math.max(w,h), scale=side/long;
    return { w, h, dw:w*scale, dh:h*scale, ox:(side-w*scale)/2, oy:(side-h*scale)/2 };
  }

  async function recognize(){
    ui.busy.classList.remove('hidden');
    try {
      setStatus(ui.status,'Распознаю ногти…');
      await new Promise(r=>requestAnimationFrame(r));

      const started=performance.now();
      // Готовим JPEG для нативного плагина.
      const { dataUrl, w, h } = toJpegDataUrl(sourceImage);

      // Вызов нативного плагина: передаём JPEG dataURL, получаем PNG-маску.
      const result = await NailSeg.segment({ image: dataUrl });
      const maskDataUrl = result.mask;

      // Сначала маска — из неё известна сторона, и только потом геометрия.
      probabilities = await decodeMaskToProbabilities(maskDataUrl);
      geometry = computeGeometry(w, h, maskSide);
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

  // Загружает PNG-маску (grayscale) и возвращает массив вероятностей 0..1.
  function decodeMaskToProbabilities(maskDataUrl){
    return new Promise((resolve,reject)=>{
      const img=new Image();
      img.onload=()=>{
        maskSide = img.naturalWidth || maskSide;
        const c=document.createElement('canvas');
        c.width=c.height=maskSide;
        const x=c.getContext('2d',{willReadFrequently:true});
        x.drawImage(img,0,0,maskSide,maskSide);
        const p=x.getImageData(0,0,maskSide,maskSide).data;
        const probs=new Float32Array(maskSide*maskSide);
        for(let i=0;i<probs.length;i++) probs[i]=p[4*i]/255;
        resolve(probs);
      };
      img.onerror=()=>reject(new Error('не удалось декодировать маску'));
      img.src=maskDataUrl;
    });
  }

  // Закрашивает дырки внутри ногтей. Фон заливается от краёв кадра; всё
  // фоновое, куда заливка не дошла, — это отверстие внутри маски. Мелкие
  // закрываем, крупные (промежутки между пальцами) оставляем.
  function fillHoles(bin, side){
    const n=side*side, seen=new Uint8Array(n), stack=new Int32Array(n);
    let sp=0;
    const push=(i)=>{ if(!seen[i] && !bin[i]){ seen[i]=1; stack[sp++]=i; } };
    for(let x=0;x<side;x++){ push(x); push((side-1)*side+x); }
    for(let y=0;y<side;y++){ push(y*side); push(y*side+side-1); }
    while(sp>0){
      const i=stack[--sp], x=i%side, y=(i-x)/side;
      if(x>0) push(i-1); if(x<side-1) push(i+1);
      if(y>0) push(i-side); if(y<side-1) push(i+side);
    }
    let area=0; for(let i=0;i<n;i++) if(bin[i]) area++;
    const limit=Math.max(40, area*HOLE_MAX_FRAC);
    const done=new Uint8Array(n);
    for(let i0=0;i0<n;i0++){
      if(bin[i0]||seen[i0]||done[i0]) continue;
      // Отдельная дырка: собираем её целиком, чтобы знать размер.
      sp=0; stack[sp++]=i0; done[i0]=1;
      const cells=[];
      while(sp>0){
        const i=stack[--sp]; cells.push(i);
        const x=i%side, y=(i-x)/side;
        const nb=[x>0?i-1:-1, x<side-1?i+1:-1, y>0?i-side:-1, y<side-1?i+side:-1];
        for(const j of nb) if(j>=0 && !bin[j] && !seen[j] && !done[j]){ done[j]=1; stack[sp++]=j; }
      }
      if(cells.length<=limit) for(const i of cells) bin[i]=1;
    }
    return bin;
  }

  function maskCanvas(){
    const side=maskSide, n=side*side;
    // Сначала бинаризуем — только чтобы найти дырки.
    const bin=new Uint8Array(n);
    for(let i=0;i<n;i++) bin[i]=probabilities[i]>THRESHOLD?1:0;
    fillHoles(bin, side);

    // В маску кладём саму вероятность, а не ноль-или-255. Ступеньки на краю
    // брались именно из ранней бинаризации: диагональ ногтя на 512 точках
    // превращалась в лесенку, и растягивание её только увеличивало. Плавное
    // поле растягивается плавно, а порог применяется уже в размере фотографии.
    const net=document.createElement('canvas');net.width=net.height=side;
    const x=net.getContext('2d'),im=x.createImageData(side,side);
    for(let i=0;i<n;i++){
      const v=Math.round(255*Math.max(probabilities[i], bin[i]?1:0));
      im.data[4*i]=im.data[4*i+1]=im.data[4*i+2]=v;im.data[4*i+3]=255;
    }
    x.putImageData(im,0,0);
    const m=document.createElement('canvas');m.width=geometry.w;m.height=geometry.h;
    const mx=m.getContext('2d'); mx.imageSmoothingEnabled=true; mx.imageSmoothingQuality='high';
    mx.drawImage(net,geometry.ox,geometry.oy,geometry.dw,geometry.dh,0,0,m.width,m.height);
    return m;
  }
  function render(){
    if(!sourceImage)return; const w=sourceImage.width,h=sourceImage.height;ui.canvas.width=w;ui.canvas.height=h;const out=ui.canvas.getContext('2d');out.drawImage(sourceImage,0,0);
    if(showingOriginal||!probabilities)return; const mask=maskCanvas(),m=mask.getContext('2d').getImageData(0,0,w,h).data,src=sourceImage.getContext('2d').getImageData(0,0,w,h),dst=out.createImageData(w,h),hex=ui.color.value,r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5),16),targetLum=.299*r+.587*g+.114*b,alpha=+ui.opacity.value/100,debug=false;
    // Средняя яркость САМИХ ногтей. Раньше блик и тень считались от яркости
    // выбранной краски: на светлом ногте отношение упиралось в потолок 1.65,
    // и малиновый #A81748 выходил ярко-розовым #FF3988 — человек нажимал
    // «малиновый», а ноготь не менялся. От средней по ногтям краска в среднем
    // получается ровно та, что выбрана, а блик и тень остаются на месте.
    let sum=0,cnt=0;
    for(let i=0;i<w*h;i++){if(m[4*i]/255<=THRESHOLD)continue;const q=4*i;sum+=.299*src.data[q]+.587*src.data[q+1]+.114*src.data[q+2];cnt++;}
    const meanLum=cnt?sum/cnt:targetLum||128;
    dst.data.set(src.data); for(let i=0;i<w*h;i++){const p=m[4*i]/255,a=Math.min(1,Math.max(0,(p-(THRESHOLD-SOFT))/(2*SOFT)))*alpha;if(a<.01)continue;const q=4*i;if(debug){dst.data[q]=255;dst.data[q+1]=45;dst.data[q+2]=130;continue;}const lum=.299*src.data[q]+.587*src.data[q+1]+.114*src.data[q+2],k=Math.max(.55,Math.min(1.45,lum/(meanLum||1)));dst.data[q]=src.data[q]*(1-a)+Math.min(255,r*k)*a;dst.data[q+1]=src.data[q+1]*(1-a)+Math.min(255,g*k)*a;dst.data[q+2]=src.data[q+2]*(1-a)+Math.min(255,b*k)*a;}
    out.putImageData(dst,0,0);
  }
  function resultDataUrl(){showingOriginal=false;render();return ui.canvas.toDataURL('image/jpeg',.92);}

  ui.color.addEventListener('input',()=>{ui.code.textContent=ui.color.value.toUpperCase();render();});
  ui.opacity.addEventListener('input',()=>{ui.opacityValue.textContent=ui.opacity.value+'%';render();});
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
