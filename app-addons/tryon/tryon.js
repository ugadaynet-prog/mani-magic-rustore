'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const ui = { start:$('startCard'), editor:$('editor'), camera:$('cameraInput'), gallery:$('galleryInput'), model:$('modelStatus'), canvas:$('resultCanvas'), busy:$('busy'), color:$('colorInput'), code:$('colorCode'), opacity:$('opacity'), opacityValue:$('opacityValue'), status:$('editorStatus'), toast:$('toast'), compare:$('compareBtn'), palette:$('palette'), newPhoto:$('newPhotoBtn'), share:$('shareBtn'), save:$('saveBtn') };
  const colors = ['#F5D0C5','#D98A91','#F04479','#D81B60','#A81748','#8B2F67','#7446B8','#335CC7','#1597A5','#3BAA70','#D6A522','#17171B'];
  ui.wrap = $('canvasWrap');
  let sourceBitmap, sourceImage, probabilities, geometry, showingOriginal = false;
  // Что стёрли последним касанием — чтобы промах можно было отменить.
  let lastErased = null;
  // Увеличение результата: масштаб и сдвиг в пикселях обёртки.
  let view = { scale: 1, tx: 0, ty: 0 };
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
  // Кадр темнее этого или мягче этого — предупреждаем, что примерка будет
  // неточной. Числа не выдуманы: посчитаны по 109 кадрам трёх экзаменов.
  // Яркость: медиана 0.49, пятый процентиль 0.25, минимум 0.12. Резкость
  // (средний квадрат лапласиана на стороне 96): медиана 0.011, пятый
  // процентиль 0.0032. Пороги поставлены ниже пятого процентиля, чтобы
  // подсказка появлялась редко и по делу.
  const DARK_MEAN = 0.16;
  const SOFT_LAP = 0.0025;
  // Сторона маски берётся из самой маски, а не задаётся константой: размер
  // входа модели менялся (384 → 512), и зашитое здесь число разъезжалось бы
  // с плагином молча — маска легла бы на фото со сдвигом и масштабом.
  let maskSide = 512;

  // Нативный плагин NailSegmentation (Kotlin + onnxruntime-android).
  // В WebView недоступен, поэтому получаем прокси через Capacitor.
  const NailSeg = window.Capacitor && window.Capacitor.Plugins
    ? window.Capacitor.Plugins.NailSegmentation
    : null;

  function toast(text, action){
    ui.toast.textContent = text;
    ui.toast.classList.remove('hidden');
    ui.toast.classList.toggle('tappable', !!action);
    ui.toast.onclick = action ? () => { ui.toast.classList.add('hidden'); action(); } : null;
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => ui.toast.classList.add('hidden'), action ? 4200 : 2400);
  }
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
  // ===== Увеличение и стирание лишнего =====
  //
  // Одно касание убирает пятно, двойное — приближает, щипок — свободный
  // масштаб. Там, где модель ошибается, она ошибается уверенно: замер 9
  // сентября показал вероятность 0.001 внутри пропущенного ногтя и 0.998
  // внутри найденного. Значит порогом лишнее не убрать — а пальцем убирается
  // за секунду.

  function applyView(){
    ui.canvas.style.transformOrigin = 'center center';
    ui.canvas.style.transform = 'translate(' + view.tx + 'px,' + view.ty + 'px) scale(' + view.scale + ')';
    ui.wrap.classList.toggle('zoomed', view.scale > 1.01);
  }
  function clampView(){
    const r = ui.wrap.getBoundingClientRect();
    const mx = Math.max(0, (r.width * view.scale - r.width) / 2);
    const my = Math.max(0, (r.height * view.scale - r.height) / 2);
    view.tx = Math.min(mx, Math.max(-mx, view.tx));
    view.ty = Math.min(my, Math.max(-my, view.ty));
  }
  function resetView(){ view = { scale:1, tx:0, ty:0 }; applyView(); }
  // Приблизить так, чтобы точка под пальцем осталась на месте.
  function zoomAt(px, py, next){
    const s0 = view.scale, s1 = Math.min(4, Math.max(1, next));
    const cx = (px - view.tx) / s0, cy = (py - view.ty) / s0;
    view.scale = s1; view.tx = px - cx * s1; view.ty = py - cy * s1;
    clampView(); applyView();
  }

  // Убрать связное пятно маски под пальцем. Координаты берутся от canvas, а он
  // уже учитывает css-трансформацию, поэтому увеличение ничего не ломает.
  function eraseAt(clientX, clientY){
    if(!probabilities || !geometry) return false;
    const rect = ui.canvas.getBoundingClientRect();
    const fx = (clientX - rect.left) / rect.width * geometry.w;
    const fy = (clientY - rect.top) / rect.height * geometry.h;
    if(fx < 0 || fy < 0 || fx >= geometry.w || fy >= geometry.h) return false;
    const side = maskSide;
    const mx = Math.round(geometry.ox + fx / geometry.w * geometry.dw);
    const my = Math.round(geometry.oy + fy / geometry.h * geometry.dh);
    if(mx < 0 || my < 0 || mx >= side || my >= side) return false;
    const start = my * side + mx;
    if(probabilities[start] <= THRESHOLD) return false;
    const stack = new Int32Array(side * side);
    const seen = new Uint8Array(side * side);
    const cells = [];
    let sp = 0; stack[sp++] = start; seen[start] = 1;
    while(sp > 0){
      const i = stack[--sp]; cells.push(i);
      const x = i % side, y = (i - x) / side;
      const nb = [x > 0 ? i-1 : -1, x < side-1 ? i+1 : -1, y > 0 ? i-side : -1, y < side-1 ? i+side : -1];
      for(const j of nb) if(j >= 0 && !seen[j] && probabilities[j] > THRESHOLD){ seen[j] = 1; stack[sp++] = j; }
    }
    const idx = new Int32Array(cells), val = new Float32Array(cells.length);
    for(let k = 0; k < cells.length; k++){ val[k] = probabilities[cells[k]]; probabilities[cells[k]] = 0; }
    lastErased = { idx, val };
    render();
    return true;
  }
  function undoErase(){
    if(!lastErased) return;
    for(let k = 0; k < lastErased.idx.length; k++) probabilities[lastErased.idx[k]] = lastErased.val[k];
    lastErased = null; render(); toast('Вернул');
  }

  // Жесты. Одиночное касание отделяем от двойного по времени, поэтому стирание
  // срабатывает с задержкой в четверть секунды: иначе двойной тап успевал бы
  // сначала стереть ноготь, а потом приблизить дырку на его месте.
  (function gestures(){
    const pts = new Map();
    let pinch = null, panning = null, tapTimer = null, lastTap = 0;
    const wrapPoint = (cx, cy) => {
      const r = ui.wrap.getBoundingClientRect();
      return { x: cx - r.left - r.width / 2, y: cy - r.top - r.height / 2 };
    };
    ui.wrap.addEventListener('pointerdown', e => {
      if(!probabilities) return;
      ui.wrap.setPointerCapture(e.pointerId);
      pts.set(e.pointerId, { x: e.clientX, y: e.clientY, t: Date.now(), x0: e.clientX, y0: e.clientY });
      if(pts.size === 2){
        const two = [...pts.values()];
        pinch = { d: Math.hypot(two[0].x - two[1].x, two[0].y - two[1].y), s: view.scale };
        clearTimeout(tapTimer);
      } else if(view.scale > 1.01){
        panning = { tx: view.tx, ty: view.ty, x: e.clientX, y: e.clientY };
      }
    });
    ui.wrap.addEventListener('pointermove', e => {
      const p = pts.get(e.pointerId); if(!p) return;
      p.x = e.clientX; p.y = e.clientY;
      if(pinch && pts.size === 2){
        const two = [...pts.values()];
        const d = Math.hypot(two[0].x - two[1].x, two[0].y - two[1].y);
        const w = wrapPoint((two[0].x + two[1].x) / 2, (two[0].y + two[1].y) / 2);
        zoomAt(w.x, w.y, pinch.s * (d / pinch.d));
        e.preventDefault();
      } else if(panning){
        view.tx = panning.tx + (e.clientX - panning.x);
        view.ty = panning.ty + (e.clientY - panning.y);
        clampView(); applyView();
        e.preventDefault();
      }
    });
    ui.wrap.addEventListener('pointerup', e => {
      const p = pts.get(e.pointerId);
      pts.delete(e.pointerId);
      if(pts.size < 2) pinch = null;
      if(pts.size === 0) panning = null;
      if(!p) return;
      const moved = Math.hypot(e.clientX - p.x0, e.clientY - p.y0);
      if(!(moved < 10 && Date.now() - p.t < 400)) return;
      const now = Date.now(), w = wrapPoint(e.clientX, e.clientY);
      if(now - lastTap < 300){
        clearTimeout(tapTimer); lastTap = 0;
        const zoomed = view.scale > 1.01;
        if(zoomed) resetView(); else zoomAt(w.x, w.y, 2.5);
        return;
      }
      lastTap = now;
      const cx = e.clientX, cy = e.clientY;
      tapTimer = setTimeout(() => {
        if(eraseAt(cx, cy)) toast('Пятно убрано · нажмите, чтобы вернуть', undoErase);
      }, 260);
    });
    ui.wrap.addEventListener('pointercancel', e => { pts.delete(e.pointerId); pinch = null; panning = null; });
    window.addEventListener('resize', () => { clampView(); applyView(); });
  })();

  // Годится ли кадр вообще: темнота и смазанность видны до всякого
  // распознавания, и честнее сказать сразу, чем красить мимо.
  function photoQuality(){
    const s = 96, c = document.createElement('canvas'); c.width = c.height = s;
    const x = c.getContext('2d', { willReadFrequently:true });
    x.drawImage(sourceImage, 0, 0, s, s);
    const d = x.getImageData(0, 0, s, s).data, g = new Float32Array(s * s);
    let sum = 0;
    for(let i = 0; i < s * s; i++){ g[i] = (0.299*d[4*i] + 0.587*d[4*i+1] + 0.114*d[4*i+2]) / 255; sum += g[i]; }
    let lap = 0, n = 0;
    for(let y = 1; y < s - 1; y++) for(let xx = 1; xx < s - 1; xx++){
      const i = y * s + xx;
      const v = 4*g[i] - g[i-1] - g[i+1] - g[i-s] - g[i+s];
      lap += v * v; n++;
    }
    return { mean: sum / (s * s), sharp: lap / n };
  }

  // Сколько отдельных пятен нашла модель: по ним видно, есть ли вообще ногти.
  function blobCount(){
    if(!probabilities) return 0;
    const side = maskSide, n = side * side;
    const seen = new Uint8Array(n), stack = new Int32Array(n);
    let count = 0;
    for(let i0 = 0; i0 < n; i0++){
      if(seen[i0] || probabilities[i0] <= THRESHOLD) continue;
      let sp = 0, size = 0; stack[sp++] = i0; seen[i0] = 1;
      while(sp > 0){
        const i = stack[--sp]; size++;
        const x = i % side, y = (i - x) / side;
        const nb = [x > 0 ? i-1 : -1, x < side-1 ? i+1 : -1, y > 0 ? i-side : -1, y < side-1 ? i+side : -1];
        for(const j of nb) if(j >= 0 && !seen[j] && probabilities[j] > THRESHOLD){ seen[j] = 1; stack[sp++] = j; }
      }
      if(size >= 40) count++;
    }
    return count;
  }

  // Что сказать про кадр после распознавания.
  function advice(nails){
    const q = photoQuality();
    if(!nails) return ['Ногтей не видно. Снимите руку крупнее, чтобы ногти занимали заметную часть кадра', 'error'];
    if(q.mean < DARK_MEAN) return ['Кадр тёмный — примерка будет неточной. Лучше переснять при свете', ''];
    if(q.sharp < SOFT_LAP) return ['Кадр смазан — примерка будет неточной. Лучше переснять', ''];
    return null;
  }

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
      lastErased = null;
      resetView();
      render();
      const hint = advice(blobCount());
      if(hint) setStatus(ui.status, hint[0], hint[1]);
      else setStatus(ui.status, 'Готово. Двойное касание — приблизить, одно — убрать лишнее пятно', 'ok');
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

  // ===== Сохранить и поделиться =====
  //
  // В приложении оба действия делались средствами браузера, и оба выходили
  // мимо: «сохранить» клало файл в каталог загрузок WebView, где владелец его
  // не нашёл, а «поделиться» упиралось в то, что WebView не умеет отдавать
  // файл в navigator.share, и открывало картинку новой вкладкой. Поэтому
  // теперь: сохранение — через MediaStore нативным плагином, то есть в
  // галерею; отправка — через системное окно Capacitor, то есть сразу в
  // список мессенджеров. В обычном браузере остаются прежние пути.
  const Cap = window.Capacitor && window.Capacitor.Plugins ? window.Capacitor.Plugins : null;

  async function cacheFile(dataUrl, name){
    const res = await Cap.Filesystem.writeFile({
      path: name, data: dataUrl.split(',')[1], directory: 'CACHE' });
    return res.uri;
  }

  if(ui.save) ui.save.addEventListener('click', async ()=>{
    let dataUrl;
    try { dataUrl = resultDataUrl(); }
    catch(e){ toast('Не удалось сохранить'); console.error(e); return; }
    // Сначала галерея: в приложении для этого давно есть плагин TryOnMedia,
    // он кладёт снимок через MediaStore в альбом «MANI Magic», то есть туда,
    // где человек его и ищет — в «Фото». В браузере плагина нет, и остаётся
    // обычная ссылка на скачивание.
    const media = Cap && Cap.TryOnMedia;
    if(media){
      try {
        await media.saveImage({ data: dataUrl, name: 'MANI-Magic-' + Date.now() + '.jpg' });
        toast('Сохранено в галерею, альбом «MANI Magic»');
        return;
      } catch(e){ console.warn('галерея недоступна, сохраняю файлом:', e); }
    }
    try {
      const link = document.createElement('a');
      link.href = dataUrl; link.download = 'mani-magic-tryon.jpg';
      link.click();
      toast('Сохранено в загрузки');
    } catch(e){ toast('Не удалось сохранить'); console.error(e); }
  });

  if(ui.share) ui.share.addEventListener('click', async ()=>{
    try {
      const dataUrl = resultDataUrl();
      if(Cap && Cap.Share && Cap.Filesystem){
        const uri = await cacheFile(dataUrl, 'mani-magic-' + Date.now() + '.jpg');
        await Cap.Share.share({
          title: 'MANI Magic', text: 'Примерка маникюра',
          files: [uri], dialogTitle: 'Отправить примерку' });
        return;
      }
      const blob = await (await fetch(dataUrl)).blob();
      const file = new File([blob], 'mani-magic.jpg', { type:'image/jpeg' });
      if(navigator.canShare && navigator.canShare({ files:[file] })) {
        await navigator.share({ files:[file], title:'MANI Magic', text:'Примерка маникюра' });
      } else {
        const link = document.createElement('a');
        link.href = dataUrl; link.download = 'mani-magic-tryon.jpg';
        link.click();
        toast('Отправка недоступна — файл сохранён');
      }
    } catch(e){ if(e.name!=='AbortError'){ toast('Не удалось поделиться'); console.error(e); } }
  });

  document.addEventListener('DOMContentLoaded', checkNativePlugin);
  if(document.readyState!=='loading') checkNativePlugin();
})();
