'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const ui = { start:$('startCard'), editor:$('editor'), camera:$('cameraInput'), gallery:$('galleryInput'), model:$('modelStatus'), canvas:$('resultCanvas'), busy:$('busy'), color:$('colorInput'), code:$('colorCode'), opacity:$('opacity'), opacityValue:$('opacityValue'), status:$('editorStatus'), toast:$('toast'), compare:$('compareBtn'), palette:$('palette'), newPhoto:$('newPhotoBtn'), share:$('shareBtn'), save:$('saveBtn'), undo:$('undoBtn') };
  const colors = ['#F5D0C5','#D98A91','#F04479','#D81B60','#A81748','#8B2F67','#7446B8','#335CC7','#1597A5','#3BAA70','#D6A522','#17171B'];
  const colorNames = ['Нежно-розовый','Пыльная роза','Ярко-розовый','Малиновый','Вишнёвый','Сливовый','Фиолетовый','Синий','Бирюзовый','Зелёный','Золотистый','Графитовый'];
  let eraseMode = false;
  function compare(original) {
    showingOriginal = original;
    ui.compare.setAttribute('aria-pressed', String(original));
    $('afterBtn').setAttribute('aria-pressed', String(!original));
    $('afterBtn').classList.toggle('active', !original);
    render();
  }
  ui.wrap = $('canvasWrap');
  if (window.Capacitor?.isNativePlatform?.()) {
    document.documentElement.style.setProperty('--safe-top', '28px');
    const applyInsets = async (tries = 20) => {
      const plugin = window.Capacitor.Plugins?.Insets;
      if (!plugin?.get) { if (tries > 0) setTimeout(() => applyInsets(tries - 1), 150); return; }
      try {
        const result = await plugin.get();
        if (!result?.ready) { if (tries > 0) setTimeout(() => applyInsets(tries - 1), 150); return; }
        const num = value => Math.max(0, Math.round(Number(value) || 0));
        const bottom = num(result.bottom);
        const top = num(result.top) < 8 && bottom > 0 ? 28 : num(result.top);
        const style = document.documentElement.style;
        style.setProperty('--safe-top', top + 'px');
        style.setProperty('--safe-bottom', bottom + 'px');
        style.setProperty('--safe-left', num(result.left) + 'px');
        style.setProperty('--safe-right', num(result.right) + 'px');
      } catch {}
    };
    applyInsets();
    window.addEventListener('resize', () => applyInsets(3));
    document.addEventListener('visibilitychange', () => { if (!document.hidden) applyInsets(3); });
  }
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
  // Пятно меньше этой доли самого крупного ногтя — крапинка (блик на фоне,
  // край кольца), а не ноготь. Считаем от ногтя, а не от кадра: на снимке
  // всей кисти мизинец — доли процента кадра, но от большого пальца это треть.
  const SPECK_FRAC = 0.12;
  // Сначала у каждого пятна срезаются выступы уже 2·OPEN_R+1 точек маски:
  // так выглядит затёк краски с ногтя на кожу. Ноготь шире в разы и не страдает.
  const OPEN_R = 2;
  // Выпуклая оболочка выравнивает зазубренный край ногтя, но берём её, только
  // если она прибавляет не больше этой доли площади. Больше — в пятне два
  // соприкасающихся ногтя, и оболочка залила бы кожу между ними.
  const HULL_GAIN = 0.10;
  // Край сглаживается размытием (σ в точках маски), а порог потом берётся
  // уже в размере фото. EDGE_SLOPE — крутизна размытого края на точку маски
  // при этом σ: через неё сдвиг порога переводится в пиксели фото.
  const EDGE_BLUR = 1.0, EDGE_SLOPE = 0.38;
  // На сколько пикселей фото краска заходит за найденный край и ширина
  // растушёвки. Модель скорее недобирает край: без запаса по контуру
  // оставался ободок старого лака.
  const GROW_PX = 1, FEATHER_PX = 1.5;
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

  // Всплывающая подсказка только сообщает. Нажимать на неё пробовали —
  // на телефоне не работает: она висит у самого низа экрана, где касания
  // забирает системная полоса жестов. Действия живут в кнопках.
  function toast(text){
    ui.toast.textContent = text;
    ui.toast.classList.remove('hidden');
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => ui.toast.classList.add('hidden'), 2400);
  }
  function setStatus(el, text, kind=''){ el.textContent=text; el.className='status '+kind; }
  function selectedColor(hex, label){ ui.color.value=hex; ui.code.textContent=label || colorNames[colors.indexOf(hex.toUpperCase())] || 'Свой цвет'; ui.code.title=hex.toUpperCase(); document.querySelectorAll('.swatch').forEach(x=>{const active=x.dataset.color.toLowerCase()===hex.toLowerCase(); x.classList.toggle('active',active); x.setAttribute('aria-pressed',String(active));}); compare(false); }
  // Базовые цвета и палитры брендов-партнёров (../brands.js, данные в ../palettes.js).
  // Без brands.js остаются только базовые цвета, как раньше.
  if (window.ManiBrands) ManiBrands.mountTryon({ palette:ui.palette, base:colors.map((hex,i)=>({hex,label:colorNames[i]})), baseActive:3, onPick:selectedColor });
  else colors.forEach((color,i)=>{ const b=document.createElement('button'); b.type='button'; b.className='swatch'+(i===3?' active':''); b.style.background=color; b.dataset.color=color; b.setAttribute('aria-label',colorNames[i]); b.title=colorNames[i]; b.setAttribute('aria-pressed',String(i===3)); b.onclick=()=>selectedColor(color); ui.palette.appendChild(b); });

  // Проверяем доступность нативного плагина при загрузке экрана.
  function checkNativePlugin(){
    console.log('checkNativePlugin: window.Capacitor =', !!window.Capacitor);
    console.log('checkNativePlugin: window.Capacitor.Plugins =', window.Capacitor && window.Capacitor.Plugins);
    console.log('checkNativePlugin: NailSeg =', NailSeg);
    console.log('checkNativePlugin: NailSeg.segment =', NailSeg && typeof NailSeg.segment);
    
    if (!window.Capacitor) {
      setStatus(ui.model, 'Распознавание доступно в Android-приложении. В браузере можно просмотреть выбранное фото.', '');
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
    setStatus(ui.model, 'Готово к примерке', 'ok');
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
    const canRecognize = checkNativePlugin();
    try {
      setStatus(ui.model,'Открываю фотографию…');
      if(sourceBitmap&&sourceBitmap.close)sourceBitmap.close();
      sourceBitmap=await decodePhoto(file);
      sourceImage=makeSourceCanvas(sourceBitmap);
      ui.start.classList.add('hidden'); ui.editor.classList.remove('hidden');
      if (canRecognize) await recognize();
      else {
        probabilities = null;
        render();
        setStatus(ui.status, 'Просмотр фото. Примерка оттенков доступна в Android-приложении.', '');
        [ui.color, ui.opacity, ui.compare, $('afterBtn'), $('eraseBtn'), ui.save, ui.share, ...ui.palette.querySelectorAll('button')].forEach(button => button.disabled = true);
      }
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
  function fitCanvas(){
    if (!sourceImage) return;
    const scale = Math.min(ui.wrap.clientWidth / sourceImage.width, ui.wrap.clientHeight / sourceImage.height);
    if (!(scale > 0)) return;
    ui.canvas.style.width = sourceImage.width * scale + 'px';
    ui.canvas.style.height = sourceImage.height * scale + 'px';
  }
  if (window.ResizeObserver) new ResizeObserver(() => { fitCanvas(); clampView(); applyView(); }).observe(ui.wrap);
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
    maskCache = null;
    if(ui.undo) ui.undo.classList.remove('hidden');
    render();
    return true;
  }
  function undoErase(){
    if(!lastErased) return;
    for(let k = 0; k < lastErased.idx.length; k++) probabilities[lastErased.idx[k]] = lastErased.val[k];
    lastErased = null;
    maskCache = null;
    if(ui.undo) ui.undo.classList.add('hidden');
    render();
    toast('Пятно вернулось');
  }
  if(ui.undo) ui.undo.addEventListener('click', undoErase);

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
        if(eraseMode && !showingOriginal && eraseAt(cx, cy)) toast('Пятно убрано');
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
      cleanProbabilities();
      maskCache = null;
      lastErased = null;
      if(ui.undo) ui.undo.classList.add('hidden');
      resetView();
      render();
      const hint = advice(blobCount());
      if(hint) setStatus(ui.status, hint[0], hint[1]);
      else setStatus(ui.status, 'Готово', 'ok');
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

  // Закрашивает дырки внутри ногтей. Фон заливается от краёв кадра (здесь —
  // рамки w×h вокруг ногтей); всё фоновое, куда заливка не дошла, — это
  // отверстие внутри маски. Мелкие закрываем, крупные (промежутки между
  // пальцами) оставляем.
  function fillHoles(bin, w, h){
    const n=w*h, seen=new Uint8Array(n), stack=new Int32Array(n);
    let sp=0;
    const push=(i)=>{ if(!seen[i] && !bin[i]){ seen[i]=1; stack[sp++]=i; } };
    for(let x=0;x<w;x++){ push(x); push((h-1)*w+x); }
    for(let y=0;y<h;y++){ push(y*w); push(y*w+w-1); }
    while(sp>0){
      const i=stack[--sp], x=i%w, y=(i-x)/w;
      if(x>0) push(i-1); if(x<w-1) push(i+1);
      if(y>0) push(i-w); if(y<h-1) push(i+w);
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
        const x=i%w, y=(i-x)/w;
        const nb=[x>0?i-1:-1, x<w-1?i+1:-1, y>0?i-w:-1, y<h-1?i+w:-1];
        for(const j of nb) if(j>=0 && !bin[j] && !seen[j] && !done[j]){ done[j]=1; stack[sp++]=j; }
      }
      if(cells.length<=limit) for(const i of cells) bin[i]=1;
    }
    return bin;
  }

  // Связные пятна маски w×h (8 соседей): номер пятна у каждой точки, площадь и рамка.
  function labelBlobs(bin, w, h){
    const n=w*h, labels=new Int32Array(n).fill(-1), queue=new Int32Array(n), blobs=[];
    for(let s=0;s<n;s++){
      if(!bin[s]||labels[s]>=0) continue;
      const id=blobs.length; let head=0, tail=0, x0=w, y0=h, x1=0, y1=0;
      queue[tail++]=s; labels[s]=id;
      while(head<tail){
        const i=queue[head++], x=i%w, y=(i-x)/w;
        if(x<x0)x0=x; if(x>x1)x1=x; if(y<y0)y0=y; if(y>y1)y1=y;
        for(let dy=-1;dy<=1;dy++) for(let dx=-1;dx<=1;dx++){
          const nx=x+dx, ny=y+dy;
          if(nx<0||ny<0||nx>=w||ny>=h) continue;
          const j=ny*w+nx;
          if(bin[j]&&labels[j]<0){ labels[j]=id; queue[tail++]=j; }
        }
      }
      blobs.push({ area:tail, x0, y0, x1, y1 });
    }
    return { labels, blobs };
  }

  // Рамка вокруг ногтей в точках маски, с запасом под сглаживание. Считается
  // при чистке; стирание пятна её только сужает, так что она остаётся верной.
  // Вся дальнейшая обработка идёт в ней: на телефоне каждый проход по всей
  // маске 576² стоит десятки миллисекунд.
  let maskBox=null;
  const BOX_PAD=OPEN_R+Math.ceil(3*EDGE_BLUR)+2;

  // Чистка маски сразу после распознавания: крапинки долой, дырки в ногтях
  // закрыть. Правится сама вероятность, поэтому стирание пальцем и счёт
  // ногтей видят ту же маску, что и краска.
  function cleanProbabilities(){
    const side=maskSide, p=probabilities;
    let x0=side, y0=side, x1=-1, y1=-1;
    for(let y=0;y<side;y++){ const o=y*side; for(let x=0;x<side;x++) if(p[o+x]>THRESHOLD){ if(x<x0)x0=x; if(x>x1)x1=x; if(y<y0)y0=y; y1=y; } }
    maskBox=null;
    if(x1<0) return;
    const rx=Math.max(0,x0-BOX_PAD), ry=Math.max(0,y0-BOX_PAD);
    const w=Math.min(side,x1+BOX_PAD+1)-rx, h=Math.min(side,y1+BOX_PAD+1)-ry, n=w*h;
    const bin=new Uint8Array(n);
    for(let y=0;y<h;y++){ const o=(ry+y)*side+rx; for(let x=0;x<w;x++) bin[y*w+x]=p[o+x]>THRESHOLD?1:0; }
    const { labels, blobs } = labelBlobs(bin, w, h);
    let largest=0; for(const b of blobs) largest=Math.max(largest, b.area);
    const speck=new Uint8Array(blobs.length);
    blobs.forEach((b,id)=>{ speck[id]=b.area<largest*SPECK_FRAC?1:0; });
    const before=new Uint8Array(n);
    for(let y=0;y<h;y++){
      const o=(ry+y)*side+rx;
      for(let x=0;x<w;x++){ const i=y*w+x; if(labels[i]>=0&&speck[labels[i]]){ p[o+x]=0; bin[i]=0; } before[i]=bin[i]; }
    }
    fillHoles(bin, w, h);
    for(let y=0;y<h;y++){ const o=(ry+y)*side+rx; for(let x=0;x<w;x++) if(bin[y*w+x]&&!before[y*w+x]) p[o+x]=1; }
    // Рамку сужаем до оставшихся пятен: крапинка у края кадра её больше не держит.
    x0=w; y0=h; x1=-1; y1=-1;
    for(const b of blobs) if(b.area>=largest*SPECK_FRAC){ x0=Math.min(x0,b.x0); y0=Math.min(y0,b.y0); x1=Math.max(x1,b.x1); y1=Math.max(y1,b.y1); }
    const bx=Math.max(0,rx+x0-BOX_PAD), by=Math.max(0,ry+y0-BOX_PAD);
    maskBox={ x:bx, y:by, w:Math.min(side,rx+x1+BOX_PAD+1)-bx, h:Math.min(side,ry+y1+BOX_PAD+1)-by };
  }

  // Горячие циклы маски вынесены в функции «на строку». Их зовут сотни раз
  // за фото, и движок их быстро и надёжно оптимизирует. Длинный цикл внутри
  // функции, которую зовут раз на фото, после сброса оптимизации шёл в
  // интерпретаторе: на втором фото 1080×1920 одно размытие занимало полсекунды.

  // Скользящее окно ширины 2r+1 по строке src → dst: «есть ли единица» или «все ли единицы».
  function morphRow(src, dst, o, w, r, dilate){
    let sum=0, inWin=0;
    for(let k=0;k<r&&k<w;k++){ sum+=src[o+k]; inWin++; }
    for(let x=0;x<w;x++){
      if(x+r<w){ sum+=src[o+x+r]; inWin++; }
      if(x-r-1>=0){ sum-=src[o+x-r-1]; inWin--; }
      dst[o+x]=dilate ? (sum>0?1:0) : (sum===inWin?1:0);
    }
  }
  // sum[x] ± строка маски; для размытия поля — своя addRow, чтобы каждая
  // функция видела один тип массивов и оставалась быстрой.
  function countRow(sum, src, o, w, sign){ for(let x=0;x<w;x++) sum[x]+=sign*src[o+x]; }
  function thresholdRow(dst, o, sum, w, need, dilate){
    for(let x=0;x<w;x++) dst[o+x]=dilate ? (sum[x]>0?1:0) : (sum[x]===need?1:0);
  }

  // Сужение (dilate=false) или расширение квадратом (2r+1)²: окно скользит
  // по строке, потом суммы по столбцам идут построчно — так быстрее по памяти.
  function morph(src, w, h, r, dilate){
    const tmp=new Uint8Array(src.length), out=new Uint8Array(src.length);
    for(let y=0;y<h;y++) morphRow(src, tmp, y*w, w, r, dilate);
    const sum=new Int32Array(w); let inWin=0;
    for(let k=0;k<r&&k<h;k++){ countRow(sum, tmp, k*w, w, 1); inWin++; }
    for(let y=0;y<h;y++){
      if(y+r<h){ countRow(sum, tmp, (y+r)*w, w, 1); inWin++; }
      if(y-r-1>=0){ countRow(sum, tmp, (y-r-1)*w, w, -1); inWin--; }
      thresholdRow(out, y*w, sum, w, inWin, dilate);
    }
    return out;
  }

  // Выпуклая оболочка точек (монотонная цепь Эндрю).
  function convexHull(pts){
    pts.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
    const cross=(o,a,b)=>(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0]);
    const lower=[], upper=[];
    for(const p of pts){ while(lower.length>=2&&cross(lower[lower.length-2],lower[lower.length-1],p)<=0) lower.pop(); lower.push(p); }
    for(let i=pts.length-1;i>=0;i--){ const p=pts[i]; while(upper.length>=2&&cross(upper[upper.length-2],upper[upper.length-1],p)<=0) upper.pop(); upper.push(p); }
    return lower.slice(0,-1).concat(upper.slice(0,-1));
  }
  function polygonArea(poly){
    let s=0;
    for(let i=0,j=poly.length-1;i<poly.length;j=i++) s+=poly[j][0]*poly[i][1]-poly[i][0]*poly[j][1];
    return Math.abs(s)/2;
  }

  // Точки края пятна с точностью до доли точки маски: крайние точки каждой
  // строки и столбца, сдвинутые туда, где вероятность p пересекает порог.
  // Если край появился от срезания выступа, порога там нет — берём границу точки.
  function edgePoints(core, labels, p, id, b, w, h){
    const pts=[];
    const cut=(inside, outside)=>{
      const a=p[inside], o=p[outside];
      return o<=THRESHOLD && a>o ? (a-THRESHOLD)/(a-o) : 0.5;
    };
    const top=new Int32Array(b.x1-b.x0+1).fill(-1), bottom=new Int32Array(b.x1-b.x0+1);
    for(let y=b.y0;y<=b.y1;y++){
      let l=-1, r=-1;
      for(let x=b.x0;x<=b.x1;x++) if(labels[y*w+x]===id){
        if(l<0) l=x;
        r=x;
        if(top[x-b.x0]<0) top[x-b.x0]=y;
        bottom[x-b.x0]=y;
      }
      if(l<0) continue;
      const il=y*w+l, ir=y*w+r;
      pts.push([l>0&&!core[il-1] ? l+0.5-cut(il,il-1) : l, y+0.5]);
      pts.push([r<w-1&&!core[ir+1] ? r+0.5+cut(ir,ir+1) : r+1, y+0.5]);
    }
    for(let x=b.x0;x<=b.x1;x++){
      const t=top[x-b.x0], d=bottom[x-b.x0];
      if(t<0) continue;
      const it=t*w+x, ib=d*w+x;
      pts.push([x+0.5, t>0&&!core[it-w] ? t+0.5-cut(it,it-w) : t]);
      pts.push([x+0.5, d<h-1&&!core[ib+w] ? d+0.5+cut(ib,ib+w) : d+1]);
    }
    return pts;
  }

  // acc[x] += k·src[o+x] для всей строки.
  function addRow(acc, src, o, w, k){ for(let x=0;x<w;x++) acc[x]+=k*src[o+x]; }
  function blurRow(f, dst, o, w, k, r){
    for(let x=0;x<w;x++){
      let s=0;
      for(let j=-r;j<=r;j++){ const xx=x+j<0?0:x+j>=w?w-1:x+j; s+=k[j+r]*f[o+xx]; }
      dst[o+x]=s;
    }
  }
  // Гауссово размытие поля w×h по строкам и столбцам.
  function blurField(f, w, h, sigma){
    const r=Math.ceil(3*sigma), k=new Float32Array(2*r+1);
    let ks=0; for(let i=-r;i<=r;i++){ k[i+r]=Math.exp(-i*i/(2*sigma*sigma)); ks+=k[i+r]; }
    for(let i=0;i<k.length;i++) k[i]/=ks;
    const tmp=new Float32Array(f.length), out=new Float32Array(f.length);
    for(let y=0;y<h;y++) blurRow(f, tmp, y*w, w, k, r);
    for(let y=0;y<h;y++){
      const row=out.subarray(y*w, y*w+w);
      for(let j=-r;j<=r;j++) addRow(row, tmp, (y+j<0?0:y+j>=h?h-1:y+j)*w, w, k[j+r]);
    }
    return out;
  }
  // Одна строка фото из поля f (рамка w точек): билинейно между строками o0 и o1.
  function upscaleRow(out, oY, f, o0, o1, fy, col, colF, X0, X1){
    for(let X=X0;X<X1;X++){
      const i=col[X-X0];
      const a=f[o0+i], b=f[o0+i+1], c=f[o1+i], d=f[o1+i+1];
      if(a+b+c+d===0) continue;
      const fx=colF[X-X0], top=a+(b-a)*fx, bottom=c+(d-c)*fx;
      out[oY+X]=255*(top+(bottom-top)*fy);
    }
  }

  // Маска для краски в размере фото: 0..255 на точку, край ногтя — на уровне
  // maskLevel(). Ступеньки на краю брались из бинаризации на 512–576 точках:
  // растянутая втрое, диагональ ногтя становилась лесенкой. Поэтому ноготь
  // рисуется гладким многоугольником (оболочкой) с долями точки, поле
  // размывается и растягивается плавно, а порог берётся уже в размере фото.
  // Всё считается только в рамке maskBox; рамка в пикселях фото — X0…X1, Y0…Y1.
  function maskField(){
    const side=maskSide, p0=probabilities, W=geometry.w, H=geometry.h;
    const out=new Uint8ClampedArray(W*H);
    if(!maskBox) return { m:out, X0:0, X1:0, Y0:0, Y1:0 };
    const rx=maskBox.x, ry=maskBox.y, w=maskBox.w, h=maskBox.h, n=w*h;
    const p=new Float32Array(n), bin=new Uint8Array(n);
    for(let y=0;y<h;y++) for(let xx=0;xx<w;xx++){ const v=p0[(ry+y)*side+rx+xx]; p[y*w+xx]=v; bin[y*w+xx]=v>THRESHOLD?1:0; }
    const core=morph(morph(bin, w, h, OPEN_R, false), w, h, OPEN_R, true);
    for(let i=0;i<n;i++) core[i]&=bin[i];
    const { labels, blobs } = labelBlobs(core, w, h);

    // Оболочки рисует холст — он сглаживает край многоугольника по долям точки.
    const net=document.createElement('canvas');net.width=w;net.height=h;
    const x=net.getContext('2d',{willReadFrequently:true});
    x.fillStyle='#000'; x.fillRect(0,0,w,h); x.fillStyle='#fff';
    const raw=new Uint8Array(blobs.length);
    blobs.forEach((b,id)=>{
      const hull=convexHull(edgePoints(core, labels, p, id, b, w, h));
      if(hull.length<3 || polygonArea(hull)>b.area*(1+HULL_GAIN)){ raw[id]=1; return; }
      x.beginPath(); x.moveTo(hull[0][0],hull[0][1]);
      for(let i=1;i<hull.length;i++) x.lineTo(hull[i][0],hull[i][1]);
      x.closePath(); x.fill();
    });
    const im=x.getImageData(0,0,w,h), field=new Float32Array(n);
    for(let i=0;i<n;i++) field[i]=im.data[4*i]/255;
    // Пятна без оболочки — по самой вероятности, с мягким краем вокруг порога;
    // соседние с ними точки ниже порога дают растушёвку наружу.
    const soft=v=>Math.min(1,Math.max(0,0.5+(v-THRESHOLD)/(2*SOFT)));
    const edge=j=>{ if(!core[j]&&p[j]<=THRESHOLD) field[j]=Math.max(field[j], soft(p[j])); };
    for(let i=0;i<n;i++){
      if(labels[i]<0||!raw[labels[i]]) continue;
      field[i]=Math.max(field[i], soft(p[i]));
      const xx=i%w;
      if(xx>0) edge(i-1);
      if(xx<w-1) edge(i+1);
      if(i>=w) edge(i-w);
      if(i+w<n) edge(i+w);
    }
    const f=blurField(field, w, h, EDGE_BLUR);
    // До размера фото растягиваем сами, билинейно и только в рамке ногтей:
    // растянуть холстом и прочитать обратно стоило дороже всей чистки.
    // У края рамки точка берётся крайняя, как у холста: иначе ноготь,
    // упёршийся в край фото, оставался бы с незакрашенной полоской.
    const sx=geometry.dw/W, sy=geometry.dh/H;
    const X0=Math.max(0,Math.floor((rx-geometry.ox)/sx)), X1=Math.min(W,Math.ceil((rx+w-geometry.ox)/sx));
    const Y0=Math.max(0,Math.floor((ry-geometry.oy)/sy)), Y1=Math.min(H,Math.ceil((ry+h-geometry.oy)/sy));
    const col=new Int32Array(Math.max(0,X1-X0)), colF=new Float32Array(col.length);
    for(let X=X0;X<X1;X++){
      const u=Math.min(w-1,Math.max(0,geometry.ox+(X+0.5)*sx-0.5-rx)), i=Math.min(w-2,Math.floor(u));
      col[X-X0]=i; colF[X-X0]=u-i;
    }
    for(let Y=Y0;Y<Y1;Y++){
      const v=Math.min(h-1,Math.max(0,geometry.oy+(Y+0.5)*sy-0.5-ry)), j=Math.min(h-2,Math.floor(v)), fy=v-j;
      upscaleRow(out, Y*W, f, j*w, j*w+w, fy, col, colF, X0, X1);
    }
    return { m:out, X0, X1, Y0, Y1 };
  }
  // Уровень края и растушёвка (от lo до hi) в долях поля. Точка маски — это
  // geometry.w/geometry.dw пикселей фото, отсюда перевод из пикселей. На
  // маленьком фото растушёвка в долях поля шире самого уровня, и без нижней
  // границы краска ложилась на весь кадр — поэтому lo не ниже 0.02.
  function maskLevel(){
    const k=geometry.w/geometry.dw, level=Math.max(0.15, 0.5-EDGE_SLOPE*GROW_PX/k), band=EDGE_SLOPE*FEATHER_PX/k;
    return { level, lo:Math.max(0.02, level-band/2), hi:level+band/2 };
  }
  // Маска и пиксели фото не зависят от цвета: считаем один раз на фото
  // (и после стирания), а смена оттенка только перекрашивает.
  let maskCache=null;
  function maskData(){
    if(!maskCache){
      const started=performance.now(), w=sourceImage.width, h=sourceImage.height;
      maskCache=Object.assign(maskField(), { src:sourceImage.getContext('2d').getImageData(0,0,w,h), ms:Math.round(performance.now()-started) });
    }
    return maskCache;
  }
  function render(){
    fitCanvas();
    if(!sourceImage)return; const w=sourceImage.width,h=sourceImage.height;ui.canvas.width=w;ui.canvas.height=h;const out=ui.canvas.getContext('2d');out.drawImage(sourceImage,0,0);
    if(showingOriginal||!probabilities)return; const {m,src,X0,X1,Y0,Y1}=maskData(),{level,lo,hi}=maskLevel(),dst=out.createImageData(w,h),hex=ui.color.value,r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5),16),targetLum=.299*r+.587*g+.114*b,alpha=+ui.opacity.value/100,debug=false;
    // Средняя яркость САМИХ ногтей. Раньше блик и тень считались от яркости
    // выбранной краски: на светлом ногте отношение упиралось в потолок 1.65,
    // и малиновый #A81748 выходил ярко-розовым #FF3988 — человек нажимал
    // «малиновый», а ноготь не менялся. От средней по ногтям краска в среднем
    // получается ровно та, что выбрана, а блик и тень остаются на месте.
    // Вне рамки ногтей маска нулевая, поэтому оба цикла идут только по рамке.
    let sum=0,cnt=0;
    for(let y=Y0;y<Y1;y++)for(let i=y*w+X0;i<y*w+X1;i++){if(m[i]/255<=level)continue;const q=4*i;sum+=.299*src.data[q]+.587*src.data[q+1]+.114*src.data[q+2];cnt++;}
    const meanLum=cnt?sum/cnt:targetLum||128;
    // У края ногтя (запас, растушёвка и пара пикселей внутрь) в точку
    // подмешан фон. Светлый фон высветлял там краску, и ноготь обводился
    // ярким ободком, поэтому у края яркость краски только приглушается.
    // Блики внутри ногтя остаются: там маска полная.
    dst.data.set(src.data); for(let y=Y0;y<Y1;y++)for(let i=y*w+X0;i<y*w+X1;i++){const p=m[i]/255,a=Math.min(1,Math.max(0,(p-lo)/(hi-lo)))*alpha;if(a<.01)continue;const q=4*i;if(debug){dst.data[q]=255;dst.data[q+1]=45;dst.data[q+2]=130;continue;}const lum=.299*src.data[q]+.587*src.data[q+1]+.114*src.data[q+2],k=Math.max(.55,Math.min(p<.9?1:1.45,lum/(meanLum||1)));dst.data[q]=src.data[q]*(1-a)+Math.min(255,r*k)*a;dst.data[q+1]=src.data[q+1]*(1-a)+Math.min(255,g*k)*a;dst.data[q+2]=src.data[q+2]*(1-a)+Math.min(255,b*k)*a;}
    out.putImageData(dst,0,0);
  }
  function resultDataUrl(){compare(false);return ui.canvas.toDataURL('image/jpeg',.92);}

  ui.color.addEventListener('input',()=>selectedColor(ui.color.value));
  ui.opacity.addEventListener('input',()=>{ui.opacityValue.textContent=ui.opacity.value+'%';render();});
  ui.compare.addEventListener('click',()=>compare(true));
  $('afterBtn').addEventListener('click',()=>compare(false));
  $('eraseBtn').addEventListener('click',()=>{ eraseMode=!eraseMode; $('eraseBtn').setAttribute('aria-pressed',String(eraseMode)); if(eraseMode) compare(false); });

  // ===== Кнопка «Другое фото» =====
  if(ui.newPhoto) ui.newPhoto.addEventListener('click',()=>{
    if(sourceBitmap&&sourceBitmap.close)sourceBitmap.close();
    sourceBitmap=sourceImage=null; probabilities=null; maskCache=null; maskBox=null;
    eraseMode = false; showingOriginal = false;
    $('eraseBtn').setAttribute('aria-pressed','false');
    ui.compare.setAttribute('aria-pressed','false'); $('afterBtn').setAttribute('aria-pressed','true');
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
