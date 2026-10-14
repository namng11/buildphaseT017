/**
 * SegVision QA - Frontend Controller
 * Quản lý tương tác Canvas, Split Slider, Gọi API và Hiển thị số liệu IoU.
 */

// STATE TOÀN CỤC
const state = {
  mode: 'infer',        // 'infer' hoặc 'eval'
  viewMode: 'split',    // 'split', 'overlay', 'error', 'raw', 'gt'
  opacity: 0.6,
  splitRatio: 0.5,
  zoom: 1.0,
  
  imageFile: null,
  gtFile: null,
  
  classes: [],
  activeClasses: new Set(Array.from({ length: 19 }, (_, i) => i)), // Mặc định bật cả 19 lớp
  
  // Dữ liệu ảnh trả về từ API
  originalImg: new Image(),
  colorMaskImg: new Image(),
  gtColorImg: new Image(),
  errorMapImg: new Image(),
  
  apiResult: null
};

// DOM ELEMENTS
const el = {
  modeInferBtn: document.getElementById('modeInferBtn'),
  modeEvalBtn: document.getElementById('modeEvalBtn'),
  
  imageDropzone: document.getElementById('imageDropzone'),
  imageInput: document.getElementById('imageInput'),
  imageFileName: document.getElementById('imageFileName'),
  
  gtSection: document.getElementById('gtSection'),
  gtDropzone: document.getElementById('gtDropzone'),
  gtInput: document.getElementById('gtInput'),
  gtFileName: document.getElementById('gtFileName'),
  
  viewSplitBtn: document.getElementById('viewSplitBtn'),
  viewOverlayBtn: document.getElementById('viewOverlayBtn'),
  viewErrorBtn: document.getElementById('viewErrorBtn'),
  viewRawBtn: document.getElementById('viewRawBtn'),
  viewGtBtn: document.getElementById('viewGtBtn'),
  
  opacitySlider: document.getElementById('opacitySlider'),
  opacityVal: document.getElementById('opacityVal'),
  
  runBtn: document.getElementById('runBtn'),
  runBtnText: document.getElementById('runBtnText'),
  loadingSpinner: document.getElementById('loadingSpinner'),
  latencyBadge: document.getElementById('latencyBadge'),
  
  canvasContainer: document.getElementById('canvasContainer'),
  canvasEmptyState: document.getElementById('canvasEmptyState'),
  imageWrapper: document.getElementById('imageWrapper'),
  baseImage: document.getElementById('baseImage'),
  overlayCanvas: document.getElementById('overlayCanvas'),
  splitHandle: document.getElementById('splitHandle'),
  
  zoomInBtn: document.getElementById('zoomInBtn'),
  zoomOutBtn: document.getElementById('zoomOutBtn'),
  zoomResetBtn: document.getElementById('zoomResetBtn'),
  
  metricsDashboard: document.getElementById('metricsDashboard'),
  mIouValue: document.getElementById('mIouValue'),
  accValue: document.getElementById('accValue'),
  classListContainer: document.getElementById('classListContainer'),
  toggleAllClasses: document.getElementById('toggleAllClasses'),
  
  exportMaskBtn: document.getElementById('exportMaskBtn'),
  exportCsvBtn: document.getElementById('exportCsvBtn'),
  modelSelector: document.getElementById('modelSelector'),
  sampleChipsContainer: document.getElementById('sampleChipsContainer')
};

// 1. KHỞI TẠO ỨNG DỤNG
async function initApp() {
  await fetchModelInfo();
  await fetchClasses();
  await fetchDemoSamples();
  setupEventListeners();
}

async function fetchDemoSamples() {
  try {
    const res = await fetch('/api/samples');
    const data = await res.json();
    if (!el.sampleChipsContainer) return;
    el.sampleChipsContainer.innerHTML = '';
    
    const countBadge = document.getElementById('sampleCountBadge');
    if (countBadge) {
      countBadge.innerText = `${data.total || (data.samples ? data.samples.length : 0)} ảnh`;
    }

    if (!data.samples || data.samples.length === 0) {
      el.sampleChipsContainer.innerHTML = '<span class="text-[10px] text-slate-500">Chưa có ảnh mẫu</span>';
      return;
    }

    data.samples.forEach((sample, idx) => {
      const btn = document.createElement('button');
      btn.className = 'px-2 py-1 rounded bg-slate-800 hover:bg-sky-500/20 text-slate-300 hover:text-sky-300 border border-darkborder hover:border-sky-500/40 text-[11px] font-medium transition cursor-pointer shrink-0';
      btn.innerText = `Ảnh ${idx + 1}`;
      btn.title = `${sample.name} (${sample.mask_url ? 'Có nhãn Ground Truth' : 'Chỉ ảnh gốc'})`;
      btn.addEventListener('click', () => loadSamplePair(sample, idx + 1));
      el.sampleChipsContainer.appendChild(btn);
    });
  } catch (err) {
    console.warn('Không thể nạp demo samples:', err);
  }
}

async function loadSamplePair(sample, sampleNum) {
  try {
    el.latencyBadge.innerText = 'Đang nạp ảnh mẫu...';
    // Tải ảnh gốc
    const imgRes = await fetch(sample.image_url);
    const imgBlob = await imgRes.blob();
    state.imageFile = new File([imgBlob], `sample_${sampleNum}.jpg`, { type: 'image/jpeg' });
    el.imageFileName.innerText = `Ảnh mẫu ${sampleNum} (${sample.name.slice(0, 8)}...)`;

    // Tải ảnh mask (nếu có)
    if (sample.mask_url) {
      const maskRes = await fetch(sample.mask_url);
      const maskBlob = await maskRes.blob();
      state.gtFile = new File([maskBlob], `sample_${sampleNum}_mask.png`, { type: 'image/png' });
      el.gtFileName.innerText = `Nhãn thật mẫu ${sampleNum}`;
    }

    // Tự động chạy luôn
    runProcessing();
  } catch (err) {
    alert('Lỗi nạp ảnh mẫu: ' + err.message);
  }
}

async function fetchModelInfo() {
  try {
    const res = await fetch('/api/model_info');
    const data = await res.json();
    if (el.modelSelector) {
      if (data.current_model.includes("Fine-tuned")) {
        el.modelSelector.value = "fine_tuned";
      } else {
        el.modelSelector.value = "baseline";
      }
    }
  } catch (err) {
    console.warn('Không thể nạp thông tin model:', err);
  }
}

async function fetchClasses() {
  try {
    const res = await fetch('/api/classes');
    const data = await res.json();
    state.classes = data.classes;
    renderClassList();
  } catch (err) {
    console.error('Không thể nạp danh mục lớp:', err);
  }
}

// 2. RENDER DANH MỤC 19 CLASSES
function renderClassList() {
  el.classListContainer.innerHTML = '';
  
  state.classes.forEach(c => {
    const item = document.createElement('div');
    item.className = 'flex items-center justify-between p-2 rounded-lg bg-slate-800/40 hover:bg-slate-800/80 border border-transparent hover:border-darkborder text-xs transition cursor-pointer select-none';
    item.dataset.classId = c.id;

    // Check xem có số liệu thống kê hoặc IoU chưa
    let statBadge = '';
    if (state.apiResult) {
      if (state.mode === 'eval' && state.apiResult.classes) {
        const clsStat = state.apiResult.classes.find(x => x.id === c.id);
        if (clsStat && clsStat.present_in_gt) {
          const colorClass = clsStat.iou >= 70 ? 'text-emerald-400 bg-emerald-500/10' : (clsStat.iou >= 40 ? 'text-amber-400 bg-amber-500/10' : 'text-red-400 bg-red-500/10');
          statBadge = `<span class="px-1.5 py-0.5 rounded mono font-semibold text-[10px] ${colorClass}">${clsStat.iou}%</span>`;
        } else {
          statBadge = `<span class="text-slate-500 text-[10px]">N/A</span>`;
        }
      } else if (state.apiResult.distribution) {
        const dist = state.apiResult.distribution.find(x => x.id === c.id);
        if (dist) {
          statBadge = `<span class="mono text-sky-400 text-[10px]">${dist.percentage}%</span>`;
        }
      }
    }

    const isActive = state.activeClasses.has(c.id);

    item.innerHTML = `
      <div class="flex items-center space-x-2">
        <input type="checkbox" ${isActive ? 'checked' : ''} class="w-3.5 h-3.5 rounded bg-darkbg border-darkborder text-sky-500 focus:ring-0 cursor-pointer">
        <span class="w-2.5 h-2.5 rounded-full shrink-0" style="background-color: ${c.hex}"></span>
        <span class="text-slate-300 font-medium capitalize ${!isActive ? 'opacity-40 line-through' : ''}">${c.name}</span>
      </div>
      <div>${statBadge}</div>
    `;

    const checkbox = item.querySelector('input');
    checkbox.addEventListener('change', (e) => {
      e.stopPropagation();
      toggleSingleClass(c.id, checkbox.checked);
    });

    item.addEventListener('click', () => {
      checkbox.checked = !checkbox.checked;
      toggleSingleClass(c.id, checkbox.checked);
    });

    el.classListContainer.appendChild(item);
  });
}

function toggleSingleClass(classId, isChecked) {
  if (isChecked) {
    state.activeClasses.add(classId);
  } else {
    state.activeClasses.delete(classId);
  }
  renderClassList();
  renderOverlay();
}

// 3. TƯƠNG TÁC CHẾ ĐỘ & VIEW MODES
function setAppMode(newMode) {
  state.mode = newMode;
  if (newMode === 'infer') {
    el.modeInferBtn.className = 'px-3 py-1.5 rounded-md bg-brand-600 text-white shadow transition-all';
    el.modeEvalBtn.className = 'px-3 py-1.5 rounded-md text-slate-400 hover:text-white transition-all';
    el.gtSection.classList.add('hidden');
    el.metricsDashboard.classList.add('hidden');
    el.viewErrorBtn.classList.add('hidden');
    el.viewGtBtn.classList.add('hidden');
    el.runBtnText.innerText = 'Chạy Phân Vùng (Inference)';
  } else {
    el.modeEvalBtn.className = 'px-3 py-1.5 rounded-md bg-emerald-600 text-white shadow transition-all';
    el.modeInferBtn.className = 'px-3 py-1.5 rounded-md text-slate-400 hover:text-white transition-all';
    el.gtSection.classList.remove('hidden');
    el.viewErrorBtn.classList.remove('hidden');
    el.viewGtBtn.classList.remove('hidden');
    el.runBtnText.innerText = 'Chạy Kiểm Duyệt & Tính mIoU';
  }
  renderClassList();
}

function setViewMode(vMode) {
  state.viewMode = vMode;
  const btns = [
    { el: el.viewSplitBtn, mode: 'split' },
    { el: el.viewOverlayBtn, mode: 'overlay' },
    { el: el.viewErrorBtn, mode: 'error' },
    { el: el.viewRawBtn, mode: 'raw' },
    { el: el.viewGtBtn, mode: 'gt' }
  ];

  btns.forEach(b => {
    if (b.mode === vMode) {
      b.el.className = 'px-2.5 py-2 rounded-lg bg-sky-500/20 text-sky-300 border border-sky-500/40 font-medium text-left';
    } else {
      b.el.className = 'px-2.5 py-2 rounded-lg bg-darkbg text-slate-400 border border-darkborder hover:text-white font-medium text-left';
    }
  });

  if (vMode === 'split') {
    el.splitHandle.classList.remove('hidden');
  } else {
    el.splitHandle.classList.add('hidden');
  }

  renderOverlay();
}

// 4. VẼ VÀ ĐIỀU KHIỂN CANVAS
function renderOverlay() {
  if (!state.originalImg.src) return;

  const canvas = el.overlayCanvas;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  if (state.viewMode === 'raw') {
    return; // Chỉ hiện ảnh gốc
  }

  let layerImg = state.colorMaskImg;
  if (state.viewMode === 'error' && state.errorMapImg.src) {
    layerImg = state.errorMapImg;
  } else if (state.viewMode === 'gt' && state.gtColorImg.src) {
    layerImg = state.gtColorImg;
  }

  if (!layerImg.complete || !layerImg.src) return;

  if (state.viewMode === 'split') {
    // Chế độ Split: Nửa bên trái là ảnh gốc, nửa bên phải là Mask
    const splitX = w * state.splitRatio;
    ctx.save();
    ctx.beginPath();
    ctx.rect(splitX, 0, w - splitX, h);
    ctx.clip();
    ctx.globalAlpha = state.opacity;
    ctx.drawImage(layerImg, 0, 0, w, h);
    ctx.restore();

    // Cập nhật vị trí thanh gạt Handle
    el.splitHandle.style.left = `${state.splitRatio * 100}%`;
  } else {
    // Chế độ Overlay toàn phần
    ctx.globalAlpha = state.opacity;
    ctx.drawImage(layerImg, 0, 0, w, h);
  }
}

// 5. GỌI API BACKEND
async function runProcessing() {
  if (!state.imageFile) {
    alert('Vui lòng chọn hoặc kéo thả một ảnh đầu vào trước!');
    return;
  }

  if (state.mode === 'eval' && !state.gtFile) {
    alert('Vui lòng nạp thêm file ảnh nhãn thật Ground Truth Mask để tính mIoU!');
    return;
  }

  // Bật trạng thái Loading
  el.runBtn.disabled = true;
  el.loadingSpinner.classList.remove('hidden');

  const formData = new FormData();
  formData.append('image', state.imageFile);

  let url = '/api/predict';
  if (state.mode === 'eval') {
    url = '/api/evaluate';
    formData.append('ground_truth', state.gtFile);
  }

  try {
    const res = await fetch(url, { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Lỗi server');
    }

    const data = await res.json();
    state.apiResult = data;

    // Cập nhật thông số Latency
    el.latencyBadge.innerText = `⚡ ${data.latency.total_ms} ms (${data.latency.device})`;

    // Nạp ảnh Base64 vào các phần tử Image
    state.originalImg.src = data.original_base64;
    state.colorMaskImg.src = data.color_mask_base64;
    
    if (data.gt_color_base64) state.gtColorImg.src = data.gt_color_base64;
    if (data.error_map_base64) state.errorMapImg.src = data.error_map_base64;

    state.originalImg.onload = () => {
      el.baseImage.src = state.originalImg.src;
      el.canvasEmptyState.classList.add('hidden');
      el.imageWrapper.classList.remove('hidden');

      // Cập nhật kích thước Canvas trùng với ảnh gốc
      el.overlayCanvas.width = el.baseImage.naturalWidth;
      el.overlayCanvas.height = el.baseImage.naturalHeight;

      if (state.viewMode === 'split') {
        el.splitHandle.classList.remove('hidden');
      }

      renderOverlay();
    };

    // Nếu ở chế độ Eval: Hiện bảng chỉ số mIoU
    if (state.mode === 'eval' && data.mean_iou !== undefined) {
      el.metricsDashboard.classList.remove('hidden');
      el.mIouValue.innerText = `${data.mean_iou}%`;
      el.accValue.innerText = `${data.overall_accuracy}%`;
    }

    renderClassList();

  } catch (err) {
    alert('Lỗi xử lý: ' + err.message);
  } finally {
    el.runBtn.disabled = false;
    el.loadingSpinner.classList.add('hidden');
  }
}

// 6. XỬ LÝ SỰ KIỆN KÉO THẢ & CLICK
function setupEventListeners() {
  // Mode selection
  el.modeInferBtn.addEventListener('click', () => setAppMode('infer'));
  el.modeEvalBtn.addEventListener('click', () => setAppMode('eval'));

  // View modes
  el.viewSplitBtn.addEventListener('click', () => setViewMode('split'));
  el.viewOverlayBtn.addEventListener('click', () => setViewMode('overlay'));
  el.viewErrorBtn.addEventListener('click', () => setViewMode('error'));
  el.viewRawBtn.addEventListener('click', () => setViewMode('raw'));
  el.viewGtBtn.addEventListener('click', () => setViewMode('gt'));

  // Opacity Slider
  el.opacitySlider.addEventListener('input', (e) => {
    state.opacity = e.target.value / 100;
    el.opacityVal.innerText = `${e.target.value}%`;
    renderOverlay();
  });

  // Upload ảnh gốc
  el.imageDropzone.addEventListener('click', () => el.imageInput.click());
  el.imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      state.imageFile = e.target.files[0];
      el.imageFileName.innerText = state.imageFile.name;
    }
  });

  // Upload Ground Truth Mask
  el.gtDropzone.addEventListener('click', () => el.gtInput.click());
  el.gtInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      state.gtFile = e.target.files[0];
      el.gtFileName.innerText = state.gtFile.name;
    }
  });

  // Run Button
  el.runBtn.addEventListener('click', runProcessing);

  // Split Slider kéo chuột
  let isDragging = false;
  el.imageWrapper.addEventListener('mousedown', (e) => {
    if (state.viewMode !== 'split') return;
    isDragging = true;
    updateSplitFromMouse(e);
  });

  window.addEventListener('mousemove', (e) => {
    if (!isDragging || state.viewMode !== 'split') return;
    updateSplitFromMouse(e);
  });

  window.addEventListener('mouseup', () => { isDragging = false; });

  function updateSplitFromMouse(e) {
    const rect = el.imageWrapper.getBoundingClientRect();
    let x = e.clientX - rect.left;
    x = Math.max(0, Math.min(x, rect.width));
    state.splitRatio = x / rect.width;
    renderOverlay();
  }

  // Zoom buttons
  el.zoomInBtn.addEventListener('click', () => {
    state.zoom = Math.min(state.zoom + 0.25, 3.0);
    applyZoom();
  });
  el.zoomOutBtn.addEventListener('click', () => {
    state.zoom = Math.max(state.zoom - 0.25, 0.5);
    applyZoom();
  });
  el.zoomResetBtn.addEventListener('click', () => {
    state.zoom = 1.0;
    applyZoom();
  });

  function applyZoom() {
    el.imageWrapper.style.transform = `scale(${state.zoom})`;
    el.zoomResetBtn.innerText = `${Math.round(state.zoom * 100)}%`;
  }

  // Bật/tắt tất cả classes
  el.toggleAllClasses.addEventListener('click', () => {
    if (state.activeClasses.size > 0) {
      state.activeClasses.clear();
    } else {
      state.activeClasses = new Set(Array.from({ length: 19 }, (_, i) => i));
    }
    renderClassList();
    renderOverlay();
  });

  // Xuất Mask PNG
  el.exportMaskBtn.addEventListener('click', () => {
    if (!state.colorMaskImg.src) {
      alert('Chưa có kết quả để xuất!');
      return;
    }
    const a = document.createElement('a');
    a.href = state.colorMaskImg.src;
    a.download = 'segvision_predicted_mask.png';
    a.click();
  });

  // Xuất Báo cáo CSV
  el.exportCsvBtn.addEventListener('click', () => {
    if (!state.apiResult) {
      alert('Chưa có dữ liệu để xuất báo cáo!');
      return;
    }
    let csv = 'Class ID,Class Name,Color,IoU (%),Pixel Count,Percentage (%)\n';
    state.classes.forEach(c => {
      let iou = 'N/A';
      let px = '0';
      let pct = '0%';
      if (state.apiResult.classes) {
        const item = state.apiResult.classes.find(x => x.id === c.id);
        if (item) {
          iou = item.iou;
          px = item.pred_pixels;
        }
      }
      if (state.apiResult.distribution) {
        const dist = state.apiResult.distribution.find(x => x.id === c.id);
        if (dist) pct = dist.percentage + '%';
      }
      csv += `${c.id},"${c.name}","${c.hex}",${iou},${px},${pct}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'segvision_evaluation_report.csv';
    a.click();
  });

  // Chuyển đổi Model
  if (el.modelSelector) {
    el.modelSelector.addEventListener('change', async (e) => {
      const targetModel = e.target.value;
      const formData = new FormData();
      formData.append('target', targetModel);
      try {
        el.latencyBadge.innerText = 'Đang chuyển model...';
        const res = await fetch('/api/switch_model', { method: 'POST', body: formData });
        const data = await res.json();
        el.latencyBadge.innerText = '⚡ Sẵn sàng';
        // Nếu đang có ảnh nạp sẵn thì chạy lại suy luận với model mới luôn
        if (state.imageFile) {
          runProcessing();
        }
      } catch (err) {
        alert('Lỗi khi chuyển model: ' + err.message);
      }
    });
  }
}

// Chạy khởi tạo
window.addEventListener('DOMContentLoaded', initApp);
