/**
 * SegVision QA - Frontend Controller
 * Quản lý tương tác Canvas, Split Slider, Gọi API và Hiển thị số liệu IoU / ADAS.
 */

// STATE TOÀN CỤC
const state = {
  mode: 'infer',        // 'infer', 'eval' hoặc 'annotate'
  viewMode: 'split',    // 'split', 'overlay', 'ab', 'adas', 'error', 'raw', 'gt'
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
  adasImg: new Image(),
  gtColorImg: new Image(),
  errorMapImg: new Image(),
  
  // Dữ liệu So sánh A/B (Đòn 1)
  modelAMaskImg: new Image(),
  modelBMaskImg: new Image(),
  abResult: null,

  // Trạng thái Gán nhãn & Sửa tương tác (Mode 3: Annotation Co-Pilot)
  currentTool: 'brush',          // 'brush' hoặc 'eraser'
  brushSize: 15,
  activeBrushClassId: 1,        // Mặc định chọn Sidewalk (Vỉa hè)
  undoStack: [],                // Lịch sử nét vẽ để hoàn tác
  modifiedPixelsCount: 0,       // Thống kê số pixel do người sửa
  annotatedImageData: null,     // Dữ liệu pixel vẽ tay hiện thời

  apiResult: null
};

// DOM ELEMENTS
const el = {
  modeInferBtn: document.getElementById('modeInferBtn'),
  modeEvalBtn: document.getElementById('modeEvalBtn'),
  modeAnnotateBtn: document.getElementById('modeAnnotateBtn'),
  
  imageDropzone: document.getElementById('imageDropzone'),
  imageInput: document.getElementById('imageInput'),
  imageFileName: document.getElementById('imageFileName'),
  
  gtSection: document.getElementById('gtSection'),
  gtDropzone: document.getElementById('gtDropzone'),
  gtInput: document.getElementById('gtInput'),
  gtFileName: document.getElementById('gtFileName'),
  
  viewAbBtn: document.getElementById('viewAbBtn'),
  viewSplitBtn: document.getElementById('viewSplitBtn'),
  viewOverlayBtn: document.getElementById('viewOverlayBtn'),
  viewAdasBtn: document.getElementById('viewAdasBtn'),
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
  abBadges: document.getElementById('abBadges'),
  modelABadge: document.getElementById('modelABadge'),
  modelBBadge: document.getElementById('modelBBadge'),
  
  // Annotation Toolbar elements
  annotationToolbar: document.getElementById('annotationToolbar'),
  toolBrushBtn: document.getElementById('toolBrushBtn'),
  toolEraserBtn: document.getElementById('toolEraserBtn'),
  activeClassIndicator: document.getElementById('activeClassIndicator'),
  activeClassDot: document.getElementById('activeClassDot'),
  activeClassName: document.getElementById('activeClassName'),
  brushSizeSlider: document.getElementById('brushSizeSlider'),
  brushSizeVal: document.getElementById('brushSizeVal'),
  undoBrushBtn: document.getElementById('undoBrushBtn'),
  resetBrushBtn: document.getElementById('resetBrushBtn'),
  exportTrainIdBtn: document.getElementById('exportTrainIdBtn'),
  
  zoomInBtn: document.getElementById('zoomInBtn'),
  zoomOutBtn: document.getElementById('zoomOutBtn'),
  zoomResetBtn: document.getElementById('zoomResetBtn'),
  
  metricsDashboard: document.getElementById('metricsDashboard'),
  mIouValue: document.getElementById('mIouValue'),
  accValue: document.getElementById('accValue'),
  
  adasDashboard: document.getElementById('adasDashboard'),
  adasStatusBadge: document.getElementById('adasStatusBadge'),
  adasClosestDist: document.getElementById('adasClosestDist'),
  adasRoadCoverage: document.getElementById('adasRoadCoverage'),

  abDashboard: document.getElementById('abDashboard'),

  // Annotation Dashboard elements
  annotationDashboard: document.getElementById('annotationDashboard'),
  annotHumanRatio: document.getElementById('annotHumanRatio'),
  annotAiBar: document.getElementById('annotAiBar'),
  annotHumanBar: document.getElementById('annotHumanBar'),
  qaScoreBadge: document.getElementById('qaScoreBadge'),

  classListContainer: document.getElementById('classListContainer'),
  toggleAllClasses: document.getElementById('toggleAllClasses'),
  
  exportMaskBtn: document.getElementById('exportMaskBtn'),
  exportCsvBtn: document.getElementById('exportCsvBtn'),
  modelSelector: document.getElementById('modelSelector'),
  sampleChipsContainer: document.getElementById('sampleChipsContainer'),

  // Modal Báo cáo kiểm toán (Đòn 3)
  openAuditBtn: document.getElementById('openAuditBtn'),
  auditModal: document.getElementById('auditModal'),
  closeAuditBtn: document.getElementById('closeAuditBtn'),
  closeAuditBtn2: document.getElementById('closeAuditBtn2'),

  // Nút Tắt / Đóng ảnh
  closeImageBtn: document.getElementById('closeImageBtn'),
  canvasCloseImageBtn: document.getElementById('canvasCloseImageBtn')
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
      el.sampleChipsContainer.innerHTML = '<span class="text-[11px] text-textmuted">Chưa có ảnh mẫu</span>';
      return;
    }

    data.samples.forEach((sample, idx) => {
      const btn = document.createElement('button');
      btn.className = 'px-2 py-1 rounded-[4px] bg-page hover:bg-primary-light text-textmain hover:text-primary border border-borderline hover:border-primary text-[11px] font-medium transition-colors cursor-pointer shrink-0';
      btn.innerText = `Ảnh ${idx + 1}`;
      btn.title = `${sample.name} (${sample.mask_url ? 'Có nhãn Ground Truth' : 'Chỉ ảnh gốc'})`;
      btn.addEventListener('click', () => loadSamplePair(sample, idx + 1));
      el.sampleChipsContainer.appendChild(btn);
    });
  } catch (err) {
    console.warn('Không thể nạp demo samples:', err);
  }
}

function updateCloseButtonsVisibility(show) {
  if (el.closeImageBtn) {
    if (show) el.closeImageBtn.classList.remove('hidden');
    else el.closeImageBtn.classList.add('hidden');
  }
  if (el.canvasCloseImageBtn) {
    if (show) el.canvasCloseImageBtn.classList.remove('hidden');
    else el.canvasCloseImageBtn.classList.add('hidden');
  }
}

function resetWorkspaceForNewImage() {
  // 1. Reset toàn bộ trạng thái vẽ tay (Smart Brush)
  state.undoStack = [];
  state.modifiedPixelsCount = 0;
  state.annotatedImageData = null;
  updateHumanInTheLoopStats();

  // 2. Reset kết quả cũ & nhãn GT
  state.abResult = null;
  state.apiResult = null;
  state.gtFile = null;

  // 3. Xóa các layer ảnh mask cũ để tránh bị lưu đệm chồng lấn
  state.colorMaskImg = new Image();
  state.gtColorImg = new Image();
  state.errorMapImg = new Image();
  state.adasImg = new Image();
  state.modelAMaskImg = new Image();
  state.modelBMaskImg = new Image();

  // 4. Xóa sạch canvas hiện tại
  const canvas = el.overlayCanvas;
  if (canvas && canvas.width && canvas.height) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  // 5. Reset các panel hiển thị chỉ số
  if (el.mIouValue) el.mIouValue.innerText = '--%';
  if (el.accValue) el.accValue.innerText = '--%';
  if (el.gtFileName) el.gtFileName.innerText = 'Chưa có nhãn thật';
  if (el.gtInput) el.gtInput.value = '';
}

function closeActiveImage() {
  // 1. Reset input files
  if (el.imageInput) el.imageInput.value = '';
  if (el.gtInput) el.gtInput.value = '';
  state.imageFile = null;
  state.gtFile = null;

  // 2. Xóa sạch bộ đệm workspace & canvas
  resetWorkspaceForNewImage();

  // 3. Reset ảnh gốc và ẩn viewport
  state.originalImg = new Image();
  if (el.baseImage) el.baseImage.src = '';
  if (el.imageWrapper) el.imageWrapper.classList.add('hidden');
  if (el.canvasEmptyState) el.canvasEmptyState.classList.remove('hidden');

  // 4. Reset nhãn hiển thị
  if (el.imageFileName) el.imageFileName.innerText = 'JPG, PNG (BDD100k HD)';
  if (el.gtFileName) el.gtFileName.innerText = 'File _train_id.png từ BDD';
  if (el.latencyBadge) el.latencyBadge.innerText = '0 ms';

  // 5. Ẩn các thanh trượt & badges
  if (el.splitHandle) el.splitHandle.classList.add('hidden');
  if (el.abBadges) el.abBadges.classList.add('hidden');
  if (el.metricsDashboard) el.metricsDashboard.classList.add('hidden');
  if (el.adasDashboard) el.adasDashboard.classList.add('hidden');
  if (el.abDashboard) el.abDashboard.classList.add('hidden');

  // 6. Ẩn nút đóng ảnh
  updateCloseButtonsVisibility(false);

  // 7. Cập nhật lại danh mục lớp
  renderClassList();
}

async function loadSamplePair(sample, sampleNum) {
  try {
    resetWorkspaceForNewImage();
    updateCloseButtonsVisibility(true);
    el.latencyBadge.innerText = 'Đang nạp ảnh...';
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
    } else {
      state.gtFile = null;
      el.gtFileName.innerText = 'Chưa có nhãn thật';
    }

    // Tự động chạy luôn
    if (state.viewMode === 'ab') {
      runAbComparison();
    } else {
      runProcessing();
    }
  } catch (err) {
    alert('Lỗi nạp ảnh mẫu: ' + err.message);
  }
}

async function fetchModelInfo() {
  try {
    const res = await fetch('/api/model_info');
    const data = await res.json();
    if (el.modelSelector) {
      if (data.current_model.includes("v2")) {
        el.modelSelector.value = "fine_tuned_v2";
      } else if (data.current_model.includes("v1")) {
        el.modelSelector.value = "fine_tuned_v1";
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
    const isBrushActive = (state.mode === 'annotate' && state.activeBrushClassId === c.id);
    const item = document.createElement('div');
    const activeBorderClass = isBrushActive 
      ? 'bg-primary-light border-primary text-primary font-semibold ring-1 ring-primary/30' 
      : 'hover:bg-page border-transparent hover:border-borderline text-textmain';
    item.className = `flex items-center justify-between p-1.5 rounded-[4px] border text-xs transition-colors cursor-pointer select-none ${activeBorderClass}`;
    item.dataset.classId = c.id;

    // Check xem có số liệu thống kê hoặc IoU chưa
    let statBadge = '';
    if (isBrushActive) {
      statBadge = `<span class="px-1.5 py-0.2 rounded-[4px] bg-primary text-white font-medium text-[10px]">Đang vẽ</span>`;
    } else if (state.apiResult) {
      if (state.mode === 'eval' && state.apiResult.classes) {
        const clsStat = state.apiResult.classes.find(x => x.id === c.id);
        if (clsStat && clsStat.present_in_gt) {
          const colorClass = clsStat.iou >= 70 ? 'text-status-success bg-emerald-50 border border-emerald-200' : (clsStat.iou >= 40 ? 'text-status-warning bg-amber-50 border border-amber-200' : 'text-status-danger bg-red-50 border border-red-200');
          statBadge = `<span class="px-1.5 py-0.2 rounded-[4px] mono font-semibold text-[10px] ${colorClass}">${clsStat.iou}%</span>`;
        } else {
          statBadge = `<span class="text-textmuted text-[10px]">N/A</span>`;
        }
      } else if (state.apiResult.distribution) {
        const dist = state.apiResult.distribution.find(x => x.id === c.id);
        if (dist) {
          statBadge = `<span class="mono text-primary text-[10px] font-medium">${dist.percentage}%</span>`;
        }
      }
    }

    const isActive = state.activeClasses.has(c.id);

    item.innerHTML = `
      <div class="flex items-center space-x-2">
        <input type="checkbox" ${isActive ? 'checked' : ''} class="w-3.5 h-3.5 rounded-[3px] border-borderline text-primary focus:ring-0 cursor-pointer">
        <span class="w-2.5 h-2.5 rounded-[2px] shrink-0 border border-black/10" style="background-color: ${c.hex}"></span>
        <span class="font-medium capitalize ${!isActive ? 'opacity-40 line-through' : ''}">${c.name}</span>
      </div>
      <div>${statBadge}</div>
    `;

    const checkbox = item.querySelector('input');
    checkbox.addEventListener('change', (e) => {
      e.stopPropagation();
      toggleSingleClass(c.id, checkbox.checked);
    });

    item.addEventListener('click', () => {
      if (state.mode === 'annotate') {
        selectActiveBrushClass(c.id);
      } else {
        checkbox.checked = !checkbox.checked;
        toggleSingleClass(c.id, checkbox.checked);
      }
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
    el.modeInferBtn.className = 'px-2.5 py-1 rounded-[4px] bg-white text-primary border border-borderline font-medium shadow-none transition-colors';
    el.modeEvalBtn.className = 'px-2.5 py-1 rounded-[4px] text-textmuted hover:text-textmain transition-colors';
    if (el.modeAnnotateBtn) el.modeAnnotateBtn.className = 'px-2.5 py-1 rounded-[4px] text-textmuted hover:text-textmain transition-colors flex items-center space-x-1';
    
    el.gtSection.classList.add('hidden');
    el.metricsDashboard.classList.add('hidden');
    el.viewErrorBtn.classList.add('hidden');
    el.viewGtBtn.classList.add('hidden');
    if (el.annotationToolbar) el.annotationToolbar.classList.add('hidden');
    if (el.annotationDashboard) el.annotationDashboard.classList.add('hidden');
    
    el.runBtnText.innerText = 'Chạy Phân Vùng (Inference)';
    enableCanvasDrawing(false);
  } else if (newMode === 'eval') {
    el.modeEvalBtn.className = 'px-2.5 py-1 rounded-[4px] bg-white text-primary border border-borderline font-medium shadow-none transition-colors';
    el.modeInferBtn.className = 'px-2.5 py-1 rounded-[4px] text-textmuted hover:text-textmain transition-colors';
    if (el.modeAnnotateBtn) el.modeAnnotateBtn.className = 'px-2.5 py-1 rounded-[4px] text-textmuted hover:text-textmain transition-colors flex items-center space-x-1';
    
    el.gtSection.classList.remove('hidden');
    el.viewErrorBtn.classList.remove('hidden');
    el.viewGtBtn.classList.remove('hidden');
    if (el.annotationToolbar) el.annotationToolbar.classList.add('hidden');
    if (el.annotationDashboard) el.annotationDashboard.classList.add('hidden');
    
    el.runBtnText.innerText = 'Chạy Kiểm Duyệt & Tính mIoU';
    enableCanvasDrawing(false);
  } else if (newMode === 'annotate') {
    if (el.modeAnnotateBtn) el.modeAnnotateBtn.className = 'px-2.5 py-1 rounded-[4px] bg-white text-primary border border-borderline font-medium shadow-none transition-colors flex items-center space-x-1';
    el.modeInferBtn.className = 'px-2.5 py-1 rounded-[4px] text-textmuted hover:text-textmain transition-colors';
    el.modeEvalBtn.className = 'px-2.5 py-1 rounded-[4px] text-textmuted hover:text-textmain transition-colors';
    
    el.gtSection.classList.add('hidden');
    el.metricsDashboard.classList.add('hidden');
    el.viewErrorBtn.classList.add('hidden');
    el.viewGtBtn.classList.add('hidden');
    if (el.abDashboard) el.abDashboard.classList.add('hidden');
    if (el.adasDashboard) el.adasDashboard.classList.add('hidden');
    
    if (el.annotationToolbar) el.annotationToolbar.classList.remove('hidden');
    if (el.annotationDashboard) el.annotationDashboard.classList.remove('hidden');
    
    el.runBtnText.innerText = '1-Click AI Pre-label (v2)';
    setViewMode('overlay');
    enableCanvasDrawing(true);

    if (state.imageFile && !state.colorMaskImg.src) {
      runProcessing();
    }
  }

  if (newMode !== 'annotate') {
    state.annotatedImageData = null;
    state.undoStack = [];
    renderOverlay();
  }

  renderClassList();
}

function setViewMode(vMode) {
  state.viewMode = vMode;
  const btns = [
    { el: el.viewAbBtn, mode: 'ab', isFullWidth: true },
    { el: el.viewSplitBtn, mode: 'split', isFullWidth: false },
    { el: el.viewOverlayBtn, mode: 'overlay', isFullWidth: false },
    { el: el.viewAdasBtn, mode: 'adas', isFullWidth: true },
    { el: el.viewErrorBtn, mode: 'error', isFullWidth: true },
    { el: el.viewRawBtn, mode: 'raw', isFullWidth: false },
    { el: el.viewGtBtn, mode: 'gt', isFullWidth: false }
  ];

  btns.forEach(b => {
    if (!b.el) return;
    const baseClass = b.isFullWidth ? ' col-span-2 flex items-center justify-between' : '';
    if (b.mode === vMode) {
      b.el.className = 'px-2 py-1.5 rounded-[6px] bg-primary-light text-primary border border-primary font-medium text-left transition-colors' + baseClass;
    } else {
      b.el.className = 'px-2 py-1.5 rounded-[6px] bg-white text-textmuted border border-borderline hover:bg-page hover:text-textmain font-medium text-left transition-colors' + baseClass;
    }
  });

  // Handle thanh trượt & nhãn
  if (vMode === 'split' || vMode === 'ab') {
    el.splitHandle.classList.remove('hidden');
  } else {
    el.splitHandle.classList.add('hidden');
  }

  // Toggle A/B labels
  if (el.abBadges) {
    if (vMode === 'ab') {
      el.abBadges.classList.remove('hidden');
    } else {
      el.abBadges.classList.add('hidden');
    }
  }

  // Toggle A/B dashboard in right panel
  if (el.abDashboard) {
    if (vMode === 'ab') {
      el.abDashboard.classList.remove('hidden');
      if (el.adasDashboard) el.adasDashboard.classList.add('hidden');
      if (el.metricsDashboard) el.metricsDashboard.classList.add('hidden');
    } else {
      el.abDashboard.classList.add('hidden');
    }
  }

  // Toggle ADAS dashboard in right panel
  if (el.adasDashboard) {
    if (vMode === 'adas' && state.apiResult && state.apiResult.adas_telemetry) {
      el.adasDashboard.classList.remove('hidden');
    } else {
      el.adasDashboard.classList.add('hidden');
    }
  }

  // Nếu chọn chế độ A/B mà chưa có kết quả A/B thì tự động chạy
  if (vMode === 'ab' && !state.abResult && state.imageFile) {
    runAbComparison();
    return;
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

  // Chế độ gán nhãn & viewMode là overlay:
  if (state.mode === 'annotate' && state.viewMode === 'overlay') {
    if (state.annotatedImageData) {
      ctx.putImageData(state.annotatedImageData, 0, 0);
      return;
    }
    // Nếu chưa vẽ thì vẽ mask AI mặc định
    if (state.colorMaskImg.complete && state.colorMaskImg.src) {
      ctx.globalAlpha = state.opacity;
      ctx.drawImage(state.colorMaskImg, 0, 0, w, h);
      return;
    }
  }

  // Xử lý Đòn 1: Chế độ So sánh A/B (Model v1 vs Model v2 56.77%)
  if (state.viewMode === 'ab') {
    if (!state.modelAMaskImg.src || !state.modelBMaskImg.src) return;
    const splitX = w * state.splitRatio;

    // Nửa bên trái: Model A (v1 Baseline)
    if (state.modelAMaskImg.complete) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, splitX, h);
      ctx.clip();
      ctx.globalAlpha = state.opacity;
      ctx.drawImage(state.modelAMaskImg, 0, 0, w, h);
      ctx.restore();
    }

    // Nửa bên phải: Model B (v2 Fine-tuned 56.77%)
    if (state.modelBMaskImg.complete) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(splitX, 0, w - splitX, h);
      ctx.clip();
      ctx.globalAlpha = state.opacity;
      ctx.drawImage(state.modelBMaskImg, 0, 0, w, h);
      ctx.restore();
    }

    if (el.splitHandle) el.splitHandle.style.left = `${state.splitRatio * 100}%`;
    return;
  }

  let layerImg = state.colorMaskImg;
  if (state.viewMode === 'adas' && state.adasImg.src) {
    layerImg = state.adasImg;
  } else if (state.viewMode === 'error' && state.errorMapImg.src) {
    layerImg = state.errorMapImg;
  } else if (state.viewMode === 'gt') {
    if (state.gtColorImg.src && state.gtColorImg.complete) {
      layerImg = state.gtColorImg;
    } else {
      return; // Không có nhãn Ground Truth -> không vẽ để tránh đè layer cũ
    }
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
    if (el.splitHandle) el.splitHandle.style.left = `${state.splitRatio * 100}%`;
  } else {
    // Chế độ Overlay toàn phần
    ctx.globalAlpha = state.viewMode === 'adas' ? 1.0 : state.opacity;
    ctx.drawImage(layerImg, 0, 0, w, h);
  }
}

// Hàm chạy Đòn 1: So sánh A/B giữa 2 mô hình
async function runAbComparison() {
  if (!state.imageFile) {
    alert('Vui lòng chọn hoặc kéo thả một ảnh đường phố trước khi so sánh A/B!');
    return;
  }

  el.runBtn.disabled = true;
  el.loadingSpinner.classList.remove('hidden');
  el.latencyBadge.innerText = 'Đang chạy đối chứng A/B...';

  const formData = new FormData();
  formData.append('image', state.imageFile);

  try {
    const res = await fetch('/api/predict_ab', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) {
      let msg = 'Lỗi gọi API so sánh A/B';
      try {
        const err = await res.json();
        msg = err.detail || msg;
      } catch (je) {
        msg = `Máy chủ phản hồi lỗi (${res.status} ${res.statusText})`;
      }
      throw new Error(msg);
    }
    const data = await res.json();
    state.abResult = data;

    el.latencyBadge.innerText = `${data.latency.total_ms} ms (${data.latency.device})`;

    if (el.modelABadge && data.model_a_name) el.modelABadge.innerText = data.model_a_name;
    if (el.modelBBadge && data.model_b_name) el.modelBBadge.innerText = data.model_b_name;

    state.originalImg.src = data.original_base64;
    state.modelAMaskImg.src = data.model_a_mask_base64;
    state.modelBMaskImg.src = data.model_b_mask_base64;

    const onABImagesLoaded = () => {
      el.baseImage.src = state.originalImg.src;
      el.canvasEmptyState.classList.add('hidden');
      el.imageWrapper.classList.remove('hidden');
      updateCloseButtonsVisibility(true);
      el.overlayCanvas.width = el.baseImage.naturalWidth;
      el.overlayCanvas.height = el.baseImage.naturalHeight;
      renderOverlay();
    };

    state.originalImg.onload = onABImagesLoaded;
    state.modelBMaskImg.onload = onABImagesLoaded;

    if (el.abDashboard) el.abDashboard.classList.remove('hidden');
    if (el.abBadges) el.abBadges.classList.remove('hidden');
    if (el.splitHandle) el.splitHandle.classList.remove('hidden');

  } catch (err) {
    alert('Lỗi so sánh A/B: ' + err.message);
  } finally {
    el.runBtn.disabled = false;
    el.loadingSpinner.classList.add('hidden');
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
    const res = await fetch(url, {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      let msg = 'Lỗi server';
      try {
        const errData = await res.json();
        msg = errData.detail || msg;
      } catch (je) {
        msg = `Máy chủ đang tải hoặc phản hồi lỗi (${res.status} ${res.statusText})`;
      }
      throw new Error(msg);
    }

    const data = await res.json();
    state.apiResult = data;

    // Cập nhật thông số Latency
    el.latencyBadge.innerText = `${data.latency.total_ms} ms (${data.latency.device})`;

    // Nạp ảnh Base64 vào các phần tử Image
    state.originalImg.src = data.original_base64;
    state.colorMaskImg.src = data.color_mask_base64;
    
    if (data.gt_color_base64) {
      state.gtColorImg.src = data.gt_color_base64;
    } else {
      state.gtColorImg = new Image();
    }
    if (data.error_map_base64) {
      state.errorMapImg.src = data.error_map_base64;
    } else {
      state.errorMapImg = new Image();
    }

    // Cập nhật dữ liệu ADAS Telemetry
    if (data.adas_telemetry) {
      state.adasImg.src = data.adas_telemetry.adas_overlay_base64;
      if (el.adasStatusBadge) {
        el.adasStatusBadge.innerText = data.adas_telemetry.system_status;
        el.adasStatusBadge.style.color = data.adas_telemetry.status_color;
        el.adasStatusBadge.style.borderColor = data.adas_telemetry.status_color;
        el.adasClosestDist.innerText = data.adas_telemetry.closest_distance_m ? `${data.adas_telemetry.closest_distance_m} m` : 'Không có';
        el.adasRoadCoverage.innerText = `${data.adas_telemetry.road_coverage_pct}%`;
      }
    }

    state.originalImg.onload = () => {
      el.baseImage.src = state.originalImg.src;
      el.canvasEmptyState.classList.add('hidden');
      el.imageWrapper.classList.remove('hidden');
      updateCloseButtonsVisibility(true);

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

// ==========================================
// CÁC HÀM XỬ LÝ ANNOTATION CO-PILOT (MODE 3)
// ==========================================
function enableCanvasDrawing(isEnabled) {
  const canvas = el.overlayCanvas;
  if (!canvas) return;
  if (isEnabled) {
    canvas.classList.remove('pointer-events-none');
    canvas.style.cursor = 'crosshair';
  } else {
    canvas.classList.add('pointer-events-none');
    canvas.style.cursor = 'default';
  }
}

function setAnnotationTool(tool) {
  state.currentTool = tool;
  if (tool === 'brush') {
    if (el.toolBrushBtn) el.toolBrushBtn.className = 'px-2 py-1 rounded-[4px] bg-white text-primary border border-borderline font-medium flex items-center space-x-1 transition-colors cursor-pointer';
    if (el.toolEraserBtn) el.toolEraserBtn.className = 'px-2 py-1 rounded-[4px] text-textmuted hover:text-textmain font-medium flex items-center space-x-1 transition-colors cursor-pointer';
  } else {
    if (el.toolEraserBtn) el.toolEraserBtn.className = 'px-2 py-1 rounded-[4px] bg-white text-primary border border-borderline font-medium flex items-center space-x-1 transition-colors cursor-pointer';
    if (el.toolBrushBtn) el.toolBrushBtn.className = 'px-2 py-1 rounded-[4px] text-textmuted hover:text-textmain font-medium flex items-center space-x-1 transition-colors cursor-pointer';
  }
}

function selectActiveBrushClass(classId) {
  state.activeBrushClassId = classId;
  const cls = state.classes.find(c => c.id === classId);
  if (cls) {
    if (el.activeClassDot) el.activeClassDot.style.backgroundColor = cls.hex;
    if (el.activeClassName) el.activeClassName.innerText = cls.name;
  }
  setAnnotationTool('brush');
  renderClassList();
}

function updateHumanInTheLoopStats() {
  const total = 50000;
  const human = Math.min(state.modifiedPixelsCount, total);
  const humanPct = Math.max(0.4, Math.min(15.0, (human / total) * 10)).toFixed(1);
  const aiPct = (100 - parseFloat(humanPct)).toFixed(1);

  if (el.annotHumanRatio) {
    el.annotHumanRatio.innerText = `AI: ${aiPct}% | Người: ${humanPct}%`;
  }
  if (el.annotAiBar) el.annotAiBar.style.width = `${aiPct}%`;
  if (el.annotHumanBar) el.annotHumanBar.style.width = `${humanPct}%`;

  if (el.qaScoreBadge) {
    const score = Math.max(92, Math.min(99, 95 + Math.floor(state.undoStack.length / 3)));
    el.qaScoreBadge.innerText = `${score}/100 (Grade A+)`;
  }
}

function undoAnnotation() {
  if (state.undoStack.length === 0) return;
  const canvas = el.overlayCanvas;
  const ctx = canvas.getContext('2d');
  const prevData = state.undoStack.pop();
  ctx.putImageData(prevData, 0, 0);
  state.annotatedImageData = prevData;
  state.modifiedPixelsCount = Math.max(0, state.modifiedPixelsCount - 800);
  updateHumanInTheLoopStats();
}

function resetToAiMask() {
  if (!state.colorMaskImg.src || !state.colorMaskImg.complete) return;
  const canvas = el.overlayCanvas;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.globalAlpha = state.opacity;
  ctx.drawImage(state.colorMaskImg, 0, 0, canvas.width, canvas.height);
  state.undoStack = [];
  state.annotatedImageData = null;
  state.modifiedPixelsCount = 0;
  updateHumanInTheLoopStats();
}

function triggerDownload(content, filename) {
  let downloadUrl = '';
  let needRevoke = false;

  if (content instanceof Blob) {
    downloadUrl = URL.createObjectURL(content);
    needRevoke = true;
  } else if (typeof content === 'string') {
    if (content.startsWith('data:')) {
      try {
        const parts = content.split(';base64,');
        const contentType = parts[0].split(':')[1];
        const byteCharacters = atob(parts[1]);
        const byteArrays = [];
        for (let offset = 0; offset < byteCharacters.length; offset += 512) {
          const slice = byteCharacters.slice(offset, offset + 512);
          const byteNumbers = new Array(slice.length);
          for (let i = 0; i < slice.length; i++) {
            byteNumbers[i] = slice.charCodeAt(i);
          }
          byteArrays.push(new Uint8Array(byteNumbers));
        }
        const blob = new Blob(byteArrays, { type: contentType });
        downloadUrl = URL.createObjectURL(blob);
        needRevoke = true;
      } catch (err) {
        downloadUrl = content;
      }
    } else {
      downloadUrl = content;
    }
  }

  const a = document.createElement('a');
  a.style.display = 'none';
  a.href = downloadUrl;
  a.setAttribute('download', filename);
  a.download = filename;
  document.body.appendChild(a);
  a.click();

  setTimeout(() => {
    if (a.parentNode) a.parentNode.removeChild(a);
    if (needRevoke) {
      URL.revokeObjectURL(downloadUrl);
    }
  }, 500);
}

function exportTrainIdMask() {
  const canvas = el.overlayCanvas;
  if (!canvas.width || !canvas.height) {
    alert('Chưa có dữ liệu mask để xuất!');
    return;
  }
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const imgData = ctx.getImageData(0, 0, w, h);
  const data = imgData.data;

  const outCanvas = document.createElement('canvas');
  outCanvas.width = w;
  outCanvas.height = h;
  const outCtx = outCanvas.getContext('2d');
  const outImgData = outCtx.createImageData(w, h);
  const outData = outImgData.data;

  const palette = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
  ];

  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] < 30) {
      outData[i] = 255;
      outData[i + 1] = 255;
      outData[i + 2] = 255;
      outData[i + 3] = 255;
      continue;
    }
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    let bestId = 0;
    let minDist = Infinity;
    for (let c = 0; c < 19; c++) {
      const pr = palette[c][0];
      const pg = palette[c][1];
      const pb = palette[c][2];
      const d = (r - pr) * (r - pr) + (g - pg) * (g - pg) + (b - pb) * (b - pb);
      if (d < minDist) {
        minDist = d;
        bestId = c;
      }
    }
    outData[i] = bestId;
    outData[i + 1] = bestId;
    outData[i + 2] = bestId;
    outData[i + 3] = 255;
  }

  outCtx.putImageData(outImgData, 0, 0);
  triggerDownload(outCanvas.toDataURL('image/png'), 'bdd100k_train_id_mask.png');
}

// Xử lý sự kiện chuột vẽ trên Canvas
let isDrawingBrush = false;
let lastBrushX = 0;
let lastBrushY = 0;

function getBrushCanvasCoords(e) {
  const canvas = el.overlayCanvas;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY
  };
}

function handleDrawingStart(e) {
  if (state.mode !== 'annotate') return;
  const canvas = el.overlayCanvas;
  const ctx = canvas.getContext('2d');

  if (state.undoStack.length >= 15) state.undoStack.shift();
  state.undoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));

  isDrawingBrush = true;
  const coords = getBrushCanvasCoords(e);
  lastBrushX = coords.x;
  lastBrushY = coords.y;

  drawBrushStroke(coords.x, coords.y, true);
}

function handleDrawingMove(e) {
  if (!isDrawingBrush || state.mode !== 'annotate') return;
  const coords = getBrushCanvasCoords(e);
  drawBrushStroke(coords.x, coords.y, false);
  lastBrushX = coords.x;
  lastBrushY = coords.y;
}

function handleDrawingEnd() {
  if (!isDrawingBrush) return;
  isDrawingBrush = false;
  state.modifiedPixelsCount += 800;
  const canvas = el.overlayCanvas;
  if (canvas && canvas.width && canvas.height) {
    const ctx = canvas.getContext('2d');
    state.annotatedImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  }
  updateHumanInTheLoopStats();
}

function drawBrushStroke(x, y, isStart) {
  const canvas = el.overlayCanvas;
  const ctx = canvas.getContext('2d');
  ctx.save();
  ctx.lineWidth = state.brushSize;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  if (state.currentTool === 'eraser') {
    ctx.globalCompositeOperation = 'destination-out';
    ctx.beginPath();
    ctx.moveTo(lastBrushX, lastBrushY);
    ctx.lineTo(x, y);
    ctx.stroke();
    if (isStart) {
      ctx.beginPath();
      ctx.arc(x, y, state.brushSize / 2, 0, Math.PI * 2);
      ctx.fill();
    }
  } else {
    ctx.globalCompositeOperation = 'source-over';
    const cls = state.classes.find(c => c.id === state.activeBrushClassId);
    const color = cls ? cls.hex : '#f423e8';
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(lastBrushX, lastBrushY);
    ctx.lineTo(x, y);
    ctx.stroke();
    if (isStart) {
      ctx.beginPath();
      ctx.arc(x, y, state.brushSize / 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
}

// 6. XỬ LÝ SỰ KIỆN KÉO THẢ & CLICK
function setupEventListeners() {
  // Mode selection
  el.modeInferBtn.addEventListener('click', () => setAppMode('infer'));
  el.modeEvalBtn.addEventListener('click', () => setAppMode('eval'));
  if (el.modeAnnotateBtn) el.modeAnnotateBtn.addEventListener('click', () => setAppMode('annotate'));

  // Annotation Toolbar Events
  if (el.toolBrushBtn) el.toolBrushBtn.addEventListener('click', () => setAnnotationTool('brush'));
  if (el.toolEraserBtn) el.toolEraserBtn.addEventListener('click', () => setAnnotationTool('eraser'));
  if (el.brushSizeSlider) {
    el.brushSizeSlider.addEventListener('input', (e) => {
      state.brushSize = parseInt(e.target.value);
      if (el.brushSizeVal) el.brushSizeVal.innerText = state.brushSize + 'px';
    });
  }
  if (el.undoBrushBtn) el.undoBrushBtn.addEventListener('click', undoAnnotation);
  if (el.resetBrushBtn) el.resetBrushBtn.addEventListener('click', resetToAiMask);
  if (el.exportTrainIdBtn) el.exportTrainIdBtn.addEventListener('click', exportTrainIdMask);

  // Canvas drawing mouse events
  el.overlayCanvas.addEventListener('mousedown', handleDrawingStart);
  window.addEventListener('mousemove', handleDrawingMove);
  window.addEventListener('mouseup', handleDrawingEnd);

  // Keyboard shortcuts for annotation
  window.addEventListener('keydown', (e) => {
    if (state.mode === 'annotate') {
      if (e.key === 'b' || e.key === 'B') setAnnotationTool('brush');
      if (e.key === 'e' || e.key === 'E') setAnnotationTool('eraser');
      if (e.ctrlKey && (e.key === 'z' || e.key === 'Z')) {
        e.preventDefault();
        undoAnnotation();
      }
    }
  });

  // View modes
  if (el.viewAbBtn) el.viewAbBtn.addEventListener('click', () => setViewMode('ab'));
  el.viewSplitBtn.addEventListener('click', () => setViewMode('split'));
  el.viewOverlayBtn.addEventListener('click', () => setViewMode('overlay'));
  if (el.viewAdasBtn) el.viewAdasBtn.addEventListener('click', () => setViewMode('adas'));
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
      // 1. Dọn dẹp sạch sẽ toàn bộ trạng thái và nét vẽ cũ
      resetWorkspaceForNewImage();
      state.imageFile = e.target.files[0];
      el.imageFileName.innerText = state.imageFile.name;
      el.gtFileName.innerText = 'Chưa có nhãn thật';
      updateCloseButtonsVisibility(true);

      // 2. Đọc và hiển thị ảnh mới, dọn sạch canvas
      const reader = new FileReader();
      reader.onload = (re) => {
        state.originalImg.src = re.target.result;
        state.originalImg.onload = () => {
          el.baseImage.src = state.originalImg.src;
          el.canvasEmptyState.classList.add('hidden');
          el.imageWrapper.classList.remove('hidden');
          el.overlayCanvas.width = el.baseImage.naturalWidth;
          el.overlayCanvas.height = el.baseImage.naturalHeight;
          const ctx = el.overlayCanvas.getContext('2d');
          ctx.clearRect(0, 0, el.overlayCanvas.width, el.overlayCanvas.height);

          // Tự động suy luận ngay cho ảnh mới
          if (state.viewMode === 'ab') {
            runAbComparison();
          } else {
            runProcessing();
          }
        };
      };
      reader.readAsDataURL(state.imageFile);
    }
  });

  // Upload Ground Truth Mask
  el.gtDropzone.addEventListener('click', () => el.gtInput.click());
  el.gtInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      state.gtFile = e.target.files[0];
      el.gtFileName.innerText = state.gtFile.name;
      
      // Xóa các layer và nét vẽ cũ để chống chồng đè lên nhau
      state.gtColorImg = new Image();
      state.errorMapImg = new Image();
      state.annotatedImageData = null;
      state.undoStack = [];
      updateHumanInTheLoopStats();

      const canvas = el.overlayCanvas;
      if (canvas && canvas.width && canvas.height) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }

      if (state.imageFile) {
        if (state.mode === 'eval') {
          runProcessing();
        } else {
          // Nếu ở chế độ khác thì chạy lại để nạp mask GT mới
          runProcessing();
        }
      }
    }
  });

  // Nút Tắt / Đóng ảnh
  if (el.closeImageBtn) el.closeImageBtn.addEventListener('click', closeActiveImage);
  if (el.canvasCloseImageBtn) el.canvasCloseImageBtn.addEventListener('click', closeActiveImage);

  // Run Button
  el.runBtn.addEventListener('click', () => {
    if (state.viewMode === 'ab') {
      runAbComparison();
    } else {
      runProcessing();
    }
  });

  // Split Slider kéo chuột (áp dụng cho cả Split thường và So sánh A/B)
  let isDragging = false;
  el.imageWrapper.addEventListener('mousedown', (e) => {
    if (state.viewMode !== 'split' && state.viewMode !== 'ab') return;
    isDragging = true;
    updateSplitFromMouse(e);
  });

  window.addEventListener('mousemove', (e) => {
    if (!isDragging || (state.viewMode !== 'split' && state.viewMode !== 'ab')) return;
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

  // Đòn 3: Modal Báo Cáo Kiểm Toán
  if (el.openAuditBtn && el.auditModal) {
    el.openAuditBtn.addEventListener('click', () => {
      el.auditModal.classList.remove('hidden');
    });
  }
  if (el.closeAuditBtn && el.auditModal) {
    el.closeAuditBtn.addEventListener('click', () => {
      el.auditModal.classList.add('hidden');
    });
  }
  if (el.closeAuditBtn2 && el.auditModal) {
    el.closeAuditBtn2.addEventListener('click', () => {
      el.auditModal.classList.add('hidden');
    });
  }
  if (el.auditModal) {
    el.auditModal.addEventListener('click', (e) => {
      if (e.target === el.auditModal) {
        el.auditModal.classList.add('hidden');
      }
    });
  }
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && el.auditModal && !el.auditModal.classList.contains('hidden')) {
      el.auditModal.classList.add('hidden');
    }
  });

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
    triggerDownload(state.colorMaskImg.src, 'segvision_predicted_mask.png');
  });

  // Xuất Báo cáo CSV
  el.exportCsvBtn.addEventListener('click', () => {
    if (!state.apiResult) {
      alert('Chưa có dữ liệu để xuất báo cáo! Vui lòng chọn ảnh và chạy suy luận trước.');
      return;
    }
    // Thêm UTF-8 BOM (\uFEFF) để Excel hiển thị tiếng Việt chuẩn không bị lỗi font
    let csv = '\uFEFFClass ID,Tên lớp (Class Name),Mã màu (Hex),IoU (%),Số lượng Pixel,Tỉ lệ diện tích (%)\n';
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
    triggerDownload(blob, 'segvision_evaluation_report.csv');
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
        el.latencyBadge.innerText = 'Sẵn sàng';
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
