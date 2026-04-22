import { useEffect, useRef, useState } from "react";
import * as THREE from "three";

// ── Block world Y positions ───────────────────────────────
const BY = [-2.2, 0.6, 3.4];

// ── Steps with camera zoom targets ───────────────────────
const STEPS = [
  { id:0,  label:"User Input",               short:"INPUT",   detail:'"where is taj mahal" → raw string, 4 words, 19 chars',                   color:"#38bdf8", zone:"input",   cam:{ p:[0,-6.8,10],  t:[0,-6,0],   fov:42 } },
  { id:1,  label:"Tokenization (BPE)",        short:"TOKENS",  detail:"Byte-pair encoding: where/2073 · is/318 · taj/256 · mahal/4905",          color:"#34d399", zone:"token",   cam:{ p:[0,-5,9],     t:[0,-5,0],   fov:38 } },
  { id:2,  label:"Token + Pos Embeddings",    short:"EMBED",   detail:"E_tok ∈ R^{50257×768} lookup + sinusoidal E_pos → 4×768 residual stream", color:"#818cf8", zone:"embed",   cam:{ p:[0,-3.2,8],   t:[0,-3.2,0], fov:36 } },
  { id:3,  label:"Block 1 · Multi-Head Attn",short:"B1-MHA",  detail:"12 heads · Wq,Wk,Wv ∈ R^{768×768} → QKᵀ/√64 → softmax → ·V",           color:"#fbbf24", zone:"b0-attn", cam:{ p:[0, BY[0],6.2],t:[0,BY[0],0],fov:32 } },
  { id:4,  label:"Block 1 · FFN + Residual",  short:"B1-FFN",  detail:"Linear(768→3072) → GELU → Linear(3072→768) + skip + LayerNorm",          color:"#f59e0b", zone:"b0-ffn",  cam:{ p:[0, BY[0],6.2],t:[0,BY[0],0],fov:32 } },
  { id:5,  label:"Block 2 · Multi-Head Attn",short:"B2-MHA",  detail:'"taj" and "mahal" attend strongly · geographic concept forming',           color:"#fb923c", zone:"b1-attn", cam:{ p:[0, BY[1],6.2],t:[0,BY[1],0],fov:32 } },
  { id:6,  label:"Block 2 · FFN + Residual",  short:"B2-FFN",  detail:"Monument + location facts activate from frozen weight matrices",           color:"#ea580c", zone:"b1-ffn",  cam:{ p:[0, BY[1],6.2],t:[0,BY[1],0],fov:32 } },
  { id:7,  label:"Block N · Multi-Head Attn",short:"BN-MHA",  detail:"Entity fully resolved · factual retrieval pattern complete",               color:"#c084fc", zone:"b2-attn", cam:{ p:[0, BY[2],6.2],t:[0,BY[2],0],fov:32 } },
  { id:8,  label:"Block N · FFN + Residual",  short:"BN-FFN",  detail:'"Agra, Uttar Pradesh, India" crystallized in residual stream',             color:"#a855f7", zone:"b2-ffn",  cam:{ p:[0, BY[2],6.2],t:[0,BY[2],0],fov:32 } },
  { id:9,  label:"LayerNorm + Unembedding",   short:"PROJ",    detail:"Final LN → W_U ∈ R^{768×50257} → raw logits over full vocab",             color:"#f472b6", zone:"proj",    cam:{ p:[0,5.6,7.5],  t:[0,5.6,0],  fov:36 } },
  { id:10, label:"Softmax → Output Token",    short:"OUTPUT",  detail:'P("Taj Mahal is in Agra")≈0.82 → greedy decode → autoregressive loop',   color:"#10b981", zone:"output",  cam:{ p:[0,7.0,8],    t:[0,7.0,0],  fov:38 } },
];

const OV = { p:[0,0,19], t:[0,0,0], fov:52 };

// ── Canvas-texture label sprite ───────────────────────────
function spr(text, hex, sw=2.2, sh=0.32) {
  const c = document.createElement("canvas");
  c.width = 768; c.height = 96;
  const ctx = c.getContext("2d");
  ctx.font = "bold 38px 'Courier New', monospace";
  ctx.fillStyle = hex;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 384, 48);
  const tex = new THREE.CanvasTexture(c);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
  sp.scale.set(sw, sh, 1);
  return sp;
}

// ── Attention detail: QKV slabs + score heatmap ──────────
function buildAttnDetail(worldY) {
  const g = new THREE.Group();

  // Q, K, V matrix slabs
  const qkvC = [0x38bdf8, 0x34d399, 0xfb923c];
  const qkvL = ["Q  (query)", "K  (key)", "V  (value)"];
  qkvC.forEach((col, qi) => {
    const x = (qi - 1) * 1.85;
    const slab = new THREE.Mesh(
      new THREE.BoxGeometry(1.1, 1.7, 0.09),
      new THREE.MeshPhongMaterial({ color: col, emissive: col, emissiveIntensity: 0.18, transparent: true, opacity: 0.88 })
    );
    slab.position.set(x, worldY + 0.5, 0.9);
    g.add(slab);

    // Matrix row stripes
    for (let r = 0; r < 11; r++) {
      const stripe = new THREE.Mesh(
        new THREE.BoxGeometry(1.04, 0.013, 0.1),
        new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.38 })
      );
      stripe.position.set(x, worldY + 0.5 - 0.77 + r * 0.154, 0.94);
      g.add(stripe);
    }

    // Emissive border
    const edge = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(1.12, 1.72, 0.11)),
      new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.9 })
    );
    edge.position.set(x, worldY + 0.5, 0.9);
    g.add(edge);

    // Matrix dimension label
    const dimSp = spr("768×64", `#${col.toString(16).padStart(6,'0')}`, 0.9, 0.22);
    dimSp.position.set(x, worldY + 0.5 - 1.05, 0.9);
    g.add(dimSp);

    const labSp = spr(qkvL[qi], `#${col.toString(16).padStart(6,'0')}`, 1.55, 0.27);
    labSp.position.set(x, worldY + 0.5 + 1.06, 0.9);
    g.add(labSp);
  });

  // Attention score heatmap — extruded boxes per cell (4×4 = token × token)
  // Higher score = taller box = more attention
  const scores = [
    0.70, 0.12, 0.11, 0.07,
    0.14, 0.58, 0.18, 0.10,
    0.04, 0.09, 0.63, 0.24,
    0.04, 0.07, 0.32, 0.57,
  ];
  const gridX = 3.6, gridY = worldY - 0.1;
  const tokens4 = ["where","is","taj","mahal"];
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      const v = scores[r * 4 + c];
      const depth = 0.05 + v * 0.55;
      const hue = 0.08 + v * 0.07;
      const cellMesh = new THREE.Mesh(
        new THREE.BoxGeometry(0.33, 0.33, depth),
        new THREE.MeshPhongMaterial({
          color: new THREE.Color().setHSL(hue, 1.0, 0.1 + v * 0.48),
          emissive: new THREE.Color().setHSL(hue, 1.0, v * 0.22),
          transparent: true, opacity: 0.22 + v * 0.78
        })
      );
      cellMesh.position.set(gridX + (c - 1.5) * 0.37, gridY + (1.5 - r) * 0.37, 0.9 + depth / 2);
      g.add(cellMesh);
      const cellEdge = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(0.33, 0.33, depth)),
        new THREE.LineBasicMaterial({ color: 0x0a1830, transparent: true, opacity: 0.55 })
      );
      cellEdge.position.copy(cellMesh.position);
      g.add(cellEdge);
    }
    // Row token labels (y-axis)
    const rsp = spr(tokens4[r], "#2a4870", 0.85, 0.2);
    rsp.position.set(gridX - 0.85, gridY + (1.5 - r) * 0.37, 0.9);
    g.add(rsp);
  }
  // Col token labels (x-axis)
  tokens4.forEach((tk, c) => {
    const csp = spr(tk, "#2a4870", 0.85, 0.18);
    csp.position.set(gridX + (c - 1.5) * 0.37, gridY - 0.9, 0.9);
    g.add(csp);
  });
  const hmLab = spr("Attention Scores  QKᵀ / √d_k", "#94a3b8", 2.8, 0.26);
  hmLab.position.set(gridX, gridY - 1.22, 0.9);
  g.add(hmLab);

  // Flow arrows: QKV → heatmap (simple thin boxes)
  [-1.85, 0, 1.85].forEach(x => {
    const arr = new THREE.Mesh(
      new THREE.BoxGeometry(0.025, 0.025, 0.7),
      new THREE.MeshBasicMaterial({ color: 0x1e3a55, transparent: true, opacity: 0.6 })
    );
    arr.position.set(x, worldY + 0.5, 1.35);
    g.add(arr);
  });

  // Title
  const titleSp = spr("Multi-Head Self-Attention  (12 heads)", "#60a0c0", 3.4, 0.28);
  titleSp.position.set(1.5, worldY + 0.5 + 1.65, 0.9);
  g.add(titleSp);

  // Softmax annotation
  const sfSp = spr("→  softmax  →  × V  →  concat  →  W_O", "#3a5570", 3.2, 0.23);
  sfSp.position.set(gridX, gridY + 1.1, 0.9);
  g.add(sfSp);

  return g;
}

// ── FFN detail: 3 neuron layers + connections + GELU ─────
function buildFFNDetail(worldY) {
  const g = new THREE.Group();

  const layerDefs = [
    { cols: 7, rows: 6, x: -2.5, col: 0x38bdf8, dim: "d_model = 768",  sp: 0.215 },
    { cols: 9, rows: 7, x:  0,   col: 0xfbbf24, dim: "d_ff = 3072",    sp: 0.215 },
    { cols: 7, rows: 6, x:  2.5, col: 0x34d399, dim: "d_model = 768",  sp: 0.215 },
  ];
  const layerPts = [];

  layerDefs.forEach(({ cols, rows, x, col, dim, sp }) => {
    const pts = [];
    const geo = new THREE.SphereGeometry(0.05, 8, 8);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const m = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({ color: col, emissive: col, emissiveIntensity: 0.38 }));
        const px = x + (c - (cols - 1) / 2) * sp;
        const py = worldY + (r - (rows - 1) / 2) * sp;
        m.position.set(px, py, 0.85);
        g.add(m);
        pts.push(new THREE.Vector3(px, py, 0.85));
      }
    }
    layerPts.push(pts);

    // Bounding frame
    const frame = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(cols * sp + 0.14, rows * sp + 0.14, 0.1)),
      new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.22 })
    );
    frame.position.set(x, worldY, 0.85);
    g.add(frame);

    const dimSp = spr(dim, `#${col.toString(16).padStart(6, '0')}`, 1.6, 0.24);
    dimSp.position.set(x, worldY - rows * sp / 2 - 0.38, 0.85);
    g.add(dimSp);
  });

  // Connection lines (sampled)
  const addLines = (fromPts, toPts, n, col) => {
    for (let i = 0; i < n; i++) {
      const f = fromPts[Math.floor(Math.random() * fromPts.length)];
      const t = toPts[Math.floor(Math.random() * toPts.length)];
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([f, t]),
        new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.07 + Math.random() * 0.14 })
      );
      g.add(line);
    }
  };
  addLines(layerPts[0], layerPts[1], 55, 0xfbbf24);
  addLines(layerPts[1], layerPts[2], 55, 0x34d399);

  // GELU activation curve
  const geluPts = [];
  for (let xi = -3.2; xi <= 3.2; xi += 0.06) {
    const yi = xi * 0.5 * (1 + Math.tanh(0.7978 * (xi + 0.044715 * xi * xi * xi)));
    geluPts.push(new THREE.Vector3(xi * 0.3, worldY - 1.1 + yi * 0.26, 0.85));
  }
  const geluLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(geluPts),
    new THREE.LineBasicMaterial({ color: 0xfbbf24, transparent: true, opacity: 0.95 })
  );
  g.add(geluLine);

  const gSp = spr("GELU activation  f(x) = x·Φ(x)", "#f59e0b", 2.6, 0.25);
  gSp.position.set(0, worldY - 1.65, 0.85);
  g.add(gSp);

  const titleSp = spr("Feed-Forward Network  (4× expansion + residual + LN)", "#60a0c0", 4.2, 0.28);
  titleSp.position.set(0, worldY + 1.15, 0.85);
  g.add(titleSp);

  return g;
}

// ── Embedding detail: bar-chart per token + sinusoid ─────
function buildEmbedDetail(worldY) {
  const g = new THREE.Group();
  const tc = [0x38bdf8, 0x34d399, 0xa78bfa, 0xfb7185];
  const tn = ["where", "is", "taj", "mahal"];

  tc.forEach((col, ti) => {
    const x = (ti - 1.5) * 1.5;
    // 14 embedding dimension bars
    for (let b = 0; b < 14; b++) {
      const h = 0.08 + Math.abs(Math.sin(b * 1.45 + ti * 2.4)) * 0.72;
      const bar = new THREE.Mesh(
        new THREE.BoxGeometry(0.07, h, 0.07),
        new THREE.MeshPhongMaterial({ color: col, emissive: col, emissiveIntensity: 0.28, transparent: true, opacity: 0.88 })
      );
      bar.position.set(x + (b - 6.5) * 0.092, worldY + h / 2, 0.85);
      g.add(bar);
    }
    // Positional encoding sine wave
    const ppts = [];
    for (let xi = 0; xi <= 13.5; xi += 0.18) {
      ppts.push(new THREE.Vector3(x + (xi - 6.5) * 0.092, worldY + 0.92 + Math.sin(xi * 0.62 + ti * 1.1) * 0.22, 0.85));
    }
    const posLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(ppts),
      new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.75 })
    );
    g.add(posLine);

    const tokSp = spr(tn[ti], `#${col.toString(16).padStart(6, '0')}`, 1.0, 0.26);
    tokSp.position.set(x, worldY - 0.55, 0.85);
    g.add(tokSp);

    // Token ID label
    const ids = ["2073", "318", "256", "4905"];
    const idSp = spr(`id:${ids[ti]}`, "#1e3a50", 0.85, 0.2);
    idSp.position.set(x, worldY - 0.82, 0.85);
    g.add(idSp);
  });

  const t1 = spr("Token Embedding  E_tok  (50257 × 768 table)", "#818cf8", 4.0, 0.27);
  t1.position.set(0, worldY + 1.35, 0.85);
  g.add(t1);
  const t2 = spr("+ Positional Encoding  E_pos  (sinusoidal)", "#4a5068", 3.6, 0.24);
  t2.position.set(0, worldY + 1.05, 0.85);
  g.add(t2);

  return g;
}

// ── Output detail: softmax probability bars ───────────────
function buildOutputDetail(worldY) {
  const g = new THREE.Group();
  const vocab = [
    { text: "Taj Mahal is in Agra, India", prob: 0.82, col: 0x10b981 },
    { text: "Taj Mahal, located in Agra",  prob: 0.09, col: 0x34d399 },
    { text: "The Taj Mahal stands in",     prob: 0.05, col: 0x6ee7b7 },
    { text: "Taj Mahal → Agra district",   prob: 0.03, col: 0x065f46 },
    { text: "Other tokens...",             prob: 0.01, col: 0x022c22 },
  ];

  vocab.forEach(({ text, prob, col }, i) => {
    const w = prob * 4.8;
    const bar = new THREE.Mesh(
      new THREE.BoxGeometry(w, 0.22, 0.09),
      new THREE.MeshPhongMaterial({ color: col, emissive: col, emissiveIntensity: 0.32, transparent: true, opacity: 0.92 })
    );
    bar.position.set(-2.4 + w / 2, worldY + (2 - i) * 0.32, 0.85);
    g.add(bar);

    const pSp = spr(`${(prob * 100).toFixed(0)}%`, `#${col.toString(16).padStart(6, '0')}`, 0.65, 0.2);
    pSp.position.set(-2.4 + w + 0.5, worldY + (2 - i) * 0.32, 0.85);
    g.add(pSp);
  });

  const titSp = spr("Softmax( W_U · h_N )  →  vocab logits (50,257)", "#f472b6", 4.0, 0.28);
  titSp.position.set(0.5, worldY + 1.6, 0.85);
  g.add(titSp);

  const noteSp = spr("→ greedy decode → autoregressive next-token", "#3a5570", 3.0, 0.22);
  noteSp.position.set(0.5, worldY + 1.28, 0.85);
  g.add(noteSp);

  return g;
}

// ── Main component ────────────────────────────────────────
export default function GPTVisualizer() {
  const mountRef = useRef(null);
  const refs     = useRef({});
  const stepRef  = useRef(0);
  const rotRef   = useRef({ y: 0.08, x: 0.06 });
  const [step, setStep]       = useState(0);
  const [playing, setPlaying] = useState(false);
  const [isZoomed, setIsZoomed] = useState(false);

  // Auto-play
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setStep(s => { const n = (s + 1) % STEPS.length; stepRef.current = n; return n; });
    }, 2800);
    return () => clearInterval(id);
  }, [playing]);

  // Step change → update camera target + detail visibility
  useEffect(() => {
    const r = refs.current;
    if (!r.camTarget) return;
    const s = STEPS[step];
    r.camTarget.p.set(...s.cam.p);
    r.camTarget.t.set(...s.cam.t);
    r.camTarget.fov = s.cam.fov;
    setIsZoomed(true);
    if (r.detailGroups) {
      Object.entries(r.detailGroups).forEach(([zone, grp]) => {
        grp.visible = (zone === s.zone);
      });
    }
    if (r.blockHighlights) {
      const zoneMap = { "b0-attn":0,"b0-ffn":0,"b1-attn":1,"b1-ffn":1,"b2-attn":2,"b2-ffn":2 };
      const aIdx = zoneMap[s.zone];
      r.blockHighlights.forEach(({ attnMesh, ffnMesh, idx }) => {
        const active = idx === aIdx;
        attnMesh.material.emissive.set(active ? s.color : "#030608");
        attnMesh.material.emissiveIntensity = active ? 0.28 : 0.06;
        ffnMesh.material.emissive.set(active ? s.color : "#030608");
        ffnMesh.material.emissiveIntensity = active ? 0.22 : 0.06;
      });
    }
    if (r.pMat) r.pMat.color.set(s.color);
  }, [step]);

  // Three.js setup
  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;
    const W = el.clientWidth, H = el.clientHeight;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setClearColor(0x020510);
    el.appendChild(renderer.domElement);

    const scene  = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(OV.fov, W / H, 0.1, 300);
    camera.position.set(...OV.p);

    // Lights
    scene.add(new THREE.AmbientLight(0x0a1428, 3.2));
    const pl1 = new THREE.PointLight(0x3377ff, 6, 70);  pl1.position.set(10, 8, 14);  scene.add(pl1);
    const pl2 = new THREE.PointLight(0xff4422, 3.5, 55); pl2.position.set(-9, -7, 10); scene.add(pl2);
    const pl3 = new THREE.PointLight(0xaa33ff, 3, 40);  pl3.position.set(2, 9, 8);    scene.add(pl3);

    // Stars
    const sArr = new Float32Array(7000 * 3);
    for (let i = 0; i < 21000; i++) sArr[i] = (Math.random() - 0.5) * 280;
    const sGeo = new THREE.BufferGeometry();
    sGeo.setAttribute("position", new THREE.BufferAttribute(sArr, 3));
    scene.add(new THREE.Points(sGeo, new THREE.PointsMaterial({ color: 0x091525, size: 0.1 })));

    // ── Token cubes at bottom ──
    const TC = [0x38bdf8, 0x34d399, 0xa78bfa, 0xfb7185];
    const TL = ["where", "is", "taj", "mahal"];
    const tokCubes = TC.map((col, i) => {
      const m = new THREE.Mesh(
        new THREE.BoxGeometry(0.8, 0.8, 0.8),
        new THREE.MeshPhongMaterial({ color: col, emissive: col, emissiveIntensity: 0.28, transparent: true, opacity: 0.92 })
      );
      m.position.set((i - 1.5) * 1.2, -6.6, 0);
      scene.add(m);
      const edge = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(0.82, 0.82, 0.82)),
        new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.6 })
      );
      edge.position.copy(m.position);
      scene.add(edge);
      const sp = spr(TL[i], `#${col.toString(16).padStart(6,'0')}`, 0.9, 0.26);
      sp.position.set((i - 1.5) * 1.2, -7.28, 0);
      scene.add(sp);
      return m;
    });

    // ── Residual stream tube ──
    const rPts = Array.from({ length: 20 }, (_, i) => new THREE.Vector3(0, -7.3 + i * 0.82, 0));
    const rTube = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(rPts), 80, 0.04, 8, false);
    scene.add(new THREE.Mesh(rTube, new THREE.MeshPhongMaterial({ color: 0x0c235a, emissive: 0x04091e, emissiveIntensity: 1 })));

    // ── Transformer blocks ──
    const BCFG = [
      { y: BY[0], aC: 0xfbbf24, mC: 0xd97706, label: "Block 1" },
      { y: BY[1], aC: 0xfb923c, mC: 0xea580c, label: "Block 2" },
      { y: BY[2], aC: 0xc084fc, mC: 0x9333ea, label: "Block N" },
    ];
    const blockHighlights = [];

    BCFG.forEach((cfg, bi) => {
      const grp = new THREE.Group();

      // Glass shell
      const shell = new THREE.Mesh(
        new THREE.BoxGeometry(6.8, 1.15, 2.1),
        new THREE.MeshPhongMaterial({ color: 0x07091a, transparent: true, opacity: 0.2, side: THREE.DoubleSide })
      );
      grp.add(shell);
      const shellEdge = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(6.82, 1.17, 2.12)),
        new THREE.LineBasicMaterial({ color: 0x0c1630 })
      );
      grp.add(shellEdge);

      // MHA slab
      const attnSlab = new THREE.Mesh(
        new THREE.BoxGeometry(6.4, 0.4, 1.95),
        new THREE.MeshPhongMaterial({ color: cfg.aC, emissive: 0x030608, emissiveIntensity: 0.06, transparent: true, opacity: 0.58 })
      );
      attnSlab.position.y = 0.35;
      grp.add(attnSlab);

      // QKV mini-slabs (visible even in overview)
      const qkvC = [0x38bdf8, 0x34d399, 0xfb923c];
      const qkvL = ["Q", "K", "V"];
      qkvC.forEach((c, qi) => {
        const qs = new THREE.Mesh(
          new THREE.BoxGeometry(0.45, 0.28, 0.22),
          new THREE.MeshPhongMaterial({ color: c, emissive: c, emissiveIntensity: 0.4, transparent: true, opacity: 0.9 })
        );
        qs.position.set((qi - 1) * 1.75, 0.35, 0.58);
        grp.add(qs);
        const qEdge = new THREE.LineSegments(
          new THREE.EdgesGeometry(new THREE.BoxGeometry(0.46, 0.29, 0.23)),
          new THREE.LineBasicMaterial({ color: c, transparent: true, opacity: 0.7 })
        );
        qEdge.position.copy(qs.position);
        grp.add(qEdge);
        const qSp = spr(qkvL[qi], `#${c.toString(16).padStart(6,'0')}`, 0.42, 0.24);
        qSp.position.set((qi - 1) * 1.75, 0.75, 0.58);
        grp.add(qSp);
      });

      // 8 attention head spheres (back row)
      for (let h = 0; h < 8; h++) {
        const hs = new THREE.Mesh(
          new THREE.SphereGeometry(0.075, 10, 10),
          new THREE.MeshPhongMaterial({ color: cfg.aC, emissive: cfg.aC, emissiveIntensity: 0.38 })
        );
        hs.position.set((h - 3.5) * 0.72, 0.35, -0.62);
        grp.add(hs);
      }

      // FFN slab
      const ffnSlab = new THREE.Mesh(
        new THREE.BoxGeometry(5.8, 0.4, 1.95),
        new THREE.MeshPhongMaterial({ color: cfg.mC, emissive: 0x030608, emissiveIntensity: 0.06, transparent: true, opacity: 0.58 })
      );
      ffnSlab.position.y = -0.35;
      grp.add(ffnSlab);

      // FFN neurons
      for (let r = 0; r < 2; r++) {
        for (let nc = 0; nc < 8; nc++) {
          const ns = new THREE.Mesh(
            new THREE.SphereGeometry(0.055, 7, 7),
            new THREE.MeshPhongMaterial({ color: cfg.mC, emissive: cfg.mC, emissiveIntensity: 0.28 })
          );
          ns.position.set((nc - 3.5) * 0.68, -0.35 + r * 0.22 - 0.11, -0.62);
          grp.add(ns);
        }
      }

      // LayerNorm slices
      [0.64, -0.04].forEach(ly => {
        const ln = new THREE.Mesh(
          new THREE.BoxGeometry(6.8, 0.055, 2.05),
          new THREE.MeshPhongMaterial({ color: 0x4466bb, transparent: true, opacity: 0.42 })
        );
        ln.position.y = ly;
        grp.add(ln);
      });

      // Residual bypass cylinder
      const byp = new THREE.Mesh(
        new THREE.CylinderGeometry(0.045, 0.045, 1.18, 8),
        new THREE.MeshPhongMaterial({ color: 0x38bdf8, emissive: 0x0a2233, emissiveIntensity: 1 })
      );
      byp.position.set(3.5, 0, 0);
      grp.add(byp);

      // Block label
      const bLab = spr(cfg.label, "#0d1e35", 1.05, 0.26);
      bLab.position.set(-2.95, 0.8, 0);
      grp.add(bLab);

      grp.position.y = cfg.y;
      scene.add(grp);
      blockHighlights.push({ attnMesh: attnSlab, ffnMesh: ffnSlab, idx: bi });
    });

    // ── Projection / unembedding box ──
    const projBox = new THREE.Mesh(
      new THREE.BoxGeometry(6.0, 0.44, 1.7),
      new THREE.MeshPhongMaterial({ color: 0xf472b6, emissive: 0x000000, transparent: true, opacity: 0.72 })
    );
    projBox.position.y = 5.4;
    scene.add(projBox);
    const projEdge = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(6.02, 0.46, 1.72)),
      new THREE.LineBasicMaterial({ color: 0xf472b6, transparent: true, opacity: 0.55 })
    );
    projEdge.position.y = 5.4;
    scene.add(projEdge);
    const projSp = spr("Unembedding  W_U  (768→50257)", "#f472b6", 3.2, 0.27);
    projSp.position.set(0, 5.4, 1.1);
    scene.add(projSp);

    // ── Output sphere + rings ──
    const outSph = new THREE.Mesh(
      new THREE.SphereGeometry(0.62, 28, 28),
      new THREE.MeshPhongMaterial({ color: 0x10b981, emissive: 0x000000 })
    );
    outSph.position.y = 6.8;
    scene.add(outSph);
    const outRing1 = new THREE.Mesh(
      new THREE.TorusGeometry(0.98, 0.028, 8, 56),
      new THREE.MeshPhongMaterial({ color: 0x10b981, emissive: 0x042a1e })
    );
    outRing1.position.y = 6.8;
    scene.add(outRing1);
    const outRing2 = new THREE.Mesh(
      new THREE.TorusGeometry(1.22, 0.016, 8, 56),
      new THREE.MeshPhongMaterial({ color: 0x10b981, emissive: 0x021510, transparent: true, opacity: 0.6 })
    );
    outRing2.position.y = 6.8;
    outRing2.rotation.x = Math.PI / 3;
    scene.add(outRing2);

    // ── Particle stream ──
    const N = 480;
    const pPos = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      pPos[i*3]   = (Math.random() - 0.5) * 1.6;
      pPos[i*3+1] = (Math.random()) * 16 - 8;
      pPos[i*3+2] = (Math.random() - 0.5) * 1.6;
    }
    const pGeo = new THREE.BufferGeometry();
    pGeo.setAttribute("position", new THREE.BufferAttribute(pPos, 3));
    const pMat = new THREE.PointsMaterial({ color: 0x38bdf8, size: 0.058, transparent: true, opacity: 0.96 });
    scene.add(new THREE.Points(pGeo, pMat));

    // ── Detail groups ──
    const detailGroups = {};

    const tokenDetail = buildEmbedDetail(-5.2);
    tokenDetail.visible = false; scene.add(tokenDetail); detailGroups["token"] = tokenDetail;
    const inputDetail = buildEmbedDetail(-5.2);
    inputDetail.visible = false; scene.add(inputDetail); detailGroups["input"] = inputDetail;
    const embedDetail = buildEmbedDetail(-3.2);
    embedDetail.visible = false; scene.add(embedDetail); detailGroups["embed"] = embedDetail;

    BY.forEach((by, bi) => {
      const aKey = ["b0-attn","b1-attn","b2-attn"][bi];
      const fKey = ["b0-ffn","b1-ffn","b2-ffn"][bi];
      const aGrp = buildAttnDetail(by);
      aGrp.visible = false; scene.add(aGrp); detailGroups[aKey] = aGrp;
      const fGrp = buildFFNDetail(by);
      fGrp.visible = false; scene.add(fGrp); detailGroups[fKey] = fGrp;
    });

    const outDetail = buildOutputDetail(5.5);
    outDetail.visible = false; scene.add(outDetail);
    detailGroups["proj"]   = outDetail;
    detailGroups["output"] = outDetail;

    // Camera lerp state
    const camTarget = { p: new THREE.Vector3(...OV.p), t: new THREE.Vector3(...OV.t), fov: OV.fov };
    const camPos  = new THREE.Vector3(...OV.p);
    const camLook = new THREE.Vector3(...OV.t);

    // Mouse controls
    let isDrag = false, pMX = 0, pMY = 0;
    const onMD = e => { isDrag = true; pMX = e.clientX; pMY = e.clientY; };
    const onMU = () => { isDrag = false; };
    const onMM = e => {
      if (!isDrag) return;
      rotRef.current.y += (e.clientX - pMX) * 0.004;
      rotRef.current.x = Math.max(-0.6, Math.min(0.6, rotRef.current.x + (e.clientY - pMY) * 0.004));
      pMX = e.clientX; pMY = e.clientY;
    };
    const onWheel = e => {
      camTarget.p.z = Math.max(4, Math.min(24, camTarget.p.z + e.deltaY * 0.014));
    };
    el.addEventListener("mousedown", onMD);
    window.addEventListener("mouseup", onMU);
    window.addEventListener("mousemove", onMM);
    el.addEventListener("wheel", onWheel);

    refs.current = { camTarget, detailGroups, blockHighlights, pGeo, pPos, pMat, outSph, outRing1, outRing2, projBox, pl1 };

    // Animation loop
    let t = 0, animId;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      t += 0.012;

      if (!isDrag) rotRef.current.y += 0.0022;
      scene.rotation.y = rotRef.current.y;
      scene.rotation.x = rotRef.current.x;

      // Camera lerp
      camPos.lerp(camTarget.p, 0.065);
      camLook.lerp(camTarget.t, 0.065);
      camera.position.copy(camPos);
      camera.lookAt(camLook);
      if (Math.abs(camera.fov - camTarget.fov) > 0.05) {
        camera.fov += (camTarget.fov - camera.fov) * 0.075;
        camera.updateProjectionMatrix();
      }

      // Token cubes animate
      tokCubes.forEach((cube, i) => {
        cube.rotation.y += 0.011;
        cube.rotation.x += 0.007;
        cube.position.y = -6.6 + Math.sin(t * 1.3 + i * 0.9) * 0.07;
      });

      // Rings spin
      outRing1.rotation.z += 0.016;
      outRing2.rotation.y += 0.022;

      // Particles
      const cs = STEPS[stepRef.current];
      const ptY = cs ? cs.cam.t[1] : 0;
      for (let i = 0; i < N; i++) {
        pPos[i*3+1] += 0.038;
        if (pPos[i*3+1] > ptY + 0.8) {
          pPos[i*3+1] = ptY - 4.0 - Math.random() * 1.8;
          pPos[i*3]   = (Math.random() - 0.5) * 0.85;
          pPos[i*3+2] = (Math.random() - 0.5) * 0.85;
        }
        pPos[i*3]   *= 0.998;
        pPos[i*3+2] *= 0.998;
      }
      pGeo.attributes.position.needsUpdate = true;

      // Output glow pulse
      if (stepRef.current === 10) {
        outSph.material.emissive.set(STEPS[10].color);
        outSph.material.emissiveIntensity = 0.22 + Math.abs(Math.sin(t * 2.2)) * 0.65;
        outRing1.material.emissiveIntensity = 0.3 + Math.abs(Math.sin(t * 2.2 + 0.5)) * 0.5;
      } else {
        outSph.material.emissive.set("#000000");
        outSph.material.emissiveIntensity = 0.04;
      }
      projBox.material.emissive.set(stepRef.current === 9 ? STEPS[9].color : "#000000");
      projBox.material.emissiveIntensity = stepRef.current === 9 ? 0.3 + Math.abs(Math.sin(t * 3)) * 0.3 : 0;

      pl1.intensity = 5 + Math.sin(t * 0.75) * 1.0;
      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      const w = el.clientWidth, h = el.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", onResize);

    // Init: zoom into first step
    setTimeout(() => {
      const s = STEPS[0];
      camTarget.p.set(...s.cam.p);
      camTarget.t.set(...s.cam.t);
      camTarget.fov = s.cam.fov;
      Object.entries(detailGroups).forEach(([z, grp]) => { grp.visible = z === s.zone; });
    }, 80);

    return () => {
      cancelAnimationFrame(animId);
      el.removeEventListener("mousedown", onMD);
      window.removeEventListener("mouseup", onMU);
      window.removeEventListener("mousemove", onMM);
      el.removeEventListener("wheel", onWheel);
      window.removeEventListener("resize", onResize);
      renderer.dispose();
      if (el.contains(renderer.domElement)) el.removeChild(renderer.domElement);
    };
  }, []);

  const setAndSync = n => { setStep(n); stepRef.current = n; };
  const goOverview = () => {
    const r = refs.current;
    if (!r.camTarget) return;
    r.camTarget.p.set(...OV.p);
    r.camTarget.t.set(...OV.t);
    r.camTarget.fov = OV.fov;
    if (r.detailGroups) Object.values(r.detailGroups).forEach(g => { g.visible = false; });
    setIsZoomed(false);
  };

  const s = STEPS[step];

  return (
    <div style={{ display:"flex", height:"100vh", width:"100vw", overflow:"hidden",
      background:"#020510", fontFamily:"'Courier New', 'Lucida Console', monospace", color:"#c8d8f0" }}>

      {/* ── LEFT PANEL ── */}
      <div style={{ width:"34%", display:"flex", flexDirection:"column",
        borderRight:"1px solid #090f20",
        background:"linear-gradient(180deg,#03061a 0%,#020410 100%)" }}>

        <div style={{ padding:"14px 16px 10px", borderBottom:"1px solid #090f20" }}>
          <div style={{ fontSize:"9px", letterSpacing:"4px", color:"#38bdf8", fontWeight:"bold" }}>GPT · FORWARD PASS</div>
          <div style={{ fontSize:"8px", color:"#0a1428", letterSpacing:"1px", marginBottom:"8px" }}>CLICK STEP → ZOOM IN · DRAG · SCROLL</div>
          <div style={{ padding:"7px 9px", background:"#040818", border:"1px solid #090f20", borderRadius:"4px", fontSize:"10px" }}>
            <span style={{ color:"#1a2a4a" }}>query: </span>
            <span style={{ color:"#a78bfa" }}>"where is taj mahal"</span>
          </div>
        </div>

        <div style={{ flex:1, overflowY:"auto", padding:"8px 10px" }}>
          {STEPS.map((st, i) => {
            const isA = i === step, isP = i < step;
            return (
              <div key={st.id} onClick={() => setAndSync(i)}
                style={{ display:"flex", gap:"10px", alignItems:"flex-start",
                  padding:"7px 9px", marginBottom:"3px", borderRadius:"5px", cursor:"pointer",
                  border:`1px solid ${isA ? st.color + "55" : "#060c1c"}`,
                  background: isA ? `${st.color}0e` : isP ? "#030810" : "#020710",
                  transform: isA ? "translateX(5px)" : "none",
                  transition:"all 0.2s ease", position:"relative" }}>
                {i < STEPS.length - 1 && (
                  <div style={{ position:"absolute", left:"17px", top:"25px", bottom:"-12px",
                    width:"1px", background: isP ? `${STEPS[i].color}44` : "#080e1e" }} />
                )}
                <div style={{ flexShrink:0, paddingTop:"2px" }}>
                  <div style={{ width:"8px", height:"8px", borderRadius:"50%",
                    background: isA ? st.color : isP ? st.color+"55" : "#090f1e",
                    boxShadow: isA ? `0 0 10px ${st.color},0 0 20px ${st.color}44` : "none",
                    border:`1px solid ${isA ? st.color : isP ? st.color+"44" : "#121c2e"}`,
                    transition:"all 0.2s" }} />
                </div>
                <div style={{ flex:1, minWidth:0 }}>
                  <div style={{ fontSize:"9px", fontWeight: isA ? "700" : "400",
                    color: isA ? st.color : isP ? "#1a2c50" : "#101828",
                    letterSpacing: isA ? "0.4px" : "0", transition:"all 0.2s" }}>
                    {st.label}
                  </div>
                  {isA && (
                    <div style={{ marginTop:"3px", fontSize:"8px", color:"#2a4060", lineHeight:"1.65" }}>
                      {st.detail}
                    </div>
                  )}
                </div>
                <div style={{ flexShrink:0, fontSize:"7px", paddingTop:"2px",
                  color: isA ? st.color + "88" : "#080e1c" }}>
                  {String(i).padStart(2,"0")}
                </div>
              </div>
            );
          })}
        </div>

        <div style={{ padding:"10px 12px", borderTop:"1px solid #090f20" }}>
          <div style={{ display:"flex", gap:"6px", marginBottom:"6px" }}>
            <button onClick={() => setAndSync(Math.max(0, step - 1))}
              style={{ flex:1, padding:"7px", cursor:"pointer", background:"#030818",
                border:"1px solid #090f20", color:"#38bdf8", borderRadius:"4px", fontSize:"10px" }}>◀</button>
            <button onClick={() => setPlaying(p => !p)}
              style={{ flex:1.4, padding:"7px", cursor:"pointer",
                background: playing ? "#10b981" : "#030818",
                border:`1px solid ${playing ? "#10b981" : "#090f20"}`,
                color: playing ? "#001a0e" : "#556677",
                borderRadius:"4px", fontSize:"10px", fontWeight:"700",
                boxShadow: playing ? "0 0 14px #10b98166" : "none" }}>
              {playing ? "⏸ PAUSE" : "▶ PLAY"}
            </button>
            <button onClick={() => setAndSync(Math.min(STEPS.length - 1, step + 1))}
              style={{ flex:1, padding:"7px", cursor:"pointer", background:"#030818",
                border:"1px solid #090f20", color:"#38bdf8", borderRadius:"4px", fontSize:"10px" }}>▶</button>
          </div>
          <button onClick={goOverview}
            style={{ width:"100%", padding:"6px", cursor:"pointer",
              background: isZoomed ? "#07101e" : "#030818",
              border:`1px solid ${isZoomed ? "#1a3050" : "#090f20"}`,
              color: isZoomed ? "#38bdf8" : "#1a2a40",
              borderRadius:"4px", fontSize:"9px", letterSpacing:"1px",
              boxShadow: isZoomed ? "0 0 8px #38bdf822" : "none" }}>
            ⊙ OVERVIEW — ZOOM OUT
          </button>

          {/* Training note */}
          <div style={{ marginTop:"7px", padding:"7px 8px", background:"#030610",
            border:"1px solid #090f20", borderRadius:"4px" }}>
            <div style={{ color:"#f59e0b", fontSize:"7px", fontWeight:"bold", letterSpacing:"1px", marginBottom:"3px" }}>
              ⚡ INFERENCE  ≠  TRAINING
            </div>
            <div style={{ color:"#0e1c30", fontSize:"7.5px", lineHeight:"1.7" }}>
              This is <span style={{ color:"#38bdf8" }}>forward pass only</span> — no weight updates.<br/>
              Gradient descent: <span style={{ color:"#c084fc" }}>∂L/∂W → W ← W − η∇W</span><br/>
              happens only during training via backprop.
            </div>
          </div>
        </div>
      </div>

      {/* ── RIGHT PANEL — Three.js ── */}
      <div style={{ flex:1, position:"relative", overflow:"hidden" }}>
        <div ref={mountRef} style={{ width:"100%", height:"100%" }} />

        {/* Active step HUD */}
        <div style={{ position:"absolute", top:"14px", left:"14px",
          background:"#020510dd", backdropFilter:"blur(10px)",
          border:`1px solid ${s.color}44`, borderRadius:"6px", padding:"9px 13px", maxWidth:"240px" }}>
          <div style={{ fontSize:"7px", color:s.color, letterSpacing:"2px", fontWeight:"bold", marginBottom:"3px" }}>
            ACTIVE LAYER
          </div>
          <div style={{ fontSize:"10px", color:"#8aaccc", marginBottom:"5px" }}>{s.label}</div>
          <div style={{ width:`${(step+1)/STEPS.length*100}%`, height:"2px",
            background:s.color, borderRadius:"1px", boxShadow:`0 0 6px ${s.color}`,
            transition:"width 0.35s ease, background 0.3s" }} />
        </div>

        {/* Hint badge */}
        <div style={{ position:"absolute", top:"14px", right:"14px", textAlign:"right" }}>
          <div style={{ color:"#060c18", fontSize:"8px", letterSpacing:"1px" }}>GPT-2 STYLE · 12 LAYERS</div>
          <div style={{ color:"#040810", fontSize:"7px" }}>drag · scroll · click steps</div>
        </div>

        {/* Legend */}
        <div style={{ position:"absolute", bottom:"12px", left:"12px",
          display:"flex", gap:"10px", flexWrap:"wrap",
          background:"#020510bb", padding:"5px 9px", borderRadius:"4px" }}>
          {[["Q/K/V","#38bdf8"],["MHA","#fbbf24"],["FFN","#fb923c"],["LN","#4466bb"],["Residual","#38bdf8"],["Output","#10b981"]].map(([l,c])=>(
            <div key={l} style={{ display:"flex", alignItems:"center", gap:"4px" }}>
              <div style={{ width:"6px", height:"6px", background:c, borderRadius:"1px" }} />
              <span style={{ color:"#0a1420", fontSize:"7px" }}>{l}</span>
            </div>
          ))}
        </div>

        {/* Step dots */}
        <div style={{ position:"absolute", bottom:"12px", right:"14px", display:"flex", gap:"4px" }}>
          {STEPS.map((_, i) => (
            <div key={i} onClick={() => setAndSync(i)} style={{
              width: i === step ? "16px" : "5px", height:"5px", borderRadius:"3px",
              background: i === step ? s.color : i < step ? "#0e1828" : "#070c16",
              cursor:"pointer", boxShadow: i === step ? `0 0 8px ${s.color}` : "none",
              transition:"all 0.25s" }} />
          ))}
        </div>
      </div>
    </div>
  );
}
