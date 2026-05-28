const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const { FaMicrochip, FaServer, FaBolt, FaLock, FaChartLine, FaRocket, FaCheckCircle, FaCubes, FaGlobe } = require("react-icons/fa");

// ═══════════════════════════════════════════════════════════════════════════
// Design System
// ═══════════════════════════════════════════════════════════════════════════
const C = {
  bg:       "0A0F1F",
  cardBg:   "141B2D",
  cardBg2:  "1A2340",
  accent:   "00D4FF",
  accent2:  "00FF88",
  warn:     "FF6B6B",
  gold:     "F0C040",
  text:     "E8EDF5",
  textSec:  "8899AA",
  border:   "1E2D4A",
  white:    "FFFFFF",
};

const FONT_H = "Arial";
const FONT_B = "Arial";

// ═══════════════════════════════════════════════════════════════════════════
// Icon Helper
// ═══════════════════════════════════════════════════════════════════════════
function renderIconSvg(IconComponent, color, size = 256) {
  return ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color, size: String(size) })
  );
}

async function iconToBase64(IconComponent, color, size = 256) {
  const svg = renderIconSvg(IconComponent, color, size);
  const pngBuffer = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + pngBuffer.toString("base64");
}

// ═══════════════════════════════════════════════════════════════════════════
// Reusable Helpers
// ═══════════════════════════════════════════════════════════════════════════
function addFooter(slide, text) {
  slide.addText(text, { x: 0.5, y: 5.15, w: 9, h: 0.35, fontSize: 8, color: C.textSec, fontFace: FONT_B, align: "left" });
}

function addSlideNum(slide, num) {
  slide.addText(String(num).padStart(2, "0"), { x: 9.2, y: 5.05, w: 0.5, h: 0.4, fontSize: 10, color: C.accent, fontFace: FONT_H, align: "right", bold: true });
}

let _pres = null;

function sectionTitle(slide, title, subtitle) {
  slide.addText(title, { x: 0.7, y: 0.3, w: 8.6, h: 0.6, fontSize: 30, color: C.white, fontFace: FONT_H, bold: true, margin: 0 });
  slide.addShape(_pres.shapes.RECTANGLE, { x: 0.7, y: 0.95, w: 0.8, h: 0.04, fill: { color: C.accent } });
  if (subtitle) {
    slide.addText(subtitle, { x: 0.7, y: 1.1, w: 8.6, h: 0.35, fontSize: 12, color: C.textSec, fontFace: FONT_B, margin: 0 });
  }
}

function card(slide, x, y, w, h, opts = {}) {
  slide.addShape(_pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: opts.fill || C.cardBg }, line: { color: opts.border || C.border, width: 0.5 } });
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════
async function main() {
  const pres = new pptxgen();
  _pres = pres;
  pres.layout = "LAYOUT_16x9";
  pres.author = "FPGA LPU Team";
  pres.title = "FPGA 大模型推理集群 — 投资人 BP";

  // Pre-render icons
  const icons = {};
  const iconList = [
    ["chip", FaMicrochip, "#00D4FF"],
    ["server", FaServer, "#00D4FF"],
    ["bolt", FaBolt, "#00FF88"],
    ["lock", FaLock, "#00D4FF"],
    ["chart", FaChartLine, "#00FF88"],
    ["rocket", FaRocket, "#F0C040"],
    ["check", FaCheckCircle, "#00FF88"],
    ["cubes", FaCubes, "#00D4FF"],
    ["globe", FaGlobe, "#00D4FF"],
  ];
  for (const [name, Icon, color] of iconList) {
    icons[name] = await iconToBase64(Icon, color);
  }

  // ═══════════════════════════════════════════════════════════════════════
  // SLIDE 1: TITLE
  // ═══════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };

    // Decorative top-right glow
    s.addShape(_pres.shapes.RECTANGLE, { x: 6, y: -1, w: 5, h: 3, fill: { color: C.accent, transparency: 92 }, rotate: 15 });

    // Icon
    s.addImage({ data: icons.chip, x: 0.8, y: 0.8, w: 0.6, h: 0.6 });

    // Title
    s.addText("FPGA 大模型推理集群", { x: 0.8, y: 1.6, w: 8.4, h: 0.8, fontSize: 40, color: C.white, fontFace: FONT_H, bold: true, margin: 0 });
    s.addText("DeepSeek V4 Pro Decode 架构方案", { x: 0.8, y: 2.4, w: 8.4, h: 0.5, fontSize: 20, color: C.accent, fontFace: FONT_H, margin: 0 });
    s.addText(`面向 Agent 时代的流式计算范式 | 不是 "更便宜的 GPU"，是架构升级`, { x: 0.8, y: 2.95, w: 8.4, h: 0.4, fontSize: 13, color: C.textSec, fontFace: FONT_B, margin: 0 });

    // Bottom bar
    s.addShape(_pres.shapes.RECTANGLE, { x: 0, y: 5.3, w: 10, h: 0.325, fill: { color: C.cardBg } });
    s.addText("CONFIDENTIAL  |  v1.3  |  May 2026", { x: 0.8, y: 5.3, w: 8.4, h: 0.325, fontSize: 8, color: C.textSec, fontFace: FONT_B, valign: "middle", margin: 0 });
  }

  // ═══════════════════════════════════════════════════════════════════════
  // SLIDE 2: THE PROBLEM
  // ═══════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    sectionTitle(s, "GPU 供应困局：结构性瓶颈", "中国大模型推理部署面临的三重约束");
    addSlideNum(s, 2);
    addFooter(s, "FPGA LPU  |  Investor Briefing");

    const cols = [
      { x: 0.7, title: "H100 / B200", sub: "NVIDIA", status: "禁运", color: C.warn, desc: "美国出口管制 3A090\n高端型号对中国 zero allocation\nCUDA 生态最强但不可获取" },
      { x: 3.8, title: "Ascend 950PR", sub: "华为", status: "排队 12 月", color: C.gold, desc: "SMIC 7nm 产能受限\nCoWoS 先进封装制裁\n标价 ¥5万/卡 → 实际 ¥25万 (5×)" },
      { x: 6.9, title: "国产 GPU", sub: "寒武纪/海光/壁仞", status: "生态欠成熟", color: C.textSec, desc: "供应量有限\n软件栈不完善\nfp4 无原生支持" },
    ];

    for (const col of cols) {
      card(s, col.x, 1.7, 2.7, 3.15);
      s.addShape(_pres.shapes.RECTANGLE, { x: col.x, y: 1.7, w: 2.7, h: 0.06, fill: { color: col.color } });
      s.addText(col.title, { x: col.x + 0.2, y: 1.9, w: 2.3, h: 0.35, fontSize: 16, color: C.white, fontFace: FONT_H, bold: true, margin: 0 });
      s.addText(col.sub, { x: col.x + 0.2, y: 2.25, w: 2.3, h: 0.25, fontSize: 10, color: C.textSec, fontFace: FONT_B, margin: 0 });
      s.addShape(_pres.shapes.RECTANGLE, { x: col.x + 0.2, y: 2.7, w: 2.3, h: 0.03, fill: { color: C.border } });
      s.addText(col.status, { x: col.x + 0.2, y: 2.85, w: 2.3, h: 0.25, fontSize: 10, color: col.color, fontFace: FONT_B, bold: true, margin: 0 });
      s.addText(col.desc, { x: col.x + 0.2, y: 3.15, w: 2.3, h: 1.5, fontSize: 10, color: C.textSec, fontFace: FONT_B, margin: 0, lineSpacingMultiple: 1.5 });
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  // SLIDE 3: OUR SOLUTION
  // ═══════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    sectionTitle(s, "我们的方案：32 芯片 FPGA 推理集群", "Agilex 7 M × 32 片 = 8 卡 × 4 芯片/卡");
    addSlideNum(s, 3);
    addFooter(s, "FPGA LPU  |  Investor Briefing");

    // Left: architecture concept
    card(s, 0.7, 1.7, 4.3, 3.15);
    s.addImage({ data: icons.cubes, x: 1.0, y: 1.95, w: 0.45, h: 0.45 });
    s.addText("架构核心", { x: 1.6, y: 1.95, w: 3.2, h: 0.45, fontSize: 16, color: C.white, fontFace: FONT_H, bold: true, valign: "middle", margin: 0 });

    const archPoints = [
      "32 芯片 Pipeline-Parallel：每芯片仅承载 2 层",
      "fp4 E2M1 原生计算：DSP 直接 MAC，零解压",
      "权重常驻 SRAM：消除 81.6% 的 HBM 读取",
      "MLA 注意力硬件加速：KV 压缩 114×",
      "MoE All-to-All：SERDES 直连，无网络栈开销",
      "聚合 HBM 带宽 29.4 TB/s，带宽/层 460 GB/s",
    ];
    s.addText(archPoints.map((pt, i) => ({ text: pt, options: { bullet: true, breakLine: i < archPoints.length - 1, fontSize: 11, color: C.text, fontFace: FONT_B, paraSpaceAfter: 6 } })), { x: 1.0, y: 2.55, w: 3.7, h: 2.1, margin: 0, valign: "top" });

    // Right: key metrics
    const metrics = [
      { num: "~14,000", label: "聚合 Decode 吞吐 (tok/s, 高并发)", color: C.accent2 },
      { num: "~660", label: "B=1 单 Session (tok/s)", color: C.accent },
      { num: "29.4 TB/s", label: "聚合 HBM 带宽 (32 片)", color: C.accent },
      { num: "¥133万", label: "系统 BOM (8 卡 / 32 芯片)", color: C.gold },
      { num: "8-12 周", label: "交期 (vs 950PR >6 月)", color: C.accent2 },
    ];

    metrics.forEach((m, i) => {
      const y = 1.7 + i * 0.63;
      card(s, 5.3, y, 4.1, 0.55);
      s.addText(m.num, { x: 5.5, y, w: 1.8, h: 0.55, fontSize: 19, color: m.color, fontFace: FONT_H, bold: true, valign: "middle", margin: 0 });
      s.addText(m.label, { x: 7.3, y, w: 1.9, h: 0.55, fontSize: 10, color: C.textSec, fontFace: FONT_B, valign: "middle", margin: 0 });
    });
  }

  // ═══════════════════════════════════════════════════════════════════════
  // SLIDE 4: NOT CHEAPER GPU — DIFFERENT PARADIGM
  // ═══════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    sectionTitle(s, `不是"更便宜的 GPU"，是不同的计算范式`, `当三个维度同时出现 10-1000× 差距时，你不是在比较替代方案——你在看两种计算范式`);
    addSlideNum(s, 4);
    addFooter(s, "FPGA LPU  |  Investor Briefing");

    // Two-column comparison
    // GPU column
    card(s, 0.7, 1.7, 4.3, 3.15);
    s.addShape(_pres.shapes.RECTANGLE, { x: 0.7, y: 1.7, w: 4.3, h: 0.06, fill: { color: C.warn } });
    s.addText("GPU SIMT 批处理模型", { x: 1.0, y: 1.9, w: 3.7, h: 0.4, fontSize: 15, color: C.warn, fontFace: FONT_H, bold: true, margin: 0 });
    const gpuIssues = [
      "B=1 时 Tensor Core 利用率仅 2-3%",
      "需要 batch 填充 warp slot 隐藏延迟",
      "Prefill ↔ Decode 切换 65-265μs (kernel launch + context switch)",
      "KV cache 地址需软件 page table 查表 (350-900ns)",
      "HBM 带宽 98% 时间在等 warp 调度，非等数据",
    ];
    s.addText(gpuIssues.map((pt, i) => ({ text: pt, options: { bullet: true, breakLine: i < gpuIssues.length - 1, fontSize: 11, color: C.textSec, fontFace: FONT_B, paraSpaceAfter: 8 } })), { x: 1.0, y: 2.45, w: 3.7, h: 2.2, margin: 0, valign: "top" });

    // FPGA column
    card(s, 5.3, 1.7, 4.3, 3.15);
    s.addShape(_pres.shapes.RECTANGLE, { x: 5.3, y: 1.7, w: 4.3, h: 0.06, fill: { color: C.accent2 } });
    s.addText("FPGA 流式权重常驻模型", { x: 5.6, y: 1.9, w: 3.7, h: 0.4, fontSize: 15, color: C.accent2, fontFace: FONT_H, bold: true, margin: 0 });
    const fpgaAdv = [
      "权重常驻 SRAM，HBM 只读激活 → ~38% 有效带宽",
      "Pipeline 填入后每个 cycle 都有输出 → 无需 batch",
      "DSP 重配置 <1μs → Prefill/Decode 切换 ~150ns",
      "硬件地址生成器 → KV 地址解析 <10ns",
      "29.4 TB/s 聚合 HBM → 有效带宽 ~83× at B=1",
    ];
    s.addText(fpgaAdv.map((pt, i) => ({ text: pt, options: { bullet: true, breakLine: i < fpgaAdv.length - 1, fontSize: 11, color: C.text, fontFace: FONT_B, paraSpaceAfter: 8 } })), { x: 5.6, y: 2.45, w: 3.7, h: 2.2, margin: 0, valign: "top" });
  }

  // ═══════════════════════════════════════════════════════════════════════
  // SLIDE 5: THREE 10-1000× GAPS
  // ═══════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    sectionTitle(s, "三个数量级差距", "当多个维度的差距同时放大，这就是范式替代，不是技术选型");
    addSlideNum(s, 5);
    addFooter(s, "FPGA LPU  |  Investor Briefing");

    const dims = [
      { num: "83×", label: "有效带宽利用率", sub: "B=1 Decode vs H100", detail: "FPGA 流式架构有效带宽 ~700 GB/s/layer\nGPU SIMT 有效带宽 ~84 GB/s (2.5% util)" },
      { num: "1,000×", label: "Prefill/Decode 切换延迟", sub: "150ns vs 65-265μs", detail: "FPGA DSP 重配置 <1μs\nGPU kernel launch + context switch" },
      { num: "1,000×", label: "KV Cache 地址解析", sub: "<10ns vs 350-900ns", detail: "FPGA 硬件地址生成器直接算物理地址\nGPU vLLM PagedAttention 软件查表" },
    ];

    dims.forEach((d, i) => {
      const x = 0.7 + i * 3.1;
      card(s, x, 1.7, 2.8, 3.15);
      s.addShape(_pres.shapes.RECTANGLE, { x, y: 1.7, w: 2.8, h: 0.05, fill: { color: C.accent } });
      s.addText(d.num, { x: x + 0.2, y: 1.95, w: 2.4, h: 0.65, fontSize: 42, color: C.accent, fontFace: FONT_H, bold: true, margin: 0 });
      s.addText(d.label, { x: x + 0.2, y: 2.6, w: 2.4, h: 0.3, fontSize: 14, color: C.white, fontFace: FONT_H, bold: true, margin: 0 });
      s.addText(d.sub, { x: x + 0.2, y: 2.9, w: 2.4, h: 0.25, fontSize: 9, color: C.accent2, fontFace: FONT_B, margin: 0 });
      s.addShape(_pres.shapes.RECTANGLE, { x: x + 0.2, y: 3.25, w: 2.4, h: 0.02, fill: { color: C.border } });
      s.addText(d.detail, { x: x + 0.2, y: 3.4, w: 2.4, h: 1.2, fontSize: 10, color: C.textSec, fontFace: FONT_B, margin: 0, lineSpacingMultiple: 1.5 });
    });
  }

  // ═══════════════════════════════════════════════════════════════════════
  // SLIDE 6: AGENT ERA — KILLER APP
  // ═══════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    sectionTitle(s, "Agent 时代：FPGA 的杀手场景", "Coding Agent 每轮 3-5 次 Prefill/Decode 交替 → GPU 最弱场景 = FPGA 最强场景");
    addSlideNum(s, 6);
    addFooter(s, "FPGA LPU  |  Investor Briefing");

    // Left: why agent favors FPGA
    card(s, 0.7, 1.7, 4.3, 3.15);
    s.addImage({ data: icons.bolt, x: 1.0, y: 1.95, w: 0.4, h: 0.4 });
    s.addText("Agent 场景放大了 FPGA 的架构优势", { x: 1.55, y: 1.95, w: 3.2, h: 0.4, fontSize: 15, color: C.accent2, fontFace: FONT_H, bold: true, valign: "middle", margin: 0 });

    const agentPoints = [
      "高频 Prefill/Decode 交替 (3-5× vs chatbot)",
      "  → FPGA 切换 150ns, GPU 切换 65-265μs → 千倍差距",
      "B=1 是 Agent 的天然 batch size",
      "  → GPU SIMT 利用率 2-3%, FPGA 流式 ~38% → 83×",
      "KV cache 80-90% 跨轮稳定 (warm start)",
      "  → 硬件 KV 地址直接映射，确定性延迟",
      "代码不出企业网络 — 物理隔离，合规硬需求",
      `  → 金融/军工/外包的刚性需求，不是"nice to have"`,
      "Per-token 1.4ms 确定性延迟",
      "  → 开发体验流畅，无 GPU 排队抖动",
    ];
    s.addText(agentPoints.map((pt, i) => {
      const isHead = i % 2 === 0;
      return { text: pt, options: { bullet: !isHead, breakLine: true, fontSize: isHead ? 11 : 9, color: isHead ? C.white : C.textSec, fontFace: FONT_B, bold: isHead, paraSpaceAfter: isHead ? 2 : 4, indentLevel: isHead ? 0 : 1 } };
    }), { x: 1.0, y: 2.5, w: 3.7, h: 2.2, margin: 0, valign: "top" });

    // Right: TAM + product
    card(s, 5.3, 1.7, 4.1, 1.45);
    s.addImage({ data: icons.server, x: 5.6, y: 1.9, w: 0.35, h: 0.35 });
    s.addText("AI IDE 盒子 — 产品形态", { x: 6.1, y: 1.9, w: 3.1, h: 0.35, fontSize: 13, color: C.white, fontFace: FONT_H, bold: true, valign: "middle", margin: 0 });
    s.addText("32 芯片 HBM-Only, 4U 机箱\n30-60 concurrent coding agent sessions\n私有部署, 代码永不出企业", { x: 5.6, y: 2.35, w: 3.5, h: 0.7, fontSize: 11, color: C.textSec, fontFace: FONT_B, margin: 0, lineSpacingMultiple: 1.5 });

    card(s, 5.3, 3.35, 4.1, 1.5);
    s.addImage({ data: icons.chart, x: 5.6, y: 3.55, w: 0.35, h: 0.35 });
    s.addText("中国市场 TAM 估算", { x: 6.1, y: 3.55, w: 3.1, h: 0.35, fontSize: 13, color: C.white, fontFace: FONT_H, bold: true, valign: "middle", margin: 0 });

    const tamData = [
      ["10 万 developer × 30% AI agent 渗透率", "= 3 万并发"],
      ["÷ 30-60 session/套", "≈ 1,000 套"],
      ["× ¥133万 BOM/套", "≈ ¥13.3 亿 (仅 coding agent)"],
      ["+ 通用 agent / RAG / 客服", "→ 叠加市场更大"],
    ];
    tamData.forEach((row, i) => {
      s.addText(row[0], { x: 5.6, y: 4.05 + i * 0.2, w: 2.8, h: 0.18, fontSize: 9, color: C.textSec, fontFace: FONT_B, margin: 0 });
      s.addText(row[1], { x: 8.5, y: 4.05 + i * 0.2, w: 0.7, h: 0.18, fontSize: 9, color: C.accent2, fontFace: FONT_B, bold: true, align: "right", margin: 0 });
    });
  }

  // ═══════════════════════════════════════════════════════════════════════
  // SLIDE 7: COMPETITIVE LANDSCAPE
  // ═══════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    sectionTitle(s, "竞争格局", `FPGA 与主流方案的定位不是"替代"，是"不同范式 + 不同场景"`);
    addSlideNum(s, 7);
    addFooter(s, "FPGA LPU  |  Investor Briefing");

    const headerOpts = { fill: { color: C.cardBg2 }, color: C.accent, bold: true, fontSize: 11, fontFace: FONT_H, align: "center", valign: "middle" };
    const cellOpts = { fill: { color: C.cardBg }, color: C.text, fontSize: 10, fontFace: FONT_B, align: "center", valign: "middle" };
    const greenOpts = { fill: { color: C.cardBg }, color: C.accent2, fontSize: 10, fontFace: FONT_B, bold: true, align: "center", valign: "middle" };
    const redOpts = { fill: { color: C.cardBg }, color: C.warn, fontSize: 10, fontFace: FONT_B, align: "center", valign: "middle" };
    const yellowOpts = { fill: { color: C.cardBg }, color: C.gold, fontSize: 10, fontFace: FONT_B, align: "center", valign: "middle" };

    const rows = [
      [ { text: "", options: headerOpts }, { text: "H100/H200", options: headerOpts }, { text: "Ascend 950PR", options: headerOpts }, { text: "FPGA (本方案)", options: { ...headerOpts, color: C.accent2 } }, { text: "ASIC (Phase 2)", options: { ...headerOpts, color: C.accent2 } } ],
      [ { text: "可获取性", options: { ...cellOpts, bold: true, align: "left" } }, { text: "管制禁售", options: redOpts }, { text: "排队 6-18 月", options: yellowOpts }, { text: "8-12 周交期", options: greenOpts }, { text: "自主可控", options: greenOpts } ],
      [ { text: "全球部署", options: { ...cellOpts, bold: true, align: "left" } }, { text: "部分受限", options: yellowOpts }, { text: "华为受限", options: redOpts }, { text: "标准 PCIe", options: greenOpts }, { text: "自主芯片", options: greenOpts } ],
      [ { text: "fp4 原生", options: { ...cellOpts, bold: true, align: "left" } }, { text: "B200+", options: yellowOpts }, { text: "需解压", options: redOpts }, { text: "DSP 原生", options: greenOpts }, { text: "硬化", options: greenOpts } ],
      [ { text: "聚合 HBM 带宽", options: { ...cellOpts, bold: true, align: "left" } }, { text: "26.8 TB/s", options: cellOpts }, { text: "16 TB/s", options: cellOpts }, { text: "29.4 TB/s", options: greenOpts }, { text: "25.6 TB/s", options: greenOpts } ],
      [ { text: "B=1 有效带宽", options: { ...cellOpts, bold: true, align: "left" } }, { text: "~84 GB/s", options: redOpts }, { text: "~175 GB/s", options: yellowOpts }, { text: "~700 GB/s  (8× vs H100)", options: greenOpts }, { text: "~700 GB/s", options: greenOpts } ],
      [ { text: "聚合 Decode 吞吐\n(高并发)", options: { ...cellOpts, bold: true, align: "left" } }, { text: "~2,000 tok/s", options: cellOpts }, { text: "~1,500 tok/s", options: cellOpts }, { text: "~14,000 tok/s", options: greenOpts }, { text: "~12,000 tok/s", options: greenOpts } ],
      [ { text: "$/百万 token\n(100 套量产)", options: { ...cellOpts, bold: true, align: "left" } }, { text: "$12-20", options: redOpts }, { text: "$16-25", options: redOpts }, { text: "$5.9", options: greenOpts }, { text: "$2.5-3.5", options: greenOpts } ],
      [ { text: "硬件售价 (单套)", options: { ...cellOpts, bold: true, align: "left" } }, { text: "~$280K", options: cellOpts }, { text: "~$275K (实际)", options: yellowOpts }, { text: "~$303K", options: cellOpts }, { text: "~$70-80K", options: greenOpts } ],
    ];

    s.addTable(rows, { x: 0.7, y: 1.7, w: 8.6, colW: [1.4, 1.5, 1.55, 2.1, 2.05], rowH: [0.45, 0.36, 0.36, 0.36, 0.36, 0.36, 0.48, 0.48, 0.36], border: { pt: 0.5, color: C.border } });
  }

  // ═══════════════════════════════════════════════════════════════════════
  // SLIDE 8: ECONOMICS
  // ═══════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    sectionTitle(s, "经济模型", "架构带宽效率 → $/token 优势是结构性的，不是定价策略");
    addSlideNum(s, 8);
    addFooter(s, "FPGA LPU  |  Investor Briefing");

    // Left: BOM breakdown
    card(s, 0.7, 1.7, 4.3, 3.15);
    s.addImage({ data: icons.cubes, x: 1.0, y: 1.95, w: 0.4, h: 0.4 });
    s.addText("系统 BOM (32 芯片 HBM-Only)", { x: 1.55, y: 1.95, w: 3.2, h: 0.4, fontSize: 14, color: C.white, fontFace: FONT_H, bold: true, valign: "middle", margin: 0 });

    const bomItems = [
      ["FPGA 芯片 (32 × ¥1.8万)", "¥57.6万"],
      ["8 卡 PCB / 散热 / 电源 / 备件", "¥43.0万"],
      ["Dual Xeon GNR + DDR5 + SSD", "¥31.4万"],
      ["──────────────", ""],
      ["总 BOM", "¥133万 ($182K)"],
    ];
    bomItems.forEach((row, i) => {
      const isTotal = i === bomItems.length - 1;
      s.addText(row[0], { x: 1.0, y: 2.55 + i * 0.38, w: 2.5, h: 0.32, fontSize: isTotal ? 13 : 11, color: isTotal ? C.white : C.textSec, fontFace: FONT_B, bold: isTotal, margin: 0 });
      s.addText(row[1], { x: 3.6, y: 2.55 + i * 0.38, w: 1.2, h: 0.32, fontSize: isTotal ? 13 : 11, color: isTotal ? C.accent2 : C.text, fontFace: FONT_B, bold: isTotal, align: "right", margin: 0 });
    });

    // Right: $/M token comparison
    card(s, 5.3, 1.7, 4.1, 1.6);
    s.addText("$/百万 Token 对标 (100 套量产)", { x: 5.6, y: 1.9, w: 3.5, h: 0.35, fontSize: 14, color: C.white, fontFace: FONT_H, bold: true, margin: 0 });
    const priceData = [
      ["H100 云租赁", "$12-20", C.warn],
      ["Ascend 950PR (实际成交价)", "$16-25", C.warn],
      ["DeepSeek V4 Pro API (benchmark)", "$1.46", C.gold],
      ["FPGA (本方案)", "$5.9", C.accent2],
      ["ASIC (Phase 2)", "$2.5-3.5", C.accent2],
    ];
    priceData.forEach((row, i) => {
      s.addText(row[0], { x: 5.6, y: 2.35 + i * 0.24, w: 2.3, h: 0.22, fontSize: 10, color: C.textSec, fontFace: FONT_B, margin: 0 });
      s.addText(row[1], { x: 7.8, y: 2.35 + i * 0.24, w: 1.4, h: 0.22, fontSize: 10, color: row[2], fontFace: FONT_B, bold: true, align: "right", margin: 0 });
    });

    card(s, 5.3, 3.45, 4.1, 1.4);
    s.addText("量产后 + 混合负载优化", { x: 5.6, y: 3.55, w: 3.5, h: 0.3, fontSize: 13, color: C.accent, fontFace: FONT_H, bold: true, margin: 0 });
    const scaleData = [
      ["FPGA 10K 套 + 优化", "$1.03/M", C.accent2],
      ["ASIC 量产 + 优化", "$0.4-0.6/M", C.accent2],
    ];
    scaleData.forEach((row, i) => {
      s.addText(row[0], { x: 5.6, y: 3.95 + i * 0.28, w: 2.5, h: 0.24, fontSize: 10, color: C.textSec, fontFace: FONT_B, margin: 0 });
      s.addText(row[1], { x: 8.1, y: 3.95 + i * 0.28, w: 1.1, h: 0.24, fontSize: 11, color: C.accent2, fontFace: FONT_B, bold: true, align: "right", margin: 0 });
    });
    s.addText(`"ASIC 比 FPGA 便宜 50-60%，每次升级都成立"`, { x: 5.6, y: 4.55, w: 3.5, h: 0.2, fontSize: 8, color: C.textSec, fontFace: FONT_B, italic: true, margin: 0 });
  }

  // ═══════════════════════════════════════════════════════════════════════
  // SLIDE 9: ROADMAP
  // ═══════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    sectionTitle(s, "路线图：FPGA 验证 → ASIC 流片", "FPGA 是低风险验证平台，ASIC 是架构优势的物理固化");
    addSlideNum(s, 9);
    addFooter(s, "FPGA LPU  |  Investor Briefing");

    const phases = [
      { phase: "Phase 1", time: "现在-18 月", title: "FPGA 原型验证", items: ["单卡 / 8 卡原型验证", "fp4 精度 + HBM 带宽实测", "vLLM 集成 + 种子客户部署", "RTL IP 积累 + 架构迭代"], color: C.accent },
      { phase: "Phase 2", time: "18-36 月", title: "ASIC 流片量产", items: ["4 FPGA → 1 ASIC (12nm)", "RTL 复用率 >70%", "硬件成本 $70-80K/套", "$/M token $2.5-3.5"], color: C.accent2 },
    ];

    phases.forEach((p, i) => {
      const x = 0.7 + i * 4.6;
      card(s, x, 1.7, 4.1, 2.4);
      s.addShape(_pres.shapes.RECTANGLE, { x, y: 1.7, w: 4.1, h: 0.05, fill: { color: p.color } });
      s.addText(p.phase, { x: x + 0.25, y: 1.9, w: 1.5, h: 0.3, fontSize: 11, color: p.color, fontFace: FONT_B, bold: true, margin: 0 });
      s.addText(p.time, { x: x + 2.7, y: 1.9, w: 1.2, h: 0.3, fontSize: 10, color: C.textSec, fontFace: FONT_B, align: "right", margin: 0 });
      s.addText(p.title, { x: x + 0.25, y: 2.25, w: 3.6, h: 0.3, fontSize: 16, color: C.white, fontFace: FONT_H, bold: true, margin: 0 });
      s.addText(p.items.map((item, j) => ({ text: item, options: { bullet: true, breakLine: j < p.items.length - 1, fontSize: 11, color: C.textSec, fontFace: FONT_B, paraSpaceAfter: 6 } })), { x: x + 0.25, y: 2.65, w: 3.6, h: 1.2, margin: 0, valign: "top" });
    });

    // Bottom: ASIC cost comparison
    card(s, 0.7, 4.35, 8.6, 0.55);
    s.addText("ASIC 终局：$70-80K/套, $2.5-3.5/M token。架构优势 (有效带宽 83× + 切换 1000× + KV 地址 1000×) 在 12nm 硅片上物理固化。成本断层使 GPU/NPU 无法在 decode 场景跟进。", { x: 0.95, y: 4.35, w: 8.1, h: 0.55, fontSize: 11, color: C.textSec, fontFace: FONT_B, valign: "middle", margin: 0, lineSpacingMultiple: 1.3 });
  }

  // ═══════════════════════════════════════════════════════════════════════
  // SLIDE 10: SUMMARY & ASK
  // ═══════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };

    // Decorative element
    s.addShape(_pres.shapes.RECTANGLE, { x: -1, y: 3, w: 6, h: 4, fill: { color: C.accent, transparency: 94 }, rotate: -10 });

    s.addImage({ data: icons.rocket, x: 0.8, y: 0.8, w: 0.5, h: 0.5 });
    s.addText("FPGA 大模型推理集群", { x: 0.8, y: 1.5, w: 8.4, h: 0.7, fontSize: 36, color: C.white, fontFace: FONT_H, bold: true, margin: 0 });
    s.addText(`不是 "更便宜的 GPU"，是面向 Decode 场景的下一代计算范式`, { x: 0.8, y: 2.2, w: 8.4, h: 0.4, fontSize: 16, color: C.accent, fontFace: FONT_H, margin: 0 });

    // Key takeaways
    const takeaways = [
      { icon: icons.bolt, text: "三个 10-1000× 数量级差距 → 架构范式升级，非替代品对标" },
      { icon: icons.globe, text: "供应链不受美国 GPU 管制 → 全球可部署，8-12 周交付" },
      { icon: icons.chart, text: "Agent 时代 Decode 占比上升 → 范式优势向流式计算倾斜" },
      { icon: icons.rocket, text: "FPGA ($182K BOM) → ASIC ($70-80K) → 成本断层 + 架构固化" },
    ];

    takeaways.forEach((t, i) => {
      s.addImage({ data: t.icon, x: 0.8, y: 2.85 + i * 0.5, w: 0.32, h: 0.32 });
      s.addText(t.text, { x: 1.25, y: 2.85 + i * 0.5, w: 8, h: 0.32, fontSize: 13, color: C.text, fontFace: FONT_B, valign: "middle", margin: 0 });
    });

    // Bottom bar
    s.addShape(_pres.shapes.RECTANGLE, { x: 0, y: 5.15, w: 10, h: 0.475, fill: { color: C.cardBg } });
    s.addText("当前阶段：RTL 优化完成, 2D 脉动阵列验证通过, 三级 Prefill 架构就绪  |  下一步：启动 Phase 1 单卡验证", { x: 0.8, y: 5.15, w: 8.4, h: 0.475, fontSize: 10, color: C.accent, fontFace: FONT_B, valign: "middle", margin: 0 });
  }

  // ═══════════════════════════════════════════════════════════════════════
  // WRITE
  // ═══════════════════════════════════════════════════════════════════════
  const outPath = "D:\\workspace\\fpgalpu\\docs\\fpga_investor_pitch.pptx";
  await pres.writeFile({ fileName: outPath });
  console.log("Written: " + outPath);
}

main().catch(err => { console.error(err); process.exit(1); });
