# ID-UQ Framework: Scientific Figure Prompts for ChatGPT

---

## Figure 1: Framework Overview Pipeline
**论文位置**: Introduction / Methodology opening
**目的**: 一目了然展示从感知到不确定性量化到决策的完整闭环

### Prompt:
```
Create a clean scientific diagram in modern academic style for a robotics paper titled "Interaction-Driven Uncertainty Quantification for Ultrasound Visual Servoing". The diagram should show a closed-loop pipeline as a horizontal flow with 4 interconnected modules connected by arrows forming a cycle:

Left module (Perception): A robotic arm holding an ultrasound probe against soft tissue. Show a simplified ultrasound B-mode image with speckle patterns. Label: "Continuum Mechanics Perception (NLM + Affine Flow)"

Second module (Uncertainty Quantification): Two distinct paths diverging from the perception output. One path (red/orange tones) represents "Physical Eye R_phys" showing a waveform that spikes during contact anomalies. The other path (blue tones) represents "Geometric Eye S_geo" showing an image quality heatmap. These should visually look like two complementary "eyes" or sensing channels.

Third module (Uncertainty Decoupling): A 2D coordinate plane (phase space) with 4 colored quadrants. The x-axis is "Geometric Observability" and y-axis is "Physical Risk". One quadrant highlighted green as "Ideal Servoing Envelope". Show data points clustering or trajectories converging toward this green quadrant.

Right module (Active Decision): Show the robotic probe adjusting its pose — pressing deeper (Z-axis) or tilting (Roll/Pitch) — with curved arrows indicating the adjustment direction. Label: "Uncertainty-Guided Null-Space Control".

A feedback arrow loops from the right module back to the left, labeled "Active Re-observation".

Style: Clean vector illustration style, white background, professional academic colors (deep blue, crimson, dark orange, forest green). No photorealistic rendering. Use simple geometric shapes and icons. Text labels should be minimal and in English. The entire diagram should fit in a single cohesive composition.

Aspect ratio: 16:9 landscape.
```

### 备选（如果 AI 画流程图效果不好）:
用 draw.io 或 TikZ 手动绘制，AI 生成的作为概念插图放在旁边。

---

## Figure 2: The "Two Eyes" Physical Principle
**论文位置**: Methodology — Uncertainty Decoupling section
**目的**: 用视觉隐喻解释为什么两个不确定性维度是正交的、互补的

### Prompt:
```
Create a conceptual scientific illustration for a robotics paper. The image should use a compelling visual metaphor: a robotic ultrasound probe has "two eyes" that sense orthogonal types of uncertainty.

The scene: A side-view of a robotic arm holding an ultrasound probe against a cross-section of soft tissue (shown as layered organic structure in warm pink/beige tones).

The "Physical Eye" (shown as a red/orange glowing sensor beam coming from the probe tip into the tissue): This eye senses MECHANICAL INTERACTION. Show it as a pressure/force visualization — concentric ripples or stress lines propagating from the probe-tissue interface into the tissue depth. It detects "contact loss" (probe lifting off, shown as a small gap with red warning indicator) and "tissue slip" (lateral displacement arrow with red highlight).

The "Geometric Eye" (shown as a blue/cyan glowing sensor beam from the probe side, looking AT the ultrasound image itself): This eye senses IMAGE QUALITY. Show it as a beam analyzing the ultrasound image frame (shown as a rectangular B-mode image beside the probe with grainy speckle texture). It detects "acoustic shadowing" (dark region in the image with blue highlight) and "speckle clarity" (sharp vs blurred region comparison).

Key concept to convey: The two eyes look in DIFFERENT DIRECTIONS (one into tissue mechanics, one at image quality) and are mathematically ORTHOGONAL (complementary, not redundant). This is the core innovation.

At the bottom, show two small signal panels side by side:
Left panel (Physical Eye output): A time-series line chart showing R_phys spiking (red) when contact fails.
Right panel (Geometric Eye output): A time-series line chart showing S_geo dropping (blue) when image quality degrades.

The two panels should show these events happening at DIFFERENT TIMES, emphasizing they capture independent failure modes.

Style: Clean scientific illustration, cross between medical textbook and robotics conference figure. White/light gray background. Professional color palette. Subtle glow effects on the two "eye" beams to distinguish them. Labels in English, minimal but clear.

Aspect ratio: 16:9 landscape.
```

---

## Figure 3: Four-Quadrant Uncertainty Phase Space
**论文位置**: Methodology — Dual-Eye Decoupling (对应 exp6)
**目的**: 论文的核心理论贡献可视化——不确定性正交分解的几何证明

### Prompt:
```
Create a clean, publication-quality scientific plot visualization. This is a 2D Cartesian "phase space" diagram that is the theoretical centerpiece of a robotics paper.

The plot:
- X-axis: "Geometric Eye S_geo (Acoustic Observability)" — arrow pointing right, labeled "Better Image Quality →"
- Y-axis: "Physical Eye R_phys (Interaction Risk)" — arrow pointing up, labeled "Higher Contact Risk →"

The 2D space is divided into 4 quadrants by two dashed lines (one horizontal at mid-Y, one vertical at mid-X):

QUADRANT I (top-right, colored light red/salmon): "Q1: Pure Sliding / Kinematic Mismatch"
Show a small icon: probe sliding sideways on tissue surface. 
Annotation: "High Risk, Clear Image — most deceptive failure mode"

QUADRANT II (top-left, colored gray): "Q2: Severe Decoupling"
Show a small icon: probe completely losing contact with tissue.
Annotation: "High Risk, Air Gap — catastrophic failure"

QUADRANT III (bottom-left, colored gold/amber): "Q3: Acoustic Shadowing"
Show a small icon: probe on tissue but ultrasound image showing dark shadow.
Annotation: "Safe Press, Poor Visibility — rib/gas obstruction"

QUADRANT IV (bottom-right, colored light green, glowing border): "Q4: Ideal Servoing Envelope"
Show a small icon: probe making perfect contact with clear image.
Annotation: "Safe Press, Clear Tissue — target operating region ★"

Inside the plot area, show:
- Scattered small dots representing real data points (from experiments)
- A few curved trajectory arrows starting from Q1/Q2/Q3 and converging toward Q4, labeled "UQ-Guided Policy"
- A concentration of points in Q4

Style: Professional academic figure. Clean lines, readable font, colorblind-friendly palette. Should look like it belongs in an IEEE ICRA or IROS paper. White background, no grid clutter.

Aspect ratio: 1:1 square or 4:3.
```

### 备注:
这张图 exp6 已经用代码生成了数据版本，AI 生成的是**概念化加强版**——加上图标、注释、轨迹箭头，比纯数据图更有阐述力。两版可以并列使用：AI 概念图在 methodology，数据图在 experiments。

---

## Figure 4: Affine Flow Physics — Why Divergence?
**论文位置**: Methodology — Continuum Mechanics Perception
**目的**: 解释为什么散度(D)是正确的物理量，而非简单的图像面积或位移

### Prompt:
```
Create a scientific illustration explaining the continuum mechanics principle behind affine flow divergence for a robotics paper on ultrasound servoing.

The figure should have three panels side by side:

LEFT PANEL — "Tissue Compression":
Show a simplified cross-section of soft tissue (wavy layered structure in pink/beige). An ultrasound probe presses down from above (shown as a rectangular probe head with a downward arrow labeled "v_z"). The tissue compresses vertically and EXPANDS LATERALLY (Poisson effect). Small arrows within the tissue show the deformation field — vertical compression arrows (downward) and lateral expansion arrows (left-right). Label: "Volumetric strain: ∇·v ≠ 0"

MIDDLE PANEL — "Affine Flow Field":
Show a simplified ultrasound image frame (rectangular, grainy gray texture). Overlay a vector field (small arrows) showing the optical flow between two consecutive frames during compression. The arrows should form a pattern radiating outward from the center — this is the DIVERGENCE field. At the bottom, show the mathematical decomposition:
- Divergence (D) = ∂u/∂x + ∂v/∂y → "Captures compression/expansion" (highlighted in red)
- Curl (R) = ∂v/∂x - ∂u/∂y → "Captures rotation" (shown in a different color, grayed out or secondary)

RIGHT PANEL — "Signal Comparison":
A simple comparison showing three possible visual signals and their response to tissue compression:
1. "Image Area" — flat line (no response) — labeled "FAILS: area is invariant under compression"
2. "Pixel Intensity" — noisy, inconsistent — labeled "FAILS: confounded by speckle"
3. "Affine Divergence (D)" — clear, correlated signal — labeled "OURS: physically meaningful" with a green checkmark

Style: Professional scientific figure, clean vector illustration style. Use consistent color coding throughout (red for divergence/compression, blue for other signals). White background. Labels in English.

Aspect ratio: 16:9 or 2:1 wide.
```

---

## Figure 5: Clinical Application Scenario & Failure Modes
**论文位置**: Introduction / Problem Statement
**目的**: 用真实感场景让审稿人立刻理解问题的临床意义

### Prompt:
```
Create a scientific/medical illustration showing the clinical application scenario for a robotics paper on autonomous ultrasound scanning.

The image should be a single composed scene with 4 labeled components:

MAIN SCENE (center, largest): A robotic arm (sleek, modern, medical-grade appearance) holding an ultrasound probe against a patient's abdomen/phantom. The scene is shown in semi-cross-section so we can see both the external setup and the internal tissue layers. The tissue is shown as layered organic structure (skin, fat, muscle layers in warm medical-illustration colors). The probe emits a fan-shaped ultrasound beam (shown as a subtle translucent cone) into the tissue.

THREE FAILURE MODE INSETS (smaller panels arranged around the main scene, connected by subtle lines):

Inset A — "Loss of Contact" (top):
The probe has lifted slightly off the skin surface. A visible air gap (white space) between probe and tissue. The ultrasound image beside it shows a dark/blurry frame with a red X. Small R_phys gauge indicator pointing to "HIGH RISK".

Inset B — "In-Plane Slip" (right):
The probe is in contact but sliding laterally. Show a lateral displacement arrow. The ultrasound image shows decorrelation artifacts. R_phys gauge at "HIGH RISK" but contact sensor (if one existed) would show "NORMAL" — illustrating why force sensors alone fail.

Inset C — "Acoustic Shadowing" (left):
The probe has good contact but the ultrasound beam encounters a rib or gas pocket (shown as a dark obstruction in the tissue cross-section). The resulting image has a dark shadow band. S_geo gauge at "LOW OBSERVABILITY".

At the bottom of the main scene, show a small dashboard-like panel with two gauges:
- R_phys gauge (red zone = dangerous, green zone = safe)
- S_geo gauge (red zone = poor visibility, green zone = clear)
These represent the ID-UQ framework's real-time uncertainty monitoring.

Style: Professional medical illustration meets robotics conference figure. Clean, precise, scientifically accurate anatomy (not cartoonish). The robotics equipment should look realistic but stylized. Color palette: medical warm tones for tissue, cool blue/gray for robot, red for risk indicators, green for safe indicators.

Aspect ratio: 16:9 landscape.
```

---

## 使用建议

| 图 | AI生成难度 | 备选方案 |
|---|---|---|
| Fig 1: Framework Pipeline | 中等（可能文字不准） | 若效果不好→ draw.io 手动绘制，AI 图作为概念插图 |
| Fig 2: Two Eyes | 低（视觉隐喻，AI 擅长） | 直接可用 |
| Fig 3: Phase Space | 中等 | exp6 已有数据版，AI 出概念版 |
| Fig 4: Affine Physics | 中高（数学符号可能不准） | TikZ/PPT 补标签 |
| Fig 5: Clinical Scenario | 低（场景描述清晰） | 直接可用 |

**关键提示**：
1. 所有 AI 生成的图需要后期在 PPT/Illustrator 中补精确的文字标注
2. 统一配色方案：红=R_phys/风险，蓝=S_geo/几何，绿=安全/理想，橙=警告
3. 中文字体在 DALL-E 中容易乱码，建议先用英文生成，再后期叠加中文标签
