// web/gl.js —— 权重直方图热力图（WebGL2）。
//
// 设计要点（对应 docs/telemetry-plan.md §8.2）：
//   - 纹理里存"原始 counts"（R32F），只把一个 uLogMax uniform 随数据更新，
//     所以全局尺度变化时不需要重传纹理。
//   - 每次新记录只 texSubImage2D 写一列。
//   - R32F 在 WebGL2 core 里不可线性过滤，所以用 NEAREST。
//   - 上传前必须 gl.pixelStorei(UNPACK_ALIGNMENT, 1)。

const VERT = `#version 300 es
const vec2 verts[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
void main() { gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0); }
`;

const FRAG = `#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform float uTotalRecords;
uniform float uBins;
uniform float uLogMax;
uniform float uFilled;
uniform vec2 uSize;
out vec4 outColor;

vec3 colormap(float t) {
  const vec3 c0 = vec3(0.267, 0.005, 0.329);
  const vec3 c1 = vec3(0.188, 0.408, 0.556);
  const vec3 c2 = vec3(0.208, 0.718, 0.473);
  const vec3 c3 = vec3(0.993, 0.906, 0.144);
  if (t < 0.3333) return mix(c0, c1, t / 0.3333);
  if (t < 0.6666) return mix(c1, c2, (t - 0.3333) / 0.3334);
  return mix(c2, c3, (t - 0.6666) / 0.3334);
}

void main() {
  float col = floor(gl_FragCoord.x * uTotalRecords / uSize.x);
  float row = floor(gl_FragCoord.y * uBins / uSize.y);
  if (col < 0.0 || col >= uTotalRecords || row < 0.0 || row >= uBins || col >= uFilled) {
    outColor = vec4(0.07, 0.086, 0.113, 1.0);
    return;
  }
  float v = texelFetch(uTex, ivec2(int(col), int(row)), 0).r;
  float t = uLogMax > 0.0 ? log(1.0 + max(v, 0.0)) / uLogMax : 0.0;
  outColor = vec4(colormap(clamp(t, 0.0, 1.0)), 1.0);
}
`;

export class HistogramHeatmap {
  constructor(canvas, { bins, layerCount, totalRecords, maxCols = 8192 }) {
    this.canvas = canvas;
    this.bins = bins;
    this.layerCount = layerCount;
    this.totalRecords = Math.max(1, totalRecords);
    this.cols = Math.min(this.totalRecords, maxCols);
    this.layer = 0;
    this.filled = 0;
    this.logMax = 0;
    this.lost = false;
    this.overflowed = false;
    this.gl = null;
    this.textures = [];
    this.initContext();
  }

  initContext() {
    const gl = this.canvas.getContext('webgl2', { antialias: false, alpha: false });
    if (!gl) {
      this.gl = null;
      return;
    }
    this.gl = gl;

    const compile = (type, src) => {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(sh) || 'shader compile failed');
      }
      return sh;
    };

    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(prog) || 'program link failed');
    }
    this.program = prog;
    this.uTex = gl.getUniformLocation(prog, 'uTex');
    this.uTotalRecords = gl.getUniformLocation(prog, 'uTotalRecords');
    this.uBins = gl.getUniformLocation(prog, 'uBins');
    this.uLogMax = gl.getUniformLocation(prog, 'uLogMax');
    this.uFilled = gl.getUniformLocation(prog, 'uFilled');
    this.uSize = gl.getUniformLocation(prog, 'uSize');

    this.vao = gl.createVertexArray();
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

    this.textures = [];
    for (let l = 0; l < this.layerCount; l += 1) {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, this.cols, this.bins, 0, gl.RED, gl.FLOAT, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      this.textures.push(tex);
    }

    gl.clearColor(0.07, 0.086, 0.113, 1);
    this.contextLost = false;
    this.canvas.addEventListener('webglcontextlost', (e) => {
      e.preventDefault();
      this.gl = null;
      this.lost = true;
    });
    this.canvas.addEventListener('webglcontextrestored', () => {
      this.initContext();
      this.reuploadAll = true;
    });
  }

  setTotalRecords(n) {
    if (n > 0) this.totalRecords = Math.max(1, n);
  }

  // 文件被截断/换 run 时整体重置：不清纹理，靠 uFilled 把旧列挡掉
  reset() {
    this.filled = 0;
    this.logMax = 0;
    this.overflowed = false;
  }

  setLayer(index) {
    this.layer = index;
    this.render();
  }

  // 一条记录里含所有层的 counts；只上传当前选中层，避免无谓的显存写入
  pushRecord(record) {
    if (!this.gl) return;
    if (this.filled >= this.cols) {
      this.overflowed = true;
      return;
    }
    const gl = this.gl;
    for (let l = 0; l < this.layerCount && l < record.layers.length; l += 1) {
      const counts = record.layers[l].counts;
      gl.bindTexture(gl.TEXTURE_2D, this.textures[l]);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, this.filled, 0, 1, this.bins, gl.RED, gl.FLOAT, counts);
      for (let b = 0; b < counts.length; b += 1) {
        if (counts[b] > this.logMax) this.logMax = counts[b];
      }
    }
    this.filled += 1;
  }

  render() {
    const gl = this.gl;
    if (!gl) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width * dpr));
    const h = Math.max(1, Math.floor(rect.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    gl.viewport(0, 0, w, h);
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.textures[this.layer] || this.textures[0]);
    gl.uniform1i(this.uTex, 0);
    gl.uniform1f(this.uTotalRecords, this.totalRecords);
    gl.uniform1f(this.uBins, this.bins);
    gl.uniform1f(this.uLogMax, this.logMax > 0 ? Math.log(1 + this.logMax) : 0);
    gl.uniform1f(this.uFilled, this.filled);
    gl.uniform2f(this.uSize, w, h);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }
}
