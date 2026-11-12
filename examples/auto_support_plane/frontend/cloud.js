const vertex = `#version 300 es
in vec3 position;
uniform vec3 eye, forward, right, up;
uniform float aspect, pointSize, band;
uniform vec4 plane;
uniform bool hasPlane, planeOnly, frozen;
out vec3 color;
void main() {
  bool selected = hasPlane && abs(dot(position, plane.xyz) + plane.w) <= band;
  vec3 p = position * vec3(0.001, -0.001, -0.001) - eye;
  float z = dot(p, forward);
  float near = 0.01, far = 100.0;
  gl_Position = vec4(dot(p,right) / (aspect * 0.4663), dot(p,up) / 0.4663,
      (far+near)/(far-near)*z - 2.0*far*near/(far-near), z);
  if (planeOnly && !selected) gl_Position = vec4(2,2,2,1);
  gl_PointSize = pointSize;
  color = selected ? (frozen ? vec3(1.0,0.71,0.29) : vec3(0.25,0.90,0.72))
      : mix(vec3(0.25,0.35,0.44), vec3(0.64,0.73,0.80), clamp(position.z/3500.0,0.0,1.0));
}`;
const fragment = `#version 300 es
precision highp float;
in vec3 color;
out vec4 outColor;
void main() {
  if (length(gl_PointCoord-vec2(0.5)) > 0.5) discard;
  outColor = vec4(color, 1.0);
}`;
const normalize = a => { const n = Math.hypot(...a); return a.map(x => x / n); };
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];

export class CloudView {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = this.gl = canvas.getContext('webgl2', { antialias: true });
    if (!gl) throw new Error('This browser needs WebGL 2 to display the point cloud');
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source); gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      return shader;
    };
    this.program = gl.createProgram();
    gl.attachShader(this.program, compile(gl.VERTEX_SHADER, vertex));
    gl.attachShader(this.program, compile(gl.FRAGMENT_SHADER, fragment));
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(this.program));
    this.buffer = gl.createBuffer();
    this.uniforms = Object.fromEntries(['eye','forward','right','up','aspect','pointSize','band','plane','hasPlane','planeOnly','frozen']
      .map(name => [name, gl.getUniformLocation(this.program, name)]));
    this.count = 0; this.pointSize = 3; this.planeOnly = false;
    this.focus = [0,0,-1.2]; this.distance = 3; this.yaw = 0.12; this.pitch = 0.12;
    this.points = new Float32Array(); this.metadata = {}; this.firstCloud = true;
    this.controls();
    this.resizeObserver = new ResizeObserver(() => this.draw());
    this.resizeObserver.observe(canvas);
  }

  controls() {
    let drag;
    this.canvas.addEventListener('contextmenu', e => e.preventDefault());
    this.canvas.addEventListener('pointerdown', e => {
      drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 2 };
      this.canvas.setPointerCapture(e.pointerId);
    });
    this.canvas.addEventListener('pointerup', () => { drag = null; });
    this.canvas.addEventListener('pointercancel', () => { drag = null; });
    this.canvas.addEventListener('pointermove', e => {
      if (!drag) return;
      const dx = e.clientX-drag.x, dy = e.clientY-drag.y;
      drag.x = e.clientX; drag.y = e.clientY;
      if (drag.pan) {
        const {right,up} = this.basis();
        this.focus = this.focus.map((v,i) => v + (-dx*right[i]+dy*up[i])*this.distance*0.0015);
      } else {
        this.yaw -= dx*0.005; this.pitch = Math.max(-1.5,Math.min(1.5,this.pitch+dy*0.005));
      }
      this.draw();
    });
    this.canvas.addEventListener('wheel', e => {
      e.preventDefault(); this.distance = Math.max(0.1,Math.min(50,this.distance*Math.exp(e.deltaY*0.001)));
      this.draw();
    }, {passive:false});
  }

  basis() {
    const offset = [Math.sin(this.yaw)*Math.cos(this.pitch), Math.sin(this.pitch), Math.cos(this.yaw)*Math.cos(this.pitch)];
    const eye = this.focus.map((v,i) => v+offset[i]*this.distance);
    const forward = offset.map(x => -x), right = normalize(cross(forward,[0,1,0])), up = cross(right,forward);
    return {eye,forward,right,up};
  }

  setCloud(points, metadata) {
    this.points = points; this.metadata = metadata; this.count = points.length/3;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer); gl.bufferData(gl.ARRAY_BUFFER,points,gl.DYNAMIC_DRAW);
    if (this.firstCloud && this.count) { this.firstCloud = false; this.reset(); }
    this.draw();
  }

  reset() {
    const min = [Infinity,Infinity,Infinity], max = [-Infinity,-Infinity,-Infinity];
    for (let i=0; i<this.points.length; i+=3) {
      const p = [this.points[i]*0.001,-this.points[i+1]*0.001,-this.points[i+2]*0.001];
      for (let j=0;j<3;j++) { min[j]=Math.min(min[j],p[j]); max[j]=Math.max(max[j],p[j]); }
    }
    if (this.count) {
      this.focus = min.map((v,i)=>(v+max[i])/2);
      this.distance = Math.max(0.8,Math.min(30,Math.hypot(...max.map((v,i)=>v-min[i]))*1.3));
    }
    this.yaw = 0.12; this.pitch = 0.12; this.draw();
  }

  draw() {
    const gl = this.gl, canvas = this.canvas, ratio = Math.min(devicePixelRatio,2);
    canvas.width = Math.max(1,Math.round(canvas.clientWidth*ratio));
    canvas.height = Math.max(1,Math.round(canvas.clientHeight*ratio));
    gl.viewport(0,0,canvas.width,canvas.height);
    gl.clearColor(0.035,0.065,0.09,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST); gl.useProgram(this.program);
    gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);
    const position = gl.getAttribLocation(this.program,'position');
    gl.enableVertexAttribArray(position); gl.vertexAttribPointer(position,3,gl.FLOAT,false,0,0);
    for (const [name,value] of Object.entries(this.basis())) gl.uniform3fv(this.uniforms[name],value);
    gl.uniform1f(this.uniforms.aspect,canvas.width/canvas.height);
    gl.uniform1f(this.uniforms.pointSize,this.pointSize*ratio);
    gl.uniform1f(this.uniforms.band,this.metadata.band_mm ?? 30);
    const plane = this.metadata.plane;
    gl.uniform4fv(this.uniforms.plane,plane ? [...plane.normal,plane.offset_mm] : [0,0,0,0]);
    gl.uniform1i(this.uniforms.hasPlane,Boolean(plane));
    gl.uniform1i(this.uniforms.planeOnly,this.planeOnly);
    gl.uniform1i(this.uniforms.frozen,Boolean(this.metadata.frozen));
    gl.drawArrays(gl.POINTS,0,this.count);
  }
}
