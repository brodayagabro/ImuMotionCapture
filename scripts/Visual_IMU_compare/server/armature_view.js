// Reused octahedral armature renderer from visual_mocap_server.py.
// Native WebGL, no CDN, Node build, Three.js or browser plugins required.
const canvas=$('gl'), gl=canvas.getContext('webgl',{antialias:true});let yaw=.35,pitch=.12,distance=3.5,target=[0,.9,0],autoCenter=true;
if(!gl){$('empty').textContent='WebGL недоступен. Включите аппаратное ускорение браузера.'}else{
 const vs='attribute vec3 p; attribute vec3 color; uniform mat4 mvp; varying vec3 c; void main(){gl_Position=mvp*vec4(p,1.0);gl_PointSize=7.0;c=color;}',fs='precision mediump float; varying vec3 c; void main(){gl_FragColor=vec4(c,1.0);}';
 function shader(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s}
 const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vs));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));gl.useProgram(program);
 const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);for(const [name,offset] of [['p',0],['color',12]]){const loc=gl.getAttribLocation(program,name);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,3,gl.FLOAT,false,24,offset)}const matrix=gl.getUniformLocation(program,'mvp');gl.enable(gl.DEPTH_TEST);
 const sub=(a,b)=>a.map((x,i)=>x-b[i]),dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],norm=a=>{const n=Math.hypot(...a);return a.map(x=>x/n)};
 function multiply(a,b){const o=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)o[c*4+r]+=a[k*4+r]*b[c*4+k];return o}
 function mvp(){const eye=[target[0]+distance*Math.cos(pitch)*Math.sin(yaw),target[1]+distance*Math.sin(pitch),target[2]+distance*Math.cos(pitch)*Math.cos(yaw)],z=norm(sub(eye,target)),x=norm(cross([0,1,0],z)),y=cross(z,x);const view=[x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1];const f=1/Math.tan(Math.PI/8),aspect=canvas.width/canvas.height,n=.02,far=150;return multiply([f/aspect,0,0,0,0,f,0,0,0,0,(far+n)/(n-far),-1,0,0,2*far*n/(n-far),0],view)}
 const pointers=new Map();canvas.onpointerdown=e=>{canvas.setPointerCapture(e.pointerId);pointers.set(e.pointerId,[e.clientX,e.clientY])};canvas.onpointerup=canvas.onpointercancel=e=>pointers.delete(e.pointerId);canvas.onpointermove=e=>{if(!pointers.has(e.pointerId))return;const old=pointers.get(e.pointerId);if(pointers.size===1){yaw-=(e.clientX-old[0])*.006;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-old[1])*.006))}else{const other=[...pointers.entries()].find(([id])=>id!==e.pointerId)[1];const before=Math.hypot(old[0]-other[0],old[1]-other[1]),after=Math.hypot(e.clientX-other[0],e.clientY-other[1]);if(after>0)distance=Math.max(.3,Math.min(80,distance*before/after))}pointers.set(e.pointerId,[e.clientX,e.clientY])};canvas.onwheel=e=>{e.preventDefault();distance=Math.max(.3,Math.min(80,distance*Math.exp(e.deltaY*.001))) };
 $('resetView').onclick=()=>{autoCenter=true;yaw=.35;pitch=.12;distance=3.5};
 // Octahedra match host/viz/pyqt_mocap/human_canvas_gl.py: 20% head,
 // radius 12% (at most 4.5 cm), eight shaded facets and dark edges.
 const add=(a,b)=>a.map((v,i)=>v+b[i]),scale=(a,k)=>a.map(v=>v*k);
 const shades=[1,.90,.78,.70,.88,.80,.96,.85];
 function boneMesh(bone,color,triangles,edges){
  const delta=sub(bone.end,bone.start),length=Math.hypot(...delta);if(length<1e-8)return;
  const direction=scale(delta,1/length),axes=bone.axes;
  const reference=axes.reduce((a,b)=>Math.abs(dot(direction,a))<=Math.abs(dot(direction,b))?a:b);
  const u=norm(cross(direction,reference)),v=cross(direction,u),radius=Math.min(length*.12,.045),center=add(bone.start,scale(delta,.2));
  const vertices=[bone.start,bone.end,add(center,scale(u,radius)),add(center,scale(v,radius)),add(center,scale(u,-radius)),add(center,scale(v,-radius))];
  for(let i=0;i<4;i++){const a=2+i,b=2+(i+1)%4;for(const [k,face] of [[0,[0,b,a]],[1,[1,a,b]]])for(const index of face)triangles.push(...vertices[index],...scale(color,shades[2*i+k]));
   for(const [a,b] of [[0,2+i],[1,2+i],[2+i,2+(i+1)%4]])edges.push(...vertices[a],.18,.24,.30,...vertices[b],.18,.24,.30);
  }
 }
 let centeredRig=null,centeredLive=false,floor=0;
 function draw(){
  const dpr=Math.min(devicePixelRatio||1,2),w=Math.max(1,Math.round(canvas.clientWidth*dpr)),h=Math.max(1,Math.round(canvas.clientHeight*dpr));
  if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}gl.viewport(0,0,w,h);gl.clearColor(248/255,250/255,252/255,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  advancePlayer(performance.now());
  const points=armature.points;
  if(autoCenter){target=[0,1,2].map(i=>points.reduce((sum,p)=>sum+p[i],0)/points.length);floor=Math.min(...points.map(p=>p[1]));const extent=Math.max(...[0,1,2].map(i=>Math.max(...points.map(p=>p[i]))-Math.min(...points.map(p=>p[i]))));distance=Math.max(.3,extent*2.5);autoCenter=false}
  const grid=[],triangles=[],edges=[];
  const line=(a,b,c)=>grid.push(...a,...c,...b,...c);
  for(let i=-5;i<=5;i++){line([target[0]+i*.5,floor,target[2]-2.5],[target[0]+i*.5,floor,target[2]+2.5],[.82,.86,.90]);line([target[0]-2.5,floor,target[2]+i*.5],[target[0]+2.5,floor,target[2]+i*.5],[.82,.86,.90])}
  for(const bone of armature.bones){const color=clip?(bone.name.includes('Leg')||bone.name.includes('Foot')?[.30,.58,.77]:[.88,.55,.41]):[.54,.62,.70];boneMesh(bone,color,triangles,edges)}
  gl.uniformMatrix4fv(matrix,false,mvp());
  function upload(data,mode){gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(data),gl.DYNAMIC_DRAW);gl.drawArrays(mode,0,data.length/6)}
  upload(grid,gl.LINES);gl.enable(gl.POLYGON_OFFSET_FILL);gl.polygonOffset(1,1);upload(triangles,gl.TRIANGLES);gl.disable(gl.POLYGON_OFFSET_FILL);upload(edges,gl.LINES);
  requestAnimationFrame(draw)
 }draw();
}
