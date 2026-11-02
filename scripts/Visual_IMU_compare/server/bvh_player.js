/* BVH reader/FK for playback only. Recording/export remains native FreeMoCap.
 * CHANNELS are evaluated in the declared order; no hardcoded joint names.
 */
'use strict';
function parseBVH(text) {
  const tokens=text.match(/[^\s{}]+|[{}]/g)||[]; let i=0, channelCount=0;
  const joints=[];
  function expect(value){if(tokens[i++]!==value)throw Error('Invalid BVH: expected '+value)}
  function number(){const value=Number(tokens[i++]);if(!Number.isFinite(value))throw Error('Invalid BVH number');return value}
  function joint(parent, end=false){
    const name=end?joints[parent].name+':End':tokens[i++];
    if(!name)throw Error('Missing BVH joint name');
    const index=joints.length, item={name,parent,offset:null,channels:[],start:channelCount};joints.push(item);
    expect('{');expect('OFFSET');item.offset=[number(),number(),number()];
    if(!end){expect('CHANNELS');const n=number();if(!Number.isInteger(n)||n<0||n>6)throw Error('Invalid channels');
      item.channels=tokens.slice(i,i+n);i+=n;channelCount+=n;
      if(item.channels.some(c=>!['Xposition','Yposition','Zposition','Xrotation','Yrotation','Zrotation'].includes(c)))throw Error('Unsupported BVH channel');
      while(tokens[i]!=='}'){if(tokens[i]==='JOINT'){i++;joint(index)}else if(tokens[i]==='End'){i++;expect('Site');joint(index,true)}else throw Error('Invalid BVH hierarchy')}
    }expect('}');
  }
  expect('HIERARCHY');expect('ROOT');joint(-1);expect('MOTION');expect('Frames:');const frameCount=number();
  expect('Frame');expect('Time:');const frameTime=number();
  if(!Number.isInteger(frameCount)||frameCount<1||!(frameTime>0)||!channelCount)throw Error('Empty/invalid BVH motion');
  if(tokens.length-i!==frameCount*channelCount)throw Error('BVH frame/channel count mismatch');
  const motion=new Float32Array(frameCount*channelCount);for(let n=0;n<motion.length;n++)motion[n]=number();
  return {joints,motion,frameCount,frameTime,channelCount};
}
function bvhArmature(clip, frame, zUp=true, scale=1, blenderPositions=false){
  const mul=(a,b)=>{const c=Array(9).fill(0);for(let r=0;r<3;r++)for(let k=0;k<3;k++)for(let j=0;j<3;j++)c[3*r+j]+=a[3*r+k]*b[3*k+j];return c};
  const mv=(a,v)=>[0,1,2].map(r=>a[r*3]*v[0]+a[r*3+1]*v[1]+a[r*3+2]*v[2]);
  const add=(a,b)=>a.map((v,j)=>v+b[j]);
  const identity=()=>[1,0,0,0,1,0,0,0,1];
  const point=v=>(zUp?[v[0],v[2],-v[1]]:v).map(x=>x*scale);
  const axis=v=>zUp?[v[0],v[2],-v[1]]:v;
  const poses=[];const bones=[];
  const offset=Math.max(0,Math.min(clip.frameCount-1,Math.floor(frame)))*clip.channelCount;
  for(const joint of clip.joints){
    // Blender's exporter writes absolute local locations including the rest
    // head offset; its importer subtracts OFFSET before applying pose.location.
    // Other BVH writers commonly emit an additive translation instead.
    let rotation=identity(), translation=blenderPositions&&joint.channels.some(c=>c.endsWith('position'))?[0,0,0]:joint.offset.slice();
    joint.channels.forEach((channel,j)=>{
      const value=clip.motion[offset+joint.start+j], k='XYZ'.indexOf(channel[0]);
      if(channel.endsWith('position')){const v=[0,0,0];v[k]=value;translation=add(translation,mv(rotation,v))}
      else{const a=value*Math.PI/180,c=Math.cos(a),s=Math.sin(a);
        const r=k===0?[1,0,0,0,c,-s,0,s,c]:k===1?[c,0,s,0,1,0,-s,0,c]:[c,-s,0,s,c,0,0,0,1];rotation=mul(rotation,r)}
    });
    if(joint.parent>=0){const parent=poses[joint.parent];translation=add(parent.position,mv(parent.rotation,translation));rotation=mul(parent.rotation,rotation)}
    poses.push({position:translation,rotation});
    if(joint.parent>=0){const parent=poses[joint.parent];bones.push({name:joint.name,start:point(parent.position),end:point(translation),tracked:true,
      axes:[0,1,2].map(j=>axis([parent.rotation[j],parent.rotation[j+3],parent.rotation[j+6]]))})}
  }
  return {points:poses.map(p=>point(p.position)),bones};
}
if(typeof module!=='undefined')module.exports={parseBVH,bvhArmature};
