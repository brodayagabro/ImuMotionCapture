const assert=require('node:assert/strict');
const {parseBVH,bvhArmature}=require('./bvh_player.js');
const text=`HIERARCHY
ROOT Hip { OFFSET 0 0 1 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
 JOINT Arm { OFFSET 1 0 0 CHANNELS 3 Yrotation Zrotation Xrotation
  End Site { OFFSET 1 0 0 } } }
MOTION
Frames: 2
Frame Time: 0.033333333
0 0 0 0 0 0 0 0 0
2 0 0 90 0 0 0 0 0
`;
const clip=parseBVH(text);assert.equal(clip.frameCount,2);assert.equal(clip.channelCount,9);
function close(actual,expected){actual.forEach((v,i)=>assert.ok(Math.abs(v-expected[i])<1e-5,`${actual} != ${expected}`))}
close(bvhArmature(clip,0,false).points[2],[2,0,1]);
close(bvhArmature(clip,1,false).points[2],[2,2,1]);
close(bvhArmature(clip,1,true).points[2],[2,1,-2]);
close(bvhArmature(clip,1,false,1,true).points[2],[2,2,0]);
// Channel declaration, not a fixed Euler convention, controls the result.
const reordered=parseBVH(text.replace('Zrotation Xrotation Yrotation','Xrotation Zrotation Yrotation'));
close(bvhArmature(reordered,1,false).points[2],[4,0,1]);
assert.throws(()=>parseBVH(text.replace('Frames: 2','Frames: 3')),/count mismatch/);
assert.throws(()=>parseBVH(text.replace('0.033333333','0')),/invalid BVH motion/);
console.log('BVH playback: hierarchy, declared channel order, translation, axes and validation OK');
