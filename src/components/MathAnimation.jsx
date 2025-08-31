\
import React from "react";
import { motion } from "framer-motion";

/**
 * 迷你演示：3 个节点，消息沿边传播，代表一次 GCN 相邻消息聚合
 */
export default function MathAnimation(){
  const nodes = [
    { id:0, x:40,  y:80 },
    { id:1, x:200, y:40 },
    { id:2, x:200, y:120 },
  ];
  const edges = [
    { src:0, dst:1 }, { src:0, dst:2 },
  ];
  return (
    <div style={{height:220, position:"relative"}}>
      <svg width="100%" height="100%" viewBox="0 0 280 180">
        {edges.map((e,i)=>(
          <line key={i} x1={nodes[e.src].x} y1={nodes[e.src].y}
                x2={nodes[e.dst].x} y2={nodes[e.dst].y}
                stroke="rgba(255,255,255,0.6)" strokeWidth="2"/>
        ))}
        {edges.map((e,i)=>(
          <motion.circle key={"m"+i} r="5" fill="#00e5ff"
            initial={{ cx:nodes[e.src].x, cy:nodes[e.src].y }}
            animate={{ cx:nodes[e.dst].x, cy:nodes[e.dst].y }}
            transition={{ duration:1.2, repeat:Infinity, repeatDelay:0.4 }}
          />
        ))}
        {nodes.map(n=>(
          <g key={n.id}>
            <circle cx={n.x} cy={n.y} r="14" fill="rgba(255,255,255,0.2)" stroke="#fff"/>
            <text x={n.x} y={n.y+4} textAnchor="middle" fill="#fff" fontSize="10">v{n.id}</text>
          </g>
        ))}
      </svg>
      <div style={{position:"absolute", right:8, bottom:8, fontSize:12, opacity:0.85}}>
        消息沿边传播 = 一次 GCN 聚合
      </div>
    </div>
  );
}
