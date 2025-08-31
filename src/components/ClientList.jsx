\
import React from "react";
import { motion } from "framer-motion";

export default function ClientList({ clients=[] }){
  return (
    <div className="clients">
      {clients.map(c=>(
        <motion.div key={c.id} className="client-item"
          initial={{opacity:0, y:8}} animate={{opacity:1, y:0}}
          transition={{duration:0.3}}
        >
          <div>客户端 #{c.id}</div>
          <div>状态：{c.status}</div>
        </motion.div>
      ))}
    </div>
  );
}
