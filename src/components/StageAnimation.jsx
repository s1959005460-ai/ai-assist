\
import React from "react";
import { motion, AnimatePresence } from "framer-motion";

export default function StageAnimation({ stage }){
  return (
    <AnimatePresence mode="wait">
      <motion.span key={stage}
        initial={{opacity:0, y:-6}}
        animate={{opacity:1, y:0}}
        exit={{opacity:0, y:6}}
        transition={{duration:0.35}}
        style={{padding:"2px 8px", background:"rgba(255,255,255,0.15)", borderRadius:8}}
      >
        {stage}
      </motion.span>
    </AnimatePresence>
  );
}
