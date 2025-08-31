\
import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import Chart from "./components/Chart";
import ClientList from "./components/ClientList";
import StageAnimation from "./components/StageAnimation";
import MathAnimation from "./components/MathAnimation";
import { startTrain, getMetrics } from "./services/api";

export default function App(){
  const [metrics, setMetrics] = useState([]);        // [{round,accuracy,loss,entropy,critical,delta}]
  const [clients, setClients] = useState([{id:1,status:"等待"}, {id:2,status:"等待"}, {id:3,status:"等待"}]);
  const [stage, setStage] = useState("就绪");
  const wsRef = useRef(null);

  const accSeries = useMemo(()=>metrics.map(m=>m.accuracy), [metrics]);
  const lossSeries = useMemo(()=>metrics.map(m=>m.loss), [metrics]);
  const roundAxis = useMemo(()=>metrics.map(m=>m.round), [metrics]);

  useEffect(()=>{
    async function init() {
      const initM = await getMetrics();
      if (initM && initM.length) setMetrics(initM);
    }
    init();

    const ws = new WebSocket("ws://127.0.0.1:8000/ws/train");
    wsRef.current = ws;
    ws.onmessage = (evt)=>{
      const msg = JSON.parse(evt.data);
      if (msg.type === "round") {
        setMetrics(prev => [...prev, msg.record]);
        if (msg.clients) setClients(msg.clients);
      } else if (msg.type === "stage") {
        setStage(msg.stage);
      } else if (msg.type === "metrics") {
        setMetrics(msg.metrics || []);
      }
    };
    ws.onclose = ()=>{ /* ignore */ };
    return ()=> ws.close();
  },[]);

  const handleStart = async ()=>{
    setStage("准备开始");
    await startTrain({ dataset:"Cora", num_clients:3, rounds:8, epochs:5, hidden:16, lr:0.01 });
  };

  return (
    <div className="container">
      <div className="header">
        <motion.h1 initial={{y:-10,opacity:0}} animate={{y:0,opacity:1}} transition={{duration:0.6}}>
          FedGNN 仪表盘 <span className="badge">实时</span>
        </motion.h1>
        <button className="button" onClick={handleStart}>开始训练</button>
      </div>

      <motion.div className="card" initial={{opacity:0,y:10}} animate={{opacity:1,y:0}}>
        <div className="stage">程序阶段：<StageAnimation stage={stage} /></div>
      </motion.div>

      <motion.div className="card" initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} transition={{delay:0.05}}>
        <h2>客户端状态</h2>
        <ClientList clients={clients} />
      </motion.div>

      <motion.div className="card" initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} transition={{delay:0.1}}>
        <h2>训练指标</h2>
        <Chart rounds={roundAxis} acc={accSeries} loss={lossSeries} metrics={metrics} />
      </motion.div>

      <motion.div className="card" initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} transition={{delay:0.15}}>
        <h2>数学运算动画（GCN 消息传递演示）</h2>
        <MathAnimation />
      </motion.div>
    </div>
  );
}
