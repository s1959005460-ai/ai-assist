\
import axios from "axios";
const base = "http://127.0.0.1:8000";

export async function getMetrics(){
  const r = await axios.get(`${base}/metrics`);
  return r.data.metrics || [];
}

export async function startTrain(payload){
  const r = await axios.post(`${base}/train`, payload);
  return r.data;
}
