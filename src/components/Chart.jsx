\
import React, { useMemo } from "react";
import ReactECharts from "echarts-for-react";

export default function Chart({ rounds, acc, loss, metrics }){
  const marks = useMemo(()=>{
    // 标记临界点
    const points = [];
    (metrics||[]).forEach(m=>{
      if (m.critical) points.push({ name:"临界点", xAxis:m.round, yAxis:m.accuracy, value:`Δ=${m.delta.toFixed(3)}` });
    });
    return points;
  }, [metrics]);

  const option = {
    backgroundColor: "transparent",
    tooltip: { trigger:'axis' },
    legend: { data:["准确率","损失"] },
    xAxis: { type:"category", data: rounds, axisLine:{lineStyle:{color:"#fff"}} },
    yAxis: [
      { type:"value", name:"准确率", min:0, max:1, axisLine:{lineStyle:{color:"#fff"}} },
      { type:"value", name:"损失", axisLine:{lineStyle:{color:"#fff"}} },
    ],
    series: [
      { name:"准确率", type:"line", smooth:true, data: acc, yAxisIndex:0,
        markPoint:{ data: marks } },
      { name:"损失", type:"line", smooth:true, data: loss, yAxisIndex:1 },
    ],
  };
  return <ReactECharts option={option} style={{ height: 380 }} />;
}
