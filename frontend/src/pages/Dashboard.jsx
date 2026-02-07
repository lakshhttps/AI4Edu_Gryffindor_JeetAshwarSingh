import React, { useEffect, useRef, useState } from 'react';
import { Activity, Brain, StopCircle, Zap, Radio, ScanFace } from 'lucide-react';
import { LineChart, Line, ResponsiveContainer, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { useNavigate } from 'react-router-dom';
import api from '../api/axios';

const Dashboard = () => {
  const videoRef = useRef(null);
  const navigate = useNavigate();
  
  // State
  const [heartRate, setHeartRate] = useState(0);
  const [attentionScore, setAttentionScore] = useState(88);
  const [dataPoints, setDataPoints] = useState(new Array(40).fill({ value: 50 })); 
  const [isRecording, setIsRecording] = useState(true);

  // --- VIDEO & SIMULATION LOGIC ---
  useEffect(() => {
    // Note: Video is now handled by the <video> tag in JSX
    
    const interval = setInterval(() => {
      if (!isRecording) return;
      
      // Simulation Logic (Soon to be replaced by ML Socket Data)
      setHeartRate(prev => {
        const change = Math.random() > 0.5 ? 1 : -1;
        return Math.min(110, Math.max(65, (prev || 72) + change));
      });
      
      setAttentionScore(prev => Math.min(100, Math.max(60, prev + (Math.random() - 0.4) * 3)));
      
      setDataPoints(prev => {
        const newPoints = [...prev, { value: Math.random() * 40 + 40 }];
        return newPoints.length > 40 ? newPoints.slice(1) : newPoints;
      });
    }, 200); 

    return () => clearInterval(interval);
  }, [isRecording]);

  // --- END SESSION HANDLER ---
  const handleEndSession = async () => {
    setIsRecording(false);
    
    const stress = heartRate > 95 ? "High" : heartRate > 80 ? "Moderate" : "Low";

    const sessionData = {
      durationMinutes: 45, 
      averageHeartRate: heartRate, 
      attentionScore: Math.floor(attentionScore),
      stressLevel: stress,
      heartRateData: dataPoints.map(p => p.value)
    };

    try {
      await api.post('/sessions', sessionData);
      navigate('/insights');
    } catch (error) {
      console.error("Failed to save session", error);
      navigate('/insights'); 
    }
  };

  return (
    <div className="min-h-screen pt-24 px-4 pb-8 max-w-7xl mx-auto">
      
      {/* Page Header */}
      <div className="flex items-center justify-between mb-6 px-2">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight flex items-center gap-3">
             <Activity className="text-emerald-500" /> 
             Live Biometrics Analysis
          </h1>
          <p className="text-slate-400 text-sm font-mono mt-1">SOURCE: /public/demo_video.mp4</p>
        </div>
        <div className="flex items-center gap-3 bg-slate-900 border border-slate-700 px-4 py-2 rounded-full">
           <div className={`w-2.5 h-2.5 rounded-full ${isRecording ? 'bg-red-500 animate-pulse' : 'bg-slate-500'}`} />
           <span className="text-xs font-mono text-slate-300 tracking-wider">
             {isRecording ? 'LIVE FEED ACTIVE' : 'SESSION PAUSED'}
           </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 min-h-150">
        
        {/* --- LEFT COLUMN: VIDEO & CHART --- */}
        <div className="lg:col-span-2 flex flex-col gap-6 h-full">
          
          {/* 1. Recorded Video Feed with HUD */}
          <div className="flex-1 relative bg-black rounded-3xl border border-slate-800 overflow-hidden shadow-2xl group min-h-100">
             {/* Scanlines Effect */}
             <div className="absolute inset-0 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.1)_50%),linear-gradient(90deg,rgba(255,0,0,0.03),rgba(0,255,0,0.01),rgba(0,0,255,0.03))] z-20 pointer-events-none bg-size-[100%_2px,3px_100%] opacity-40"></div>
             
             {/* THE VIDEO PLAYER (Replace demo_video.mp4 with your filename) */}
             <video 
               ref={videoRef}
               src="/demo_video.mp4" 
               autoPlay 
               muted 
               loop 
               playsInline
               className="w-full h-full object-cover opacity-90 scale-x-[-1]" 
             />

             {/* HUD Elements */}
             <div className="absolute top-4 left-4 z-30 bg-black/60 backdrop-blur-md border border-white/10 px-3 py-1 rounded-full flex items-center gap-2">
               <ScanFace className="w-4 h-4 text-emerald-400" />
               <span className="text-[10px] text-emerald-400 font-mono tracking-widest uppercase">rPPG_Tracking_Active</span>
             </div>

             {/* UI Scan Box */}
             {isRecording && (
                <div className="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
                  <div className="w-56 h-72 border border-emerald-500/20 rounded-3xl relative shadow-[0_0_50px_rgba(16,185,129,0.1)]">
                      <div className="absolute top-0 left-1/2 -translate-x-1/2 -mt-1 w-20 h-1 bg-emerald-500 shadow-[0_0_10px_#10b981]"></div>
                      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 -mb-1 w-20 h-1 bg-emerald-500 shadow-[0_0_10px_#10b981]"></div>
                  </div>
                </div>
             )}
          </div>
          
          {/* 2. Real-time Chart */}
          <div className="h-64 bg-slate-900/60 backdrop-blur-xl border border-slate-800 rounded-3xl p-6 relative">
             <div className="flex justify-between items-center mb-2">
               <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest flex items-center gap-2">
                 <Radio className="w-4 h-4 text-emerald-500" />
                 Signal Output
               </h3>
               <span className="text-[10px] font-mono text-emerald-500 bg-emerald-500/10 px-2 py-1 rounded">STABLE_SIGNAL</span>
             </div>
             
             <div className="w-full h-full pb-6">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={dataPoints}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <YAxis domain={[0, 100]} hide />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#fff' }} 
                      itemStyle={{ color: '#10b981' }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="value" 
                      stroke="#10B981" 
                      strokeWidth={3} 
                      dot={false} 
                      isAnimationActive={false} 
                    />
                  </LineChart>
                </ResponsiveContainer>
             </div>
          </div>
        </div>

        {/* --- RIGHT COLUMN: ANALYTICS --- */}
        <div className="flex flex-col gap-4 h-full">
          
          {/* Heart Rate Card */}
          <div className="flex-1 bg-slate-900/60 backdrop-blur-xl border border-slate-800 rounded-3xl p-8 flex flex-col justify-center relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-32 bg-emerald-500/5 rounded-full blur-3xl -mr-16 -mt-16"></div>
            <div className="relative z-10">
               <div className="flex items-center gap-3 mb-4">
                 <div className="p-3 bg-slate-800 rounded-xl group-hover:bg-slate-700 transition-colors">
                   <Activity className="w-6 h-6 text-emerald-500" />
                 </div>
                 <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Real-time Heart Rate</span>
               </div>
               <span className="text-6xl font-mono font-bold text-white tracking-tighter">{heartRate}</span>
               <span className="text-xl text-slate-500 font-medium ml-2 uppercase">Bpm</span>
            </div>
          </div>

          {/* Attention Score Card */}
          <div className="flex-1 bg-slate-900/60 backdrop-blur-xl border border-slate-800 rounded-3xl p-8 flex flex-col justify-center relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-32 bg-indigo-500/5 rounded-full blur-3xl -mr-16 -mt-16"></div>
            <div className="relative z-10">
               <div className="flex items-center gap-3 mb-4">
                 <div className="p-3 bg-slate-800 rounded-xl group-hover:bg-slate-700 transition-colors">
                   <Brain className="w-6 h-6 text-indigo-500" />
                 </div>
                 <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Attention Level</span>
               </div>
               <span className="text-6xl font-mono font-bold text-white tracking-tighter">{Math.floor(attentionScore)}</span>
               <span className="text-xl text-slate-500 font-medium ml-2">%</span>
            </div>
          </div>

          {/* Stress Level Box */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-3xl p-6 flex items-center justify-between">
              <div className="flex items-center gap-3">
                 <Zap className="w-5 h-5 text-amber-400" />
                 <span className="text-sm font-bold text-slate-400 uppercase">Stress Indicator</span>
              </div>
              <span className="text-xl font-mono font-bold text-white">LOW</span>
          </div>

          {/* Action Button */}
          <button 
            onClick={handleEndSession}
            className="mt-2 w-full py-5 bg-linear-to-r from-red-600 to-red-500 hover:from-red-500 hover:to-red-400 text-white font-bold rounded-2xl flex items-center justify-center gap-3 shadow-lg shadow-red-500/20 transition-all transform active:scale-[0.98] border border-red-400/20"
          >
              <StopCircle className="w-6 h-6" />
              <span className="tracking-wide">TERMINATE SESSION</span>
          </button>

        </div>
      </div>
    </div>
  );
};

export default Dashboard;