import React, { useEffect, useState } from 'react';
import { Clock, Brain, Activity, Calendar, TrendingUp, BarChart3, AlertCircle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Area, AreaChart } from 'recharts';
import api from '../api/axios';

const Insights = () => {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    totalSessions: 0,
    avgFocus: 0,
    totalMinutes: 0
  });

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const { data } = await api.get('/sessions');
        setSessions(data);

        // Calculate Summary Stats
        if (data.length > 0) {
          const totalMins = data.reduce((acc, curr) => acc + (curr.durationMinutes || 0), 0);
          const avgAtt = data.reduce((acc, curr) => acc + (curr.attentionScore || 0), 0) / data.length;
          
          setStats({
            totalSessions: data.length,
            avgFocus: Math.round(avgAtt),
            totalMinutes: totalMins
          });
        }
      } catch (error) {
        console.error("Error fetching sessions", error);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, []);

  // Format data for the chart (Reverse so oldest is left, newest is right)
  const chartData = [...sessions].reverse().map((s, index) => ({
    name: `S${index + 1}`,
    focus: s.attentionScore,
    date: new Date(s.createdAt).toLocaleDateString()
  }));

  if (loading) return (
    <div className="min-h-screen bg-[#020617] pt-24 flex items-center justify-center text-emerald-500">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald-500"></div>
    </div>
  );

  return (
    <div className="min-h-screen bg-[#020617] pt-24 px-4 pb-12">
      <div className="max-w-7xl mx-auto space-y-8">
        
        {/* HEADER */}
        <div>
          <h1 className="text-3xl font-bold mb-2 bg-linear-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
            Performance Analytics
          </h1>
          <p className="text-slate-400 text-sm">Track your cognitive focus and stress trends over time.</p>
        </div>

        {/* TOP STATS CARDS */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <StatCard 
            icon={<Brain className="text-indigo-400" />} 
            label="Average Focus" 
            value={`${stats.avgFocus}%`} 
            sub="Global Average"
            color="indigo"
          />
          <StatCard 
            icon={<Clock className="text-emerald-400" />} 
            label="Total Training" 
            value={`${stats.totalMinutes}m`} 
            sub={`${stats.totalSessions} Sessions Completed`}
            color="emerald"
          />
          <StatCard 
            icon={<Activity className="text-pink-400" />} 
            label="Consistency" 
            value="High" 
            sub="Last 7 Days"
            color="pink"
          />
        </div>

        {sessions.length === 0 ? (
          <div className="text-slate-500 bg-slate-900/50 p-12 rounded-xl border border-slate-800 text-center flex flex-col items-center gap-4">
            <div className="p-4 bg-slate-800 rounded-full"><BarChart3 className="w-8 h-8 opacity-50" /></div>
            <p>No training sessions recorded yet.</p>
            <p className="text-sm">Start a "New Session" from the dashboard to see analytics.</p>
          </div>
        ) : (
          <>
            {/* CHART SECTION */}
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 h-80 relative overflow-hidden">
              <h3 className="text-slate-200 font-semibold mb-6 flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-emerald-500" /> Focus Trend
              </h3>
              <ResponsiveContainer width="100%" height="85%">
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="colorFocus" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="name" stroke="#64748b" tick={{fontSize: 12}} />
                  <YAxis stroke="#64748b" tick={{fontSize: 12}} domain={[0, 100]} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#fff' }}
                    labelStyle={{ color: '#94a3b8' }}
                  />
                  <Area type="monotone" dataKey="focus" stroke="#10b981" fillOpacity={1} fill="url(#colorFocus)" strokeWidth={3} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* RECENT SESSIONS LIST */}
            <div className="space-y-4">
              <h3 className="text-slate-200 font-semibold pl-1">Recent Sessions</h3>
              <div className="grid gap-3">
                {sessions.map((session) => (
                  <div key={session._id} className="bg-slate-900/80 border border-slate-800 p-5 rounded-xl hover:border-emerald-500/30 transition-all group flex flex-col md:flex-row justify-between items-center gap-4">
                    
                    {/* Left: Date & ID */}
                    <div className="flex items-center gap-4 w-full md:w-auto">
                      <div className="p-3 bg-slate-800 rounded-lg group-hover:bg-slate-700 transition-colors">
                        <Calendar className="h-5 w-5 text-slate-400" />
                      </div>
                      <div>
                        <p className="text-slate-200 font-medium">{new Date(session.createdAt).toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' })}</p>
                        <p className="text-xs text-slate-500 font-mono">{new Date(session.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</p>
                      </div>
                    </div>

                    {/* Middle: Metrics */}
                    <div className="flex flex-wrap justify-center gap-6 w-full md:w-auto bg-slate-950/30 py-2 px-4 rounded-lg border border-white/5">
                      <div className="flex items-center gap-2">
                        <Clock className="w-3 h-3 text-emerald-500" />
                        <span className="text-sm font-mono text-slate-300">{session.durationMinutes}m</span>
                      </div>
                      <div className="w-px h-4 bg-slate-700"></div>
                      <div className="flex items-center gap-2">
                        <Activity className="w-3 h-3 text-pink-500" />
                        <span className="text-sm font-mono text-slate-300">{session.averageHeartRate} BPM</span>
                      </div>
                      <div className="w-px h-4 bg-slate-700"></div>
                      <div className="flex items-center gap-2">
                        <Brain className="w-3 h-3 text-indigo-500" />
                        <span className="text-sm font-mono text-slate-300">{session.attentionScore}%</span>
                      </div>
                    </div>

                    {/* Right: Badge */}
                    <div className={`px-3 py-1 rounded-full text-[10px] font-bold tracking-wider uppercase border ${
                      session.stressLevel === 'High' 
                        ? 'bg-red-500/10 border-red-500/20 text-red-400' 
                        : session.stressLevel === 'Moderate'
                        ? 'bg-yellow-500/10 border-yellow-500/20 text-yellow-400'
                        : 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                    }`}>
                      {session.stressLevel} Stress
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

// Simple reusable Stat Card Component
const StatCard = ({ icon, label, value, sub, color }) => (
  <div className="bg-slate-900/60 border border-slate-800 p-6 rounded-xl hover:bg-slate-800/50 transition-colors">
    <div className="flex justify-between items-start mb-4">
      <div className={`p-2 rounded-lg bg-${color}-500/10`}>
        {icon}
      </div>
      {/* Small trend indicator (static for now) */}
      <span className="text-xs font-medium text-emerald-400 flex items-center gap-1">
        +2.5% <TrendingUp className="w-3 h-3" />
      </span>
    </div>
    <div className="text-3xl font-bold text-white mb-1">{value}</div>
    <div className="text-sm text-slate-400">{label}</div>
    <div className="text-xs text-slate-600 mt-2">{sub}</div>
  </div>
);

export default Insights;