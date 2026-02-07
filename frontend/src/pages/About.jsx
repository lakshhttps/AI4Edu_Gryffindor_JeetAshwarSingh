import React from 'react';
import { Shield, Code, Cpu } from 'lucide-react';

const About = () => {
  return (
    <div className="min-h-screen bg-[#020617] pt-24 px-4 pb-12">
      <div className="max-w-4xl mx-auto text-center space-y-12">
        <div>
          <h1 className="text-4xl font-bold text-white mb-4">Reinventing Focus with AI</h1>
          <p className="text-slate-400 text-lg">NeuroFlow uses non-invasive rPPG technology to democratize cognitive health monitoring.</p>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          <div className="p-6 bg-slate-900 rounded-2xl border border-slate-800">
            <Shield className="w-10 h-10 text-emerald-500 mx-auto mb-4" />
            <h3 className="text-white font-bold mb-2">Privacy First</h3>
            <p className="text-slate-500 text-sm">No video is ever stored. Processing happens in real-time.</p>
          </div>
          <div className="p-6 bg-slate-900 rounded-2xl border border-slate-800">
            <Cpu className="w-10 h-10 text-indigo-500 mx-auto mb-4" />
            <h3 className="text-white font-bold mb-2">Edge AI</h3>
            <p className="text-slate-500 text-sm">Powered by lightweight computer vision models.</p>
          </div>
          <div className="p-6 bg-slate-900 rounded-2xl border border-slate-800">
            <Code className="w-10 h-10 text-cyan-500 mx-auto mb-4" />
            <h3 className="text-white font-bold mb-2">Open Source</h3>
            <p className="text-slate-500 text-sm">Built for the research community.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;