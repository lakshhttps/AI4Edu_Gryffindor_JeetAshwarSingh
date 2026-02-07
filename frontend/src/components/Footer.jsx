import React from 'react';

const Footer = () => {
  return (
    <footer className="bg-[#020617] border-t border-slate-800 py-8 mt-auto">
      <div className="max-w-7xl mx-auto px-4 text-center">
        <p className="text-slate-500 text-sm">
          © 2026 NeuroFlow Project. Built for IIT Ropar Hackathon.
        </p>
        <div className="flex justify-center items-center gap-3 mt-4 text-xs text-slate-600 font-mono">
          <span className="hover:text-emerald-500 transition-colors cursor-default">rPPG Engine v1.0</span>
          <span className="text-slate-800">•</span>
          <span className="hover:text-cyan-500 transition-colors cursor-default">MERN Stack</span>
          <span className="text-slate-800">•</span>
          <span className="hover:text-slate-400 transition-colors cursor-pointer">Privacy Policy</span>
        </div>
      </div>
    </footer>
  );
};

export default Footer;