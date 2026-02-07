import React from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Activity, LogOut, LayoutDashboard, FileText, Info, MessageSquare } from 'lucide-react'; // Added MessageSquare Icon
import { useAuth } from '../context/AuthContext';

const Navbar = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const getLinkClass = (path) => {
    const baseClass = "flex items-center gap-2 px-4 py-2 rounded-full border transition-all text-sm font-medium";
    const activeClass = "text-emerald-400 bg-emerald-500/10 border-emerald-500/50 shadow-[0_0_15px_-3px_rgba(16,185,129,0.2)]";
    const inactiveClass = "text-slate-400 hover:text-slate-200 border-transparent hover:bg-slate-800/50";
    
    return `${baseClass} ${location.pathname === path ? activeClass : inactiveClass}`;
  };

  return (
    <nav className="fixed top-0 w-full z-50 bg-[#020617]/80 backdrop-blur-xl border-b border-slate-800 shadow-2xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* LOGO */}
          <Link to="/" className="flex items-center gap-3 group">
            <div className="p-2 bg-linear-to-br from-emerald-500/20 to-cyan-500/20 rounded-xl border border-emerald-500/20 group-hover:border-emerald-500/50 transition-all">
              <Activity className="h-6 w-6 text-emerald-400" />
            </div>
            <span className="text-xl font-bold bg-linear-to-r from-emerald-400 via-teal-200 to-cyan-400 bg-clip-text text-transparent tracking-tight">
              NeuroFlow
            </span>
          </Link>

          {/* CENTER LINKS (Added Feedback Here) */}
          {user && (
            <div className="hidden md:flex items-center gap-2">
              <Link to="/dashboard" className={getLinkClass('/dashboard')}>
                <LayoutDashboard className="h-4 w-4" />
                Monitor
              </Link>
              <Link to="/insights" className={getLinkClass('/insights')}>
                <FileText className="h-4 w-4" />
                Insights
              </Link>
              {/* NEW FEEDBACK LINK */}
              <Link to="/feedback" className={getLinkClass('/feedback')}>
                <MessageSquare className="h-4 w-4" />
                Feedback
              </Link>
              <Link to="/about" className={getLinkClass('/about')}>
                <Info className="h-4 w-4" />
                About
              </Link>
            </div>
          )}

          {/* RIGHT SIDE */}
          <div className="flex items-center gap-6">
            {user ? (
              <>
                <div className="hidden lg:flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900/80 border border-slate-700/50">
                   <span className="relative flex h-2 w-2">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                   </span>
                   <span className="text-[10px] text-slate-400 font-mono tracking-widest uppercase">System Online</span>
                </div>

                <div className="flex items-center gap-4 pl-6 border-l border-slate-800/50">
                  <div className="text-right hidden sm:block">
                    <p className="text-sm font-medium text-slate-200">{user.name}</p>
                    <p className="text-xs text-slate-500">Admin Access</p>
                  </div>
                  
                  <button 
                    onClick={handleLogout} 
                    className="p-2 text-slate-400 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-all"
                    title="Logout"
                  >
                    <LogOut className="h-5 w-5" />
                  </button>
                </div>
              </>
            ) : (
              <Link 
                to="/login" 
                className="px-5 py-2 text-sm font-medium text-emerald-950 bg-emerald-400 hover:bg-emerald-300 rounded-full transition-colors shadow-[0_0_20px_-5px_rgba(52,211,153,0.4)]"
              >
                Sign In
              </Link>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;