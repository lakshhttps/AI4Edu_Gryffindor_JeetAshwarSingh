import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { ArrowRight, Lock, Mail, User, Building } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const Register = () => {
  const [formData, setFormData] = useState({ name: '', email: '', password: '', institution: '' });
  const [error, setError] = useState('');
  const { register } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await register(formData.name, formData.email, formData.password, formData.institution);
      navigate('/dashboard');
    } catch (err) {
      setError('Registration failed. Email might be taken.');
    }
  };

  return (
    <div className="min-h-screen bg-[#020617] flex items-center justify-center p-4 relative">
       <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-indigo-500/10 rounded-full blur-[100px]" />
      
      <div className="relative w-full max-w-md bg-slate-900/60 backdrop-blur-xl border border-slate-800 p-8 rounded-2xl shadow-2xl">
        <h2 className="text-3xl font-bold text-white mb-6 text-center">Join the Lab</h2>
        
        {error && <div className="mb-4 p-3 bg-red-500/10 text-red-400 text-sm rounded-lg">{error}</div>}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="relative group">
            <User className="absolute left-3 top-3.5 h-5 w-5 text-slate-500" />
            <input 
              type="text" 
              placeholder="Full Name"
              required
              className="w-full bg-slate-950 border border-slate-800 rounded-xl py-3 pl-10 pr-4 text-white focus:border-emerald-500 focus:outline-none"
              onChange={(e) => setFormData({...formData, name: e.target.value})}
            />
          </div>

          <div className="relative group">
            <Mail className="absolute left-3 top-3.5 h-5 w-5 text-slate-500" />
            <input 
              type="email" 
              placeholder="Email Address"
              required
              className="w-full bg-slate-950 border border-slate-800 rounded-xl py-3 pl-10 pr-4 text-white focus:border-emerald-500 focus:outline-none"
              onChange={(e) => setFormData({...formData, email: e.target.value})}
            />
          </div>

          <div className="relative group">
            <Building className="absolute left-3 top-3.5 h-5 w-5 text-slate-500" />
            <input 
              type="text" 
              placeholder="Institution (e.g. IIT Ropar)"
              required
              className="w-full bg-slate-950 border border-slate-800 rounded-xl py-3 pl-10 pr-4 text-white focus:border-emerald-500 focus:outline-none"
              onChange={(e) => setFormData({...formData, institution: e.target.value})}
            />
          </div>

          <div className="relative group">
            <Lock className="absolute left-3 top-3.5 h-5 w-5 text-slate-500" />
            <input 
              type="password" 
              placeholder="Password"
              required
              className="w-full bg-slate-950 border border-slate-800 rounded-xl py-3 pl-10 pr-4 text-white focus:border-emerald-500 focus:outline-none"
              onChange={(e) => setFormData({...formData, password: e.target.value})}
            />
          </div>

          <button type="submit" className="w-full bg-linear-to-r from-emerald-600 to-emerald-500 text-white font-semibold py-3 rounded-xl flex items-center justify-center gap-2 mt-4">
            Create Account <ArrowRight className="h-4 w-4" />
          </button>
        </form>
        <div className="mt-6 text-center">
          <Link to="/login" className="text-sm text-slate-400 hover:text-emerald-400">Already have an account? Sign in</Link>
        </div>
      </div>
    </div>
  );
};

export default Register;