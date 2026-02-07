import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import PrivateRoute from './components/PrivateRoute';

// Pages
import Login from './pages/login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import Insights from './pages/Insights';
import FeedbackPage from './pages/Feedback';
import About from './pages/About';

function App() {
  return (
    <AuthProvider>
      <Router>
        {/* Main Layout Wrapper */}
        <div className="bg-[#020617] min-h-screen text-slate-200 flex flex-col font-sans selection:bg-emerald-500/30">
          
          <Navbar />
          
          <main className="grow relative">
            {/* Global Ambient Background Effects */}
            <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
                <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-emerald-500/5 rounded-full blur-[120px]" />
                <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-indigo-500/5 rounded-full blur-[120px]" />
            </div>

            <div className="relative z-10">
              <Routes>
                {/* Public Routes */}
                <Route path="/login" element={<Login />} />
                <Route path="/register" element={<Register />} />
                <Route path="/about" element={<About />} />

                {/* Protected Routes (Require Login) */}
                <Route element={<PrivateRoute />}>
                  {/* Redirect root to dashboard if logged in */}
                  <Route path="/" element={<Navigate to="/dashboard" replace />} />
                  
                  <Route path="/dashboard" element={<Dashboard />} />
                  <Route path="/insights" element={<Insights />} />
                  <Route path="/feedback" element={<FeedbackPage />} />
                </Route>

                {/* Catch All - Redirect to Login */}
                <Route path="*" element={<Navigate to="/login" replace />} />
              </Routes>
            </div>
          </main>

          <Footer />
        </div>
      </Router>
    </AuthProvider>
  );
}

export default App;