// src/pages/FeedbackPage.js
import React, { useState } from "react";
import { Star, Send, MessageSquare, Loader2, ThumbsUp } from "lucide-react";
import api from "../api/axios";

const FeedbackPage = () => {
  const [rating, setRating] = useState(0);
  const [hoverRating, setHoverRating] = useState(0);
  const [message, setMessage] = useState("");
  
  // Safely get user name or default to empty
  const [name, setName] = useState(() => {
    try {
      const userStr = localStorage.getItem("user");
      return userStr ? JSON.parse(userStr).name : "";
    } catch (e) {
      return "";
    }
  });

  const [status, setStatus] = useState({ type: "", text: "" });
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (rating === 0) {
      setStatus({ type: "error", text: "Please select a star rating." });
      return;
    }

    setIsSubmitting(true);
    setStatus({ type: "", text: "" });

    try {
      // Uses the configured Axios instance (Port 5000, Auth Headers)
      await api.post("/feedback", { name, rating, message });

      setStatus({ type: "success", text: "Thank you! Your feedback helps us improve NeuroFlow." });
      setRating(0);
      setMessage("");
      
      // Clear success message after 3 seconds
      setTimeout(() => setStatus({ type: "", text: "" }), 3000);

    } catch (err) {
      const errorMsg = err.response?.data?.message || "Server connection failed. Please try again.";
      setStatus({ type: "error", text: errorMsg });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen pt-20 pb-12 px-4 bg-slate-950 flex items-center justify-center">
      
      {/* Background Decor */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-emerald-500/5 rounded-full blur-3xl"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl"></div>
      </div>

      <div className="max-w-lg w-full bg-slate-900/80 backdrop-blur-xl border border-slate-800 rounded-2xl p-8 relative shadow-2xl">
        
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex p-3 rounded-xl bg-slate-800 border border-slate-700 mb-4 shadow-lg">
            <MessageSquare className="w-6 h-6 text-emerald-400" />
          </div>
          <h1 className="text-3xl font-bold bg-linear-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
            We Value Your Feedback
          </h1>
          <p className="text-slate-400 mt-2 text-sm">
            Help us make NeuroFlow the best biometrics platform for students.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          
          {/* Star Rating */}
          <div className="flex flex-col items-center gap-2">
            <label className="text-sm font-medium text-slate-300">Rate your experience</label>
            <div className="flex gap-2">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star}
                  type="button"
                  className="focus:outline-none transition-transform hover:scale-110 active:scale-95"
                  onMouseEnter={() => setHoverRating(star)}
                  onMouseLeave={() => setHoverRating(0)}
                  onClick={() => setRating(star)}
                >
                  <Star
                    className={`w-10 h-10 transition-colors duration-200 ${
                      star <= (hoverRating || rating)
                        ? "text-yellow-400 fill-yellow-400 drop-shadow-[0_0_8px_rgba(250,204,21,0.4)]"
                        : "text-slate-600 fill-transparent"
                    }`}
                  />
                </button>
              ))}
            </div>
            <div className="h-6 text-center">
              {rating > 0 && (
                <span className="text-sm text-yellow-400 font-medium animate-in fade-in slide-in-from-bottom-2">
                  {rating === 5 ? "Excellent! 🎉" : 
                   rating === 4 ? "Very Good! 🌟" :
                   rating === 3 ? "Average 😐" : 
                   rating === 2 ? "Poor 😞" : "Terrible 😫"}
                </span>
              )}
            </div>
          </div>

          {/* Name Field */}
          <div className="space-y-2">
            <label htmlFor="name" className="text-sm font-medium text-slate-300">
              Your Name
            </label>
            <input
              type="text"
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter your name"
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-3 text-slate-200 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition-all placeholder:text-slate-600"
            />
          </div>

          {/* Message Field */}
          <div className="space-y-2">
            <label htmlFor="message" className="text-sm font-medium text-slate-300">
              Your Message
            </label>
            <textarea
              id="message"
              rows="4"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Tell us what you liked or how we can improve..."
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-3 text-slate-200 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition-all placeholder:text-slate-600 resize-none"
            />
          </div>

          {/* Action Button */}
          <button
            type="submit"
            disabled={isSubmitting}
            className="w-full bg-linear-to-r from-emerald-500 to-cyan-600 hover:from-emerald-400 hover:to-cyan-500 text-white font-bold py-3.5 rounded-xl shadow-lg shadow-emerald-900/20 transition-all flex items-center justify-center gap-2 disabled:opacity-70 disabled:cursor-not-allowed"
          >
            {isSubmitting ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Submitting...
              </>
            ) : (
              <>
                <Send className="w-5 h-5" />
                Submit Feedback
              </>
            )}
          </button>

          {/* Status Message */}
          {status.text && (
            <div className={`p-4 rounded-lg flex items-center gap-3 text-sm ${
              status.type === 'success' 
                ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' 
                : 'bg-red-500/10 text-red-400 border border-red-500/20'
            }`}>
              {status.type === 'success' ? <ThumbsUp className="w-4 h-4" /> : null}
              {status.text}
            </div>
          )}

        </form>
      </div>
    </div>
  );
};

export default FeedbackPage;