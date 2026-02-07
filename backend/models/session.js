const mongoose = require('mongoose');

const sessionSchema = mongoose.Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    required: true,
    ref: 'User',
    index: true
  },
  durationMinutes: { 
    type: Number, 
    required: true,
    min: 0 
  },
  averageHeartRate: { 
    type: Number, 
    required: true,
    min: 0 
  },
  attentionScore: { 
    type: Number, 
    required: true,
    min: 0,
    max: 100 
  },
  stressLevel: {
    type: String,
    enum: ['Low', 'Moderate', 'High'], 
    required: true
  },
  heartRateData: { 
    type: [Number],
    default: [] 
  }
}, {
  timestamps: true
});

module.exports = mongoose.model('Session', sessionSchema);