const asyncHandler = require('express-async-handler');
const Session = require('../models/session');

const createSession = asyncHandler(async (req, res) => {
  const { 
    durationMinutes, 
    averageHeartRate, 
    attentionScore, 
    stressLevel, 
    heartRateData 
  } = req.body;

  if (attentionScore === undefined || !averageHeartRate) {
    res.status(400);
    throw new Error('Invalid session data: Missing required metrics');
  }

  const session = await Session.create({
    user: req.user.id,
    durationMinutes,
    averageHeartRate,
    attentionScore,
    stressLevel,
    heartRateData
  });

  res.status(201).json(session);
});

const getSessions = asyncHandler(async (req, res) => {
  const sessions = await Session.find({ user: req.user.id })
    .select('-heartRateData') 
    .sort({ createdAt: -1 })
    .lean();
    
  res.status(200).json(sessions);
});

const getSessionDetails = asyncHandler(async (req, res) => {
  const session = await Session.findById(req.params.id).lean();
  
  if (!session) {
    res.status(404);
    throw new Error('Session not found');
  }

  if (session.user.toString() !== req.user.id) {
    res.status(401);
    throw new Error('Not authorized');
  }

  res.status(200).json(session);
});

module.exports = { createSession, getSessions, getSessionDetails };