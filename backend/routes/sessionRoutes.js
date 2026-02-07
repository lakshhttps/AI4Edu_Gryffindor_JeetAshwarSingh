const express = require('express');
const router = express.Router();
const { 
  createSession, 
  getSessions, 
  getSessionDetails 
} = require('../controllers/sessionControllers');
const { protect } = require('../middleware/authMiddleware');

router.route('/')
  .post(protect, createSession)
  .get(protect, getSessions);

router.route('/:id')
  .get(protect, getSessionDetails);

module.exports = router;