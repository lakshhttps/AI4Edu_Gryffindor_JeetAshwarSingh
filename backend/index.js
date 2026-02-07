const express = require('express');
const cors = require('cors');
const http = require('http');
const { Server } = require("socket.io");
const { spawn } = require('child_process');
require('dotenv').config();
const connectDB = require('./config/db');
const { errorHandler } = require('./middleware/errorMiddleware');

const port = process.env.PORT || 5000;

connectDB();

const app = express();
const server = http.createServer(app);

const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

app.use(cors({ origin: '*' }));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use('/api/users', require('./routes/userRoutes'));
app.use('/api/sessions', require('./routes/sessionRoutes'));

app.use(errorHandler);

// --- UPDATED PARSER ---
const parseMLOutput = (data) => {
  const str = data.toString().trim();
  if (!str) return null;

  try {
    // We expect the ML Lead to print JSON strings
    return JSON.parse(str);
  } catch (e) {
    console.log("Non-JSON Python output:", str);
    return null;
  }
};

// --- AUTO-RESTARTING PYTHON PROCESS ---
let pythonProcess = null;

const startPythonInference = () => {
  console.log("🚀 Launching ML Inference Engine...");
  
  // Updated to inference.py as per your ML lead's instructions
  pythonProcess = spawn('python', ['-u', 'inference.py']);

  pythonProcess.stdout.on('data', (data) => {
    const cleanData = parseMLOutput(data);
    if (cleanData) {
      // Send data to React Dashboard instantly
      io.emit('dashboard_update', {
        heartRate: cleanData.heart_rate,
        status: cleanData.status,
        rppg: cleanData.rppg,
        timestamp: new Date().toLocaleTimeString()
      });
    }
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`[ML Error]: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}. Restarting...`);
    setTimeout(startPythonInference, 2000); // Restart after 2 seconds
  });
};

// Start the engine
startPythonInference();

server.listen(port, () => {
    console.log(`-----------------------------------------`);
    console.log(`✅ NeuroFlow Backend Active on Port ${port}`);
    console.log(`📡 Socket.io Ready for Dashboard Updates`);
    console.log(`-----------------------------------------`);
});