#!/usr/bin/env node

import { createServer } from 'http';
import worker from './index.js';
const PORT = process.env.PORT || 3000;

// Mock environment from .env
const env = {
  TOKEN_POOL_SIZE: process.env.TOKEN_POOL_SIZE || '5',
  API_KEY: process.env.API_KEY || 'sk-z2api-key-2024',
  SHOW_THINK_TAGS: process.env.SHOW_THINK_TAGS || 'true'
};

const server = createServer(async (req, res) => {
  try {
    // Convert Node.js request to Cloudflare Workers Request
    const url = `http://${req.headers.host}${req.url}`;
    let body = null;
    
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      const chunks = [];
      for await (const chunk of req) {
        chunks.push(chunk);
      }
      body = Buffer.concat(chunks).toString();
    }
    
    const request = new Request(url, {
      method: req.method,
      headers: req.headers,
      body: body
    });
    
    // Call worker
    const response = await worker.fetch(request, env);
    
    // Convert Cloudflare Workers Response to Node.js response
    res.statusCode = response.status;
    
    for (const [key, value] of response.headers) {
      res.setHeader(key, value);
    }
    
    if (response.body) {
      const reader = response.body.getReader();
      const pump = async () => {
        const { done, value } = await reader.read();
        if (done) {
          res.end();
          return;
        }
        res.write(value);
        pump();
      };
      pump();
    } else {
      res.end();
    }
  } catch (error) {
    console.error('Server error:', error);
    res.statusCode = 500;
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify({ error: 'Internal server error' }));
  }
});

server.listen(PORT, () => {
  console.log(`🚀 Z.AI Proxy Server running on http://localhost:${PORT}`);
  console.log(`📋 Available endpoints:`);
  console.log(`   GET  /health                   - Health check`);
  console.log(`   GET  /v1/models                - List models`);
  console.log(`   POST /v1/chat/completions      - Chat completions`);
  console.log(`🔑 API Key: ${env.API_KEY}`);
  console.log(`🎯 Token pool size: ${env.TOKEN_POOL_SIZE}`);
  console.log(`🔄 Load balancing: Enabled for token pool`);
});

process.on('SIGINT', () => {
  console.log('\n👋 Shutting down server...');
  server.close(() => {
    process.exit(0);
  });
});
