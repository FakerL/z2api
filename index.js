/**
 * Z.AI Proxy - OpenAI-compatible API for Z.AI
 * Single file Cloudflare Workers implementation
 */

// Configuration
function getSettings(env = {}) {
  return {
    UPSTREAM_URL: 'https://chat.z.ai/api/chat/completions',
    UPSTREAM_MODELS: {
      'glm-4.5': '0727-360B-API',
      'glm-4.5-air': '0727-106B-API',
    },
    API_KEY: env.API_KEY || 'sk-z2api-key-2024',
    TOKEN_POOL_SIZE: env.TOKEN_POOL_SIZE !== undefined ? parseInt(env.TOKEN_POOL_SIZE) : 5,
  };
}

// Token Manager - Simple token pool with load balancing
class TokenManager {
  constructor(poolSize) {
    this.poolSize = poolSize;
    this.tokens = []; // Array of { token, lastUsed, failures }
    this.currentIndex = 0;
    this.maxFailures = 3;
    this.failureResetTime = 300000; // 5 minutes
    this.tokenCacheDuration = 3600000; // 1 hour
    this.initialized = false;
  }

  getCommonHeaders() {
    return {
      'Accept': '*/*',
      'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'Content-Type': 'application/json',
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
      'sec-ch-ua': '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
      'sec-ch-ua-mobile': '?0',
      'sec-ch-ua-platform': '"Windows"',
      'Origin': 'https://chat.z.ai',
      'Referer': 'https://chat.z.ai/',
      'Sec-Fetch-Dest': 'empty',
      'Sec-Fetch-Mode': 'cors',
      'Sec-Fetch-Site': 'same-origin'
    };
  }

  async fetchToken() {
    try {
      const response = await fetch('https://chat.z.ai/api/v1/auths/', {
        method: 'GET',
        headers: {
          ...this.getCommonHeaders(),
          'Pragma': 'no-cache',
        }
      });

      if (!response.ok) {
        throw new Error(`Token fetch failed: ${response.status} - ${await response.text()}`);
      }

      const data = await response.json();
      return data.token;
    } catch (error) {
      console.error('Token fetch failed:', error);
      throw error;
    }
  }

  async initializeTokenPool() {
    if (this.initialized) return;
    
    console.log(`Initializing token pool with ${this.poolSize} tokens...`);
    const tokenPromises = [];
    
    for (let i = 0; i < this.poolSize; i++) {
      tokenPromises.push(this.fetchToken().catch(error => {
        console.warn(`Failed to fetch token ${i + 1}:`, error.message);
        return null;
      }));
    }
    
    const tokens = await Promise.all(tokenPromises);
    const now = Date.now();
    
    this.tokens = tokens
      .filter(token => token !== null)
      .map(token => ({
        token,
        lastUsed: 0,
        failures: 0,
        createdAt: now
      }));
    
    console.log(`Successfully initialized ${this.tokens.length}/${this.poolSize} tokens`);
    this.initialized = true;
  }

  async getValidToken() {
    if (!this.initialized) {
      await this.initializeTokenPool();
    }

    if (this.tokens.length === 0) {
      throw new Error('No tokens available');
    }

    const now = Date.now();
    
    // Find available tokens (reset failures after timeout)
    const availableTokens = this.tokens.filter(tokenInfo => {
      if (tokenInfo.failures >= this.maxFailures && 
          now - (tokenInfo.lastFailure || 0) > this.failureResetTime) {
        tokenInfo.failures = 0;
        delete tokenInfo.lastFailure;
      }
      return tokenInfo.failures < this.maxFailures;
    });

    if (availableTokens.length === 0) {
      throw new Error('All tokens have exceeded failure threshold');
    }

    // Sort by last used time for load balancing (least recently used first)
    availableTokens.sort((a, b) => a.lastUsed - b.lastUsed);
    
    const selectedToken = availableTokens[0];
    selectedToken.lastUsed = now;
    
    return selectedToken.token;
  }

  recordFailure(failedToken) {
    const tokenInfo = this.tokens.find(t => t.token === failedToken);
    if (tokenInfo) {
      tokenInfo.failures = (tokenInfo.failures || 0) + 1;
      tokenInfo.lastFailure = Date.now();
      console.warn(`Recorded token failure (${tokenInfo.failures}/${this.maxFailures})`);
    }
  }

  async refreshToken(failedToken) {
    const tokenInfo = this.tokens.find(t => t.token === failedToken);
    if (tokenInfo) {
      this.recordFailure(failedToken);
      
      try {
        const newToken = await this.fetchToken();
        tokenInfo.token = newToken;
        tokenInfo.failures = 0;
        tokenInfo.lastUsed = Date.now();
        tokenInfo.createdAt = Date.now();
        delete tokenInfo.lastFailure;
        console.log('Refreshed token');
        return newToken;
      } catch (error) {
        console.error('Failed to refresh token:', error);
        this.recordFailure(failedToken);
      }
    }
    return null;
  }

  getTokensStatus() {
    return this.tokens.map((tokenInfo, index) => ({
      id: index,
      hasToken: !!tokenInfo.token,
      failures: tokenInfo.failures || 0,
      lastUsed: tokenInfo.lastUsed || 0,
      createdAt: tokenInfo.createdAt || 0,
      isAvailable: (tokenInfo.failures || 0) < this.maxFailures
    }));
  }
}

// Request Handler - Simplified to work with token pool
class RequestHandler {
  constructor(tokenManager) {
    this.tokenManager = tokenManager;
  }

  async getNextToken() {
    return await this.tokenManager.getValidToken();
  }

  async handleTokenFailure(failedToken) {
    return await this.tokenManager.refreshToken(failedToken);
  }

  markTokenSuccess(token) {
    // Token success is already tracked in TokenManager via lastUsed
  }
}

// Proxy Handler
class ProxyHandler {
  constructor(requestHandler, settings) {
    this.requestHandler = requestHandler;
    this.settings = settings;
  }

  cleanThinkingContent(text) {
    if (!text) return '';
    
    return text.replace(/<glm_block.*?<\/glm_block>/gs, '')
        .replace(/^<details.+summary>/gs, '')
        .replace(/\n> /g, '\n')
	.trim();
  }

  cleanAnswerContent(text) {
    if (!text) return '';
    return text.replace(/<glm_block.*?<\/glm_block>/gs, '');
  }

  serializeMessages(messages) {
    return messages.map(m => ({
      role: m.role || 'user',
      content: m.content || String(m)
    }));
  }

  async prepareUpstreamRequest(request) {
    const token = await this.requestHandler.getNextToken();
    if (!token) {
      throw new Error('No available authentication tokens');
    }

    const model = this.settings.UPSTREAM_MODELS[request.model];
    const thinking = request.thinking?.type !== 'disabled';

    const body = {
      stream: true,
      model: model,
      messages: this.serializeMessages(request.messages),
      chat_id: crypto.randomUUID(),
      features: {
        image_generation: false,
        code_interpreter: false,
        web_search: false,
        auto_web_search: false,
        preview_mode: false,
        enable_thinking: thinking,
      },
      id: crypto.randomUUID(),
      params: {},
      tool_servers: [],
      variables: {
        '{{USER_NAME}}': 'User',
        '{{USER_LOCATION}}': 'Unknown',
        '{{CURRENT_DATETIME}}': new Date().toISOString().slice(0, 19).replace('T', ' '),
      },
    };

    const headers = {
      ...this.requestHandler.tokenManager.getCommonHeaders(),
      'Authorization': `Bearer ${token}`,
      'Accept': 'application/json, text/event-stream',
      'Accept-Encoding': 'gzip, deflate, br, zstd',
      'x-fe-version': 'prod-fe-1.0.56'
    };

    return { body, headers, token };
  }

  async *streamProxyResponse(request) {
    let token = null;
    
    try {
      const { body, headers, token: usedToken } = await this.prepareUpstreamRequest(request);
      token = usedToken;
      
      const completionId = `chatcmpl-${crypto.randomUUID().toString().replace(/-/g, '').substring(0, 29)}`;
      let thinkOpen = false;
      let currentPhase = null;

      const response = await fetch(this.settings.UPSTREAM_URL, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const newToken = await this.requestHandler.handleTokenFailure(token);
        if (newToken && response.status === 401) {
          // Retry with refreshed token
          const newHeaders = { ...headers, 'Authorization': `Bearer ${newToken}` };
          const retryResponse = await fetch(this.settings.UPSTREAM_URL, {
            method: 'POST',
            headers: newHeaders,
            body: JSON.stringify(body),
          });
          
          if (retryResponse.ok) {
            response = retryResponse;
            token = newToken;
          }
        }
        
        if (!response.ok) {
          const errorText = await response.text();
          const errorMsg = `Error: ${response.status} - ${errorText}`;
          
          yield `data: ${JSON.stringify({
          id: completionId,
          object: 'chat.completion.chunk',
          created: Math.floor(Date.now() / 1000),
          model: request.model,
          choices: [{
            index: 0,
            delta: { content: errorMsg },
            finish_reason: 'stop'
          }]
        })}\n\n`;
          yield 'data: [DONE]\n\n';
          return;
        }
      }

      this.requestHandler.markTokenSuccess(token);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');

          for (const line of lines) {
            const trimmedLine = line.trim();
            if (!trimmedLine.startsWith('data: ')) continue;

            const payloadStr = trimmedLine.substring(6);
            if (payloadStr === '[DONE]') {
              if (thinkOpen) {
                yield `data: ${JSON.stringify({
                  id: completionId,
                  object: 'chat.completion.chunk',
                  created: Math.floor(Date.now() / 1000),
                  model: request.model,
                  choices: [{
                    index: 0,
                    delta: { content: '</think>' },
                    finish_reason: null
                  }]
                })}\n\n`;
              }
              yield `data: ${JSON.stringify({
                id: completionId,
                object: 'chat.completion.chunk',
                created: Math.floor(Date.now() / 1000),
                model: request.model,
                choices: [{
                  index: 0,
                  delta: {},
                  finish_reason: 'stop'
                }]
              })}\n\n`;
              yield 'data: [DONE]\n\n';
              return;
            }

            let data;
            try {
              data = JSON.parse(payloadStr).data || {};
            } catch (e) {
              continue;
            }

            const newPhase = data.phase;
            if (newPhase) currentPhase = newPhase;
            if (!currentPhase) continue;

            const content = data.delta_content || data.edit_content;
            if (!content) continue;

            let cleanedText = '';
            if (currentPhase === 'thinking') {
              cleanedText = this.cleanThinkingContent(content);
            } else if (currentPhase === 'answer') {
              cleanedText = this.cleanAnswerContent(content);
            }

            const match = content.match(/(.*<\/details>)(.*)/s);
            if (match) {
              const [, , answerPart] = match;
              cleanedText = this.cleanAnswerContent(answerPart);
              currentPhase = 'answer';
            }

            if (currentPhase === 'thinking') {
              if (!thinkOpen) {
                yield `data: ${JSON.stringify({
                  id: completionId,
                  object: 'chat.completion.chunk',
                  created: Math.floor(Date.now() / 1000),
                  model: request.model,
                  choices: [{
                    index: 0,
                    delta: { content: '<think>' },
                    finish_reason: null
                  }]
                })}\n\n`;
                thinkOpen = true;
              }
              if (cleanedText) {
                yield `data: ${JSON.stringify({
                  id: completionId,
                  object: 'chat.completion.chunk',
                  created: Math.floor(Date.now() / 1000),
                  model: request.model,
                  choices: [{
                    index: 0,
                    delta: { content: cleanedText },
                    finish_reason: null
                  }]
                })}\n\n`;
              }
            } else if (currentPhase === 'answer') {
              if (thinkOpen) {
                yield `data: ${JSON.stringify({
                  id: completionId,
                  object: 'chat.completion.chunk',
                  created: Math.floor(Date.now() / 1000),
                  model: request.model,
                  choices: [{
                    index: 0,
                    delta: { content: '</think>' },
                    finish_reason: null
                  }]
                })}\n\n`;
                thinkOpen = false;
              }

              yield `data: ${JSON.stringify({
                id: completionId,
                object: 'chat.completion.chunk',
                created: Math.floor(Date.now() / 1000),
                model: request.model,
                choices: [{
                  index: 0,
                  delta: { content: cleanedText },
                  finish_reason: null
                }]
              })}\n\n`;
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    } catch (error) {
      console.error('Stream error:', error);
      throw error;
    }
  }

  async nonStreamProxyResponse(request) {
    let token = null;
    
    try {
      const { body, headers, token: usedToken } = await this.prepareUpstreamRequest(request);
      token = usedToken;
      
      const rawThinkingParts = [];
      const rawAnswerParts = [];
      let currentPhase = null;

      const response = await fetch(this.settings.UPSTREAM_URL, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const newToken = await this.requestHandler.handleTokenFailure(token);
        if (newToken && response.status === 401) {
          // Retry with refreshed token
          const newHeaders = { ...headers, 'Authorization': `Bearer ${newToken}` };
          const retryResponse = await fetch(this.settings.UPSTREAM_URL, {
            method: 'POST',
            headers: newHeaders,
            body: JSON.stringify(body),
          });
          
          if (retryResponse.ok) {
            response = retryResponse;
            token = newToken;
          }
        }
        
        if (!response.ok) {
          const errorDetail = await response.text();
          throw new Error(`Upstream error: ${errorDetail}`);
        }
      }

      this.requestHandler.markTokenSuccess(token);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');

          for (const line of lines) {
            const trimmedLine = line.trim();
            if (!trimmedLine.startsWith('data: ')) continue;

            const payloadStr = trimmedLine.substring(6);
            if (payloadStr === '[DONE]') break;

            try {
              const parsed = JSON.parse(payloadStr);
              const data = parsed.data || {};
              
              // Handle API errors
              if (data.error || parsed.error) {
                const error = data.error || parsed.error;
                rawAnswerParts.push(`Error: ${error.detail || error.message || 'Unknown error'}`);
                continue;
              }
              
              const newPhase = data.phase;
              if (newPhase) currentPhase = newPhase;
              if (!currentPhase) continue;

              const content = data.delta_content || data.edit_content;
              if (!content) continue;

              const match = content.match(/(.*<\/details>)(.*)/s);
              if (match) {
                const [, , answerPart] = match;
                rawAnswerParts.push(answerPart);
              } else {
                if (currentPhase === 'thinking') {
                  rawThinkingParts.push(content);
                } else if (currentPhase === 'answer') {
                  rawAnswerParts.push(content);
                }
              }
            } catch (e) {
              continue;
            }
          }
        }
      } finally {
        reader.releaseLock();
      }

      const fullAnswer = rawAnswerParts.join('');
      const cleanedAnswerText = this.cleanAnswerContent(fullAnswer).trim();
      let finalContent = cleanedAnswerText;

      if (rawThinkingParts.length > 0) {
        const cleanedThinkText = this.cleanThinkingContent(rawThinkingParts.join(''));
        if (cleanedThinkText) {
          finalContent = `<think>${cleanedThinkText}</think>${cleanedAnswerText}`;
        }
      }

      return {
        id: `chatcmpl-${crypto.randomUUID().toString().replace(/-/g, '').substring(0, 29)}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: request.model,
        choices: [{
          index: 0,
          message: {
            role: 'assistant',
            content: finalContent
          },
          finish_reason: 'stop'
        }],
      };
    } catch (error) {
      console.error('Non-stream processing failed:', error);
      throw error;
    }
  }

  async handleChatCompletion(request, corsHeaders = {}) {
    const isStream = request.stream !== undefined ? Boolean(request.stream) : false;
    
    if (isStream) {
      const encoder = new TextEncoder();
      const self = this;
      const stream = new ReadableStream({
        async start(controller) {
          try {
            for await (const chunk of self.streamProxyResponse(request)) {
              controller.enqueue(encoder.encode(chunk));
            }
          } catch (error) {
            console.error('Streaming error:', error);
            controller.error(error);
          } finally {
            controller.close();
          }
        }
      });

      return new Response(stream, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
          ...corsHeaders,
        },
      });
    } else {
      const result = await this.nonStreamProxyResponse(request);
      return new Response(JSON.stringify(result), {
        headers: {
          'Content-Type': 'application/json',
          ...corsHeaders,
        },
      });
    }
  }
}

// Global instances
let tokenManager;
let requestHandler;
let proxyHandler;

// Main Worker
export default {
  async fetch(request, env) {
    const settings = getSettings(env);
    
    // Initialize on first request
    if (!requestHandler) {
      tokenManager = new TokenManager(settings.TOKEN_POOL_SIZE);
      requestHandler = new RequestHandler(tokenManager);
      proxyHandler = new ProxyHandler(requestHandler, settings);
    }

    const url = new URL(request.url);

    // CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        status: 200,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization',
          'Access-Control-Max-Age': '86400',
        },
      });
    }

    // Add CORS headers
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Credentials': 'true',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': '*',
    };

    try {
      // Health check
      if (url.pathname === '/health') {
        return new Response(JSON.stringify({ status: 'healthy' }), {
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      }

      // List models
      if (url.pathname === '/v1/models' && request.method === 'GET') {
        const models = Object.keys(settings.UPSTREAM_MODELS).map(model => ({
          id: model,
          object: 'model',
          owned_by: 'z-ai'
        }));
        
        return new Response(JSON.stringify({ 
          object: 'list',
          data: models 
        }), {
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      }

      // Chat completions
      if (url.pathname === '/v1/chat/completions' && request.method === 'POST') {
        // Verify authentication
        const authHeader = request.headers.get('Authorization');
        if (!authHeader || !authHeader.startsWith('Bearer ')) {
          return new Response(JSON.stringify({
            error: { message: 'Authorization header required' }
          }), {
            status: 401,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }

        const token = authHeader.substring(7);
        if (token !== settings.API_KEY) {
          return new Response(JSON.stringify({
            error: { message: 'Invalid API key' }
          }), {
            status: 401,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }

        // Initialize token pool if needed
        if (!tokenManager.initialized) {
          try {
            await tokenManager.initializeTokenPool();
            if (tokenManager.tokens.length === 0) {
              return new Response(JSON.stringify({
                error: { message: 'Service unavailable: Failed to initialize token pool' }
              }), {
                status: 503,
                headers: { 'Content-Type': 'application/json', ...corsHeaders },
              });
            }
          } catch (error) {
            return new Response(JSON.stringify({
              error: { message: 'Service unavailable: Token initialization failed' }
            }), {
              status: 503,
              headers: { 'Content-Type': 'application/json', ...corsHeaders },
            });
          }
        }

        const requestBody = await request.json();
        
        // Validate model
        if (!settings.UPSTREAM_MODELS[requestBody.model]) {
          return new Response(JSON.stringify({
            error: { message: `Model '${requestBody.model}' not found` }
          }), {
            status: 404,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }

        // Handle chat completion
        return await proxyHandler.handleChatCompletion(requestBody, corsHeaders);
      }

      // 404 for other routes
      return new Response(JSON.stringify({
        error: { message: 'Not found' }
      }), {
        status: 404,
        headers: { 'Content-Type': 'application/json', ...corsHeaders },
      });

    } catch (error) {
      console.error('Unexpected error:', error);
      return new Response(JSON.stringify({
        error: { message: 'Internal server error' }
      }), {
        status: 500,
        headers: { 'Content-Type': 'application/json', ...corsHeaders },
      });
    }
  },
};
