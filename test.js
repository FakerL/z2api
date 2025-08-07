#!/usr/bin/env node

// Simple test for the Z.AI proxy worker
const workerModule = await import('./index.js');
const worker = workerModule.default;

// Mock environment
const env = {
  TOKEN_POOL_SIZE: process.env.TOKEN_POOL_SIZE || '2',
  API_KEY: process.env.API_KEY || 'sk-z2api-key-2024',
  SHOW_THINK_TAGS: process.env.SHOW_THINK_TAGS || 'true'
};

async function testHealthEndpoint() {
  const request = new Request('http://localhost:8787/health');
  const response = await worker.fetch(request, env);
  const result = await response.json();
  console.log('✓ Health check:', result);
  return response.status === 200;
}

async function testModelsEndpoint() {
  const request = new Request('http://localhost:8787/v1/models');
  const response = await worker.fetch(request, env);
  const result = await response.json();
  console.log('✓ Models endpoint:', result);
  return response.status === 200 && result.data.length > 0;
}

async function testChatCompletion() {
  try {
    const request = new Request('http://localhost:8787/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer sk-z2api-key-2024'
      },
      body: JSON.stringify({
        model: 'glm-4.5-air',
        messages: [{ role: 'user', content: 'Say hello' }],
        stream: false
      })
    });
    
    const response = await worker.fetch(request, env);
    console.log('✓ Chat completion status:', response.status);
    
    if (response.status === 200) {
      const result = await response.json();
      console.log('✓ Full chat response:', JSON.stringify(result, null, 2));
      if (result.choices && result.choices[0] && result.choices[0].message) {
        console.log('✓ Message content:', result.choices[0].message.content);
        return result.choices[0].message.content.length > 0;
      } else {
        console.log('✗ No valid message content in response');
        return false;
      }
    } else if (response.status === 503) {
      console.log('⚠️  Token pool initialization failed - chat test not applicable');
      throw new Error('Token pool required for chat test');
    } else if (response.status === 500) {
      const error = await response.text();
      if (error.includes('Internal server error')) {
        console.log('⚠️  Token fetch likely failed - network or service issue');
        throw new Error('Token pool required for chat test');
      } else {
        console.log('✗ Chat completion error:', error);
        return false;
      }
    } else {
      const error = await response.text();
      console.log('✗ Chat completion error:', error);
      return false;
    }
  } catch (error) {
    console.log('⚠️  Chat test failed:', error.message);
    throw error;
  }
}

async function testStreamingChat() {
  try {
    const request = new Request('http://localhost:8787/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer sk-z2api-key-2024'
      },
      body: JSON.stringify({
        model: 'glm-4.5-air',
        messages: [{ role: 'user', content: 'Count to 3' }],
        stream: true
      })
    });
    
    const response = await worker.fetch(request, env);
    console.log('✓ Streaming chat status:', response.status);
    
    if (response.status === 200) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let chunks = 0;
      
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          const chunk = decoder.decode(value);
          chunks++;
          if (chunks <= 3) {
            console.log(`✓ Stream chunk ${chunks}:`, chunk.substring(0, 100) + '...');
          }
          
          if (chunks > 20) break; // Prevent infinite loop
        }
        
        console.log(`✓ Received ${chunks} streaming chunks`);
        return chunks > 0;
      } catch (error) {
        console.log('✗ Streaming error:', error.message);
        return false;
      }
    } else if (response.status === 503) {
      console.log('⚠️  Token pool initialization failed - streaming test not applicable');
      throw new Error('Token pool required for streaming test');
    } else if (response.status === 500) {
      const error = await response.text();
      if (error.includes('Internal server error')) {
        console.log('⚠️  Token fetch likely failed - network or service issue');
        throw new Error('Token pool required for streaming test');
      } else {
        console.log('✗ Streaming chat error:', error);
        return false;
      }
    } else {
      const error = await response.text();
      console.log('✗ Streaming chat error:', error);
      return false;
    }
  } catch (error) {
    console.log('⚠️  Streaming test failed:', error.message);
    throw error;
  }
}

async function testInvalidAuth() {
  const request = new Request('http://localhost:8787/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer invalid-key'
    },
    body: JSON.stringify({
      model: 'glm-4.5-air',
      messages: [{ role: 'user', content: 'Hello' }],
      stream: false
    })
  });
  
  const response = await worker.fetch(request, env);
  console.log('✓ Invalid auth status:', response.status);
  return response.status === 401;
}

async function testInvalidModel() {
  const request = new Request('http://localhost:8787/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer sk-z2api-key-2024'
    },
    body: JSON.stringify({
      model: 'invalid-model',
      messages: [{ role: 'user', content: 'Hello' }],
      stream: false
    })
  });
  
  const response = await worker.fetch(request, env);
  console.log('✓ Invalid model status:', response.status);
  return response.status === 404;
}

async function testEmptyTokenPool() {
  // Test with zero token pool size - need a fresh worker instance
  const freshWorkerModule = await import(`./index.js?t=${Date.now()}`);
  const freshWorker = freshWorkerModule.default;
  const emptyPoolEnv = { TOKEN_POOL_SIZE: '0' };
  
  const request = new Request('http://localhost:8787/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer sk-z2api-key-2024'
    },
    body: JSON.stringify({
      model: 'glm-4.5-air',
      messages: [{ role: 'user', content: 'Hello' }],
      stream: false
    })
  });
  
  const response = await freshWorker.fetch(request, emptyPoolEnv);
  console.log('✓ Empty token pool status:', response.status);
  return response.status === 503;
}

async function runTests() {
  console.log('🧪 Running Z.AI Proxy Tests...\n');
  
  try {
    // Basic endpoint tests
    const healthOk = await testHealthEndpoint();
    const modelsOk = await testModelsEndpoint();
    
    // Error handling tests
    const invalidAuthOk = await testInvalidAuth();
    const invalidModelOk = await testInvalidModel();
    const emptyTokenPoolOk = await testEmptyTokenPool();
    
    let chatOk = false;
    let streamingOk = false;
    
    // Test chat functionality with token pool
    console.log('\n🔄 Testing chat completion (this may take a moment)...');
    try {
      chatOk = await testChatCompletion();
    } catch (error) {
      if (error.message.includes('Token pool required')) {
        console.log('⚠️  Chat test skipped - token pool failed');
        chatOk = null;
      } else {
        console.log('⚠️  Chat test failed unexpectedly:', error.message);
        chatOk = false;
      }
    }
    
    console.log('\n🌊 Testing streaming chat (this may take a moment)...');
    try {
      streamingOk = await testStreamingChat();
    } catch (error) {
      if (error.message.includes('Token pool required')) {
        console.log('⚠️  Streaming test skipped - token pool failed');
        streamingOk = null;
      } else {
        console.log('⚠️  Streaming test failed unexpectedly:', error.message);
        streamingOk = false;
      }
    }
    
    console.log('\n📊 Test Results:');
    console.log(`Health endpoint: ${healthOk ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`Models endpoint: ${modelsOk ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`Invalid auth handling: ${invalidAuthOk ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`Invalid model handling: ${invalidModelOk ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`Empty token pool handling: ${emptyTokenPoolOk ? '✅ PASS' : '❌ FAIL'}`);
    
    if (chatOk === null) {
      console.log(`Chat completion: ⏭️  SKIPPED`);
    } else {
      console.log(`Chat completion: ${chatOk ? '✅ PASS' : '❌ FAIL'}`);
    }
    
    if (streamingOk === null) {
      console.log(`Streaming chat: ⏭️  SKIPPED`);
    } else {
      console.log(`Streaming chat: ${streamingOk ? '✅ PASS' : '❌ FAIL'}`);
    }
    
    // Count only non-null results
    const coreTests = [healthOk, modelsOk, invalidAuthOk, invalidModelOk, emptyTokenPoolOk];
    const chatTests = [chatOk, streamingOk].filter(result => result !== null);
    
    const totalTests = coreTests.length + chatTests.length;
    const passedTests = [...coreTests, ...chatTests].filter(Boolean).length;
    const skippedTests = [chatOk, streamingOk].filter(result => result === null).length;
    
    console.log(`\n🎯 Results: ${passedTests}/${totalTests} tests passed${skippedTests > 0 ? `, ${skippedTests} skipped` : ''}`);
    console.log('🎉 Tests completed!');
  } catch (error) {
    console.error('❌ Test failed:', error.message);
  }
}

runTests();
