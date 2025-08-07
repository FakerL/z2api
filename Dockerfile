FROM node:20-slim

WORKDIR /app

COPY index.js server.js test.js ./

# Expose port 3000
EXPOSE 3000

# Set default environment variables
ENV PORT=3000
ENV TOKEN_POOL_SIZE=5

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD node -e "fetch('http://localhost:${PORT}/health').then(r=>r.ok?process.exit(0):process.exit(1)).catch(()=>process.exit(1))"

# Start the server
CMD ["node", "server.js"]
