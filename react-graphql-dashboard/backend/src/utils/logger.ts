import winston from 'winston';

// Create logger instance with appropriate configuration for development
export const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.colorize({ all: true }),
    winston.format.printf(({ timestamp, level, message, ...meta }) => {
      let log = `${timestamp} [${level}]: ${message}`;
      
      // Add metadata if present
      if (Object.keys(meta).length > 0) {
        log += ` ${JSON.stringify(meta, null, 2)}`;
      }
      
      return log;
    })
  ),
  transports: [
    // Console transport for development
    new winston.transports.Console({
      handleExceptions: true,
      handleRejections: true,
    }),
    
    // File transport for persistent logging
    new winston.transports.File({
      filename: 'logs/error.log',
      level: 'error',
      handleExceptions: true,
      handleRejections: true,
    }),
    
    new winston.transports.File({
      filename: 'logs/combined.log',
      handleExceptions: true,
      handleRejections: true,
    }),
  ],
  exitOnError: false,
});

// Create logs directory if it doesn't exist
import { existsSync, mkdirSync } from 'fs';
if (!existsSync('logs')) {
  mkdirSync('logs');
}

// Add performance logging helpers
export const logPerformance = (operation: string, startTime: number, metadata?: any) => {
  const duration = Date.now() - startTime;
  logger.info(`Performance: ${operation}`, {
    duration: `${duration}ms`,
    ...metadata
  });
};

export const logError = (error: Error, context?: string) => {
  logger.error(`Error${context ? ` in ${context}` : ''}`, {
    message: error.message,
    stack: error.stack,
    name: error.name
  });
};

export default logger;
