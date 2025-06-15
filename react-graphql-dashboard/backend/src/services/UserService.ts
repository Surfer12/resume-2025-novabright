import { logger } from '../utils/logger';

export interface User {
  id: string;
  name: string;
  email: string;
  role: string;
  createdAt: Date;
  lastActive: Date;
}

export class UserService {
  private users: Map<string, User> = new Map();

  constructor() {
    // Initialize with some mock data for development
    this.initializeMockData();
  }

  private initializeMockData() {
    const mockUsers: User[] = [
      {
        id: '1',
        name: 'Admin User',
        email: 'admin@dashboard.com',
        role: 'admin',
        createdAt: new Date('2024-01-01'),
        lastActive: new Date()
      },
      {
        id: '2',
        name: 'Demo User',
        email: 'demo@dashboard.com',
        role: 'user',
        createdAt: new Date('2024-01-15'),
        lastActive: new Date(Date.now() - 1000 * 60 * 30) // 30 minutes ago
      }
    ];

    mockUsers.forEach(user => this.users.set(user.id, user));
    logger.info(`UserService initialized with ${mockUsers.length} mock users`);
  }

  async getUser(id: string): Promise<User | null> {
    const user = this.users.get(id);
    if (user) {
      logger.debug(`Retrieved user: ${user.name}`);
    }
    return user || null;
  }

  async getAllUsers(): Promise<User[]> {
    const users = Array.from(this.users.values());
    logger.debug(`Retrieved ${users.length} users`);
    return users;
  }

  async createUser(userData: Omit<User, 'id' | 'createdAt' | 'lastActive'>): Promise<User> {
    const user: User = {
      ...userData,
      id: Math.random().toString(36).substr(2, 9),
      createdAt: new Date(),
      lastActive: new Date()
    };

    this.users.set(user.id, user);
    logger.info(`Created new user: ${user.name}`);
    return user;
  }

  async updateUser(id: string, updates: Partial<User>): Promise<User | null> {
    const user = this.users.get(id);
    if (!user) {
      logger.warn(`Attempted to update non-existent user: ${id}`);
      return null;
    }

    const updatedUser = { ...user, ...updates, lastActive: new Date() };
    this.users.set(id, updatedUser);
    logger.info(`Updated user: ${updatedUser.name}`);
    return updatedUser;
  }

  async deleteUser(id: string): Promise<boolean> {
    const deleted = this.users.delete(id);
    if (deleted) {
      logger.info(`Deleted user: ${id}`);
    } else {
      logger.warn(`Attempted to delete non-existent user: ${id}`);
    }
    return deleted;
  }

  async getUsersByRole(role: string): Promise<User[]> {
    const users = Array.from(this.users.values()).filter(user => user.role === role);
    logger.debug(`Retrieved ${users.length} users with role: ${role}`);
    return users;
  }
}
