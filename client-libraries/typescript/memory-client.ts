/**
 * BrainOps Memory Coordination Client
 * Ready-to-use TypeScript client for Weathercraft ERP, MyRoofGenius, and Codex
 *
 * Installation:
 * 1. Copy this file to your project's lib/ directory
 * 2. Use it to track context and enable persistent AI memory
 *
 * Usage:
 * ```typescript
 * import { MemoryClient } from './memory-client';
 *
 * const memory = new MemoryClient({
 *   apiUrl: 'https://brainops-ai-agents.onrender.com',
 *   tenantId: 'your-app-name',
 *   userId: currentUserId
 * });
 *
 * // Track user action
 * await memory.trackAction('page_view', { page: '/customers' });
 *
 * // Get AI context
 * const context = await memory.getAIContext();
 * ```
 */

export interface MemoryConfig {
  apiUrl: string;
  tenantId?: string;
  userId?: string;
  apiKey?: string;
}

export interface ContextEntry {
  key: string;
  value: any;
  layer: 'ephemeral' | 'session' | 'short_term' | 'long_term' | 'permanent';
  scope: 'global' | 'tenant' | 'user' | 'session' | 'agent';
  priority: 'critical' | 'high' | 'medium' | 'low';
  category: string;
  source?: string;
  tenant_id?: string;
  user_id?: string;
  session_id?: string;
  metadata?: Record<string, any>;
  expires_in_seconds?: number;
}

export interface SessionInfo {
  session_id: string;
  status: string;
  start_time: string;
  tenant_id?: string;
  user_id?: string;
}

export interface SessionContext {
  session_id: string;
  tenant_id?: string;
  user_id?: string;
  status: string;
  duration_seconds: number;
  conversation: {
    message_count: number;
    recent_messages: Array<{
      timestamp: string;
      role: string;
      content: string;
      metadata?: Record<string, any>;
    }>;
    summary: string;
  };
  agents: {
    current?: string;
    previous?: string;
    active: string[];
    handoff_count: number;
  };
  tasks: {
    pending: any[];
    completed: any[];
    completion_rate: number;
  };
  memory: {
    critical_facts: Record<string, any>;
    working_memory: Record<string, any>;
    long_term_refs: string[];
  };
  metadata?: Record<string, any>;
}

export class MemoryClient {
  private config: MemoryConfig;
  private sessionId?: string;

  constructor(config: MemoryConfig) {
    this.config = config;
  }

  // ========================================================================
  // CONTEXT OPERATIONS
  // ========================================================================

  /**
   * Store context entry
   */
  async storeContext(entry: Partial<ContextEntry>): Promise<string> {
    const response = await this.fetch('/memory/context/store', {
      method: 'POST',
      body: JSON.stringify({
        ...entry,
        tenant_id: entry.tenant_id || this.config.tenantId,
        user_id: entry.user_id || this.config.userId,
        session_id: entry.session_id || this.sessionId,
        source: entry.source || 'client'
      })
    });

    const { entry_id } = await response.json();
    return entry_id;
  }

  /**
   * Retrieve context by key
   */
  async retrieveContext(key: string, scope: string = 'tenant'): Promise<any> {
    const response = await this.fetch('/memory/context/retrieve', {
      method: 'POST',
      body: JSON.stringify({
        key,
        scope,
        tenant_id: this.config.tenantId,
        user_id: this.config.userId,
        session_id: this.sessionId
      })
    });

    const data = await response.json();
    return data.value;
  }

  /**
   * Search context
   */
  async searchContext(params: {
    query: string;
    scope?: string;
    layer?: string;
    category?: string;
    limit?: number;
  }): Promise<any[]> {
    const response = await this.fetch('/memory/context/search', {
      method: 'POST',
      body: JSON.stringify({
        ...params,
        tenant_id: this.config.tenantId
      })
    });

    const { results } = await response.json();
    return results;
  }

  // ========================================================================
  // SESSION OPERATIONS
  // ========================================================================

  /**
   * Start a new session
   */
  async startSession(sessionId?: string, initialContext?: Record<string, any>): Promise<SessionInfo> {
    this.sessionId = sessionId || this.generateSessionId();

    const response = await this.fetch('/memory/session/start', {
      method: 'POST',
      body: JSON.stringify({
        session_id: this.sessionId,
        tenant_id: this.config.tenantId,
        user_id: this.config.userId,
        initial_context: initialContext || {}
      })
    });

    return response.json();
  }

  /**
   * Resume existing session
   */
  async resumeSession(sessionId: string): Promise<SessionInfo> {
    this.sessionId = sessionId;

    const response = await this.fetch(`/memory/session/resume/${sessionId}`, {
      method: 'POST'
    });

    return response.json();
  }

  /**
   * End session
   */
  async endSession(reason: string = 'completed'): Promise<void> {
    if (!this.sessionId) return;

    await this.fetch(`/memory/session/end/${this.sessionId}?reason=${reason}`, {
      method: 'POST'
    });

    this.sessionId = undefined;
  }

  /**
   * Add message to session
   */
  async addMessage(role: string, content: string, metadata?: Record<string, any>): Promise<void> {
    if (!this.sessionId) {
      await this.startSession();
    }

    await this.fetch('/memory/session/message', {
      method: 'POST',
      body: JSON.stringify({
        session_id: this.sessionId,
        role,
        content,
        metadata: metadata || {}
      })
    });
  }

  /**
   * Add task to session
   */
  async addTask(task: Record<string, any>, status: string = 'pending'): Promise<void> {
    if (!this.sessionId) {
      await this.startSession();
    }

    await this.fetch('/memory/session/task', {
      method: 'POST',
      body: JSON.stringify({
        session_id: this.sessionId,
        task,
        status
      })
    });
  }

  /**
   * Get complete session context
   */
  async getSessionContext(sessionId?: string): Promise<SessionContext> {
    const id = sessionId || this.sessionId;
    if (!id) {
      throw new Error('No session ID provided');
    }

    const response = await this.fetch(`/memory/session/context/${id}`);
    const { context } = await response.json();
    return context;
  }

  // ========================================================================
  // CONVENIENCE METHODS
  // ========================================================================

  /**
   * Track user action (high-level helper)
   */
  async trackAction(actionType: string, data: Record<string, any>): Promise<void> {
    await this.storeContext({
      key: `action_${actionType}_${Date.now()}`,
      value: { type: actionType, ...data, timestamp: new Date().toISOString() },
      layer: 'session',
      scope: 'user',
      priority: 'medium',
      category: 'user_activity'
    });
  }

  /**
   * Store user preference (permanent)
   */
  async storePreference(key: string, value: any): Promise<void> {
    await this.storeContext({
      key: `pref_${key}`,
      value,
      layer: 'permanent',
      scope: 'user',
      priority: 'medium',
      category: 'preferences'
    });
  }

  /**
   * Get user preferences
   */
  async getPreferences(): Promise<Record<string, any>> {
    const results = await this.searchContext({
      query: 'preferences',
      scope: 'user',
      category: 'preferences'
    });

    const prefs: Record<string, any> = {};
    for (const result of results) {
      const key = result.key.replace('pref_', '');
      prefs[key] = result.value;
    }
    return prefs;
  }

  /**
   * Get context for AI (convenience)
   */
  async getAIContext(): Promise<{
    recentActions: any[];
    preferences: Record<string, any>;
    sessionContext?: SessionContext;
  }> {
    const [recentActions, preferences, sessionContext] = await Promise.all([
      this.searchContext({
        query: 'user_activity',
        scope: 'user',
        category: 'user_activity',
        limit: 20
      }),
      this.getPreferences(),
      this.sessionId ? this.getSessionContext() : Promise.resolve(undefined)
    ]);

    return {
      recentActions: recentActions.map(r => r.value),
      preferences,
      sessionContext
    };
  }

  /**
   * Store critical fact (permanent, high priority)
   */
  async storeCriticalFact(key: string, value: any): Promise<void> {
    await this.storeContext({
      key: `critical_${key}`,
      value,
      layer: 'permanent',
      scope: 'tenant',
      priority: 'critical',
      category: 'critical_facts'
    });
  }

  // ========================================================================
  // UTILITIES
  // ========================================================================

  private async fetch(path: string, init?: RequestInit): Promise<Response> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json'
    };

    if (this.config.apiKey) {
      headers['Authorization'] = `Bearer ${this.config.apiKey}`;
    }

    const response = await fetch(`${this.config.apiUrl}${path}`, {
      ...init,
      headers: {
        ...headers,
        ...init?.headers
      }
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Memory API error: ${error}`);
    }

    return response;
  }

  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substring(7)}`;
  }

  /**
   * Get current session ID
   */
  getSessionId(): string | undefined {
    return this.sessionId;
  }

  /**
   * Set session ID manually
   */
  setSessionId(sessionId: string): void {
    this.sessionId = sessionId;
  }
}

// ========================================================================
// USAGE EXAMPLES
// ========================================================================

/*
// Example 1: Basic setup
const memory = new MemoryClient({
  apiUrl: 'https://brainops-ai-agents.onrender.com',
  tenantId: 'my-app',
  userId: 'user123'
});

// Example 2: Track user navigation
await memory.trackAction('page_view', {
  page: '/customers/123',
  referrer: '/dashboard'
});

// Example 3: Store user preference
await memory.storePreference('theme', 'dark');
await memory.storePreference('language', 'en');

// Example 4: AI chat with context
await memory.startSession();

await memory.addMessage('user', 'I need help with invoicing');
await memory.addMessage('assistant', 'I can help with that...');

const context = await memory.getAIContext();
// Use context in AI prompt for contextual responses

// Example 5: Critical business data
await memory.storeCriticalFact('customer_123_contract_end', '2025-12-31');

// Example 6: Track task completion
await memory.addTask({ id: 'task1', description: 'Create invoice' }, 'pending');
// ... later
await memory.addTask({ id: 'task1', description: 'Create invoice' }, 'completed');
*/
