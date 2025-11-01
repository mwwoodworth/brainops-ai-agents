# Memory Coordination System - Practical Integration Guide
**How to Actually Use This in Production**

---

## The Problem You Identified

You're right - a memory system is **useless if not used**. Here's how to integrate it into:
1. **Weathercraft ERP** (internal tool)
2. **MyRoofGenius** (customer-facing SaaS)
3. **Codex VSCode Extension** (AI coding assistant)

---

## 1. Weathercraft ERP Integration

### Current State (No Memory)
```typescript
// User opens customer page
// AI assistant has NO IDEA what they just did
// Every interaction starts from scratch ❌
```

### With Memory Coordination
```typescript
// src/lib/memory-client.ts
import { MemoryClient } from './memory-coordination-client';

const memory = new MemoryClient({
  apiUrl: 'https://brainops-ai-agents.onrender.com',
  tenantId: 'weathercraft_erp'
});

// Track user context automatically
export async function trackUserAction(action: {
  type: string;
  data: any;
  userId: string;
}) {
  await memory.storeContext({
    key: `user_action_${action.userId}_${Date.now()}`,
    value: action,
    layer: 'session',
    scope: 'user',
    priority: 'medium',
    category: 'user_activity',
    userId: action.userId
  });
}

// AI assistant knows EVERYTHING the user did
export async function getAIContext(userId: string) {
  const recentActions = await memory.searchContext({
    query: `user:${userId}`,
    scope: 'user',
    layer: 'session',
    limit: 20
  });

  return {
    recentPages: recentActions.filter(a => a.value.type === 'page_view'),
    recentEdits: recentActions.filter(a => a.value.type === 'edit'),
    currentFocus: recentActions[0] // Last action
  };
}
```

### Integration Points

#### A. Page Navigation Tracking
```typescript
// src/app/layout.tsx
'use client';

import { usePathname } from 'next/navigation';
import { useEffect } from 'react';
import { trackUserAction } from '@/lib/memory-client';

export function NavigationTracker() {
  const pathname = usePathname();

  useEffect(() => {
    trackUserAction({
      type: 'page_view',
      data: { pathname, timestamp: new Date() },
      userId: getCurrentUserId()
    });
  }, [pathname]);

  return null;
}
```

#### B. Customer Context Preservation
```typescript
// When user opens customer page
async function loadCustomerPage(customerId: string) {
  // Store that user is viewing this customer
  await memory.storeContext({
    key: `current_customer_${userId}`,
    value: { customerId, openedAt: new Date() },
    layer: 'session',
    scope: 'user',
    priority: 'high',
    category: 'current_context'
  });

  // AI now knows user is working with this customer
  // Can proactively suggest related jobs, invoices, etc.
}
```

#### C. AI Assistant with Full Context
```typescript
// src/components/AIAssistant.tsx
async function getAIResponse(userQuery: string) {
  // Get full user context
  const context = await memory.getSessionContext(sessionId);

  const prompt = `
    User is working on: ${context.memory.working_memory.current_page}
    Recent actions: ${context.conversation.recent_messages}
    Current customer: ${context.memory.critical_facts.customer_id}

    User asks: ${userQuery}

    Provide contextual help.
  `;

  // AI response is now CONTEXTUAL, not generic
  return await callLLM(prompt);
}
```

---

## 2. MyRoofGenius Integration

### Current State (No Continuity)
```typescript
// Customer signs up → chooses plan → configures settings
// Next day, logs in → AI has ZERO memory of onboarding ❌
// Has to ask same questions again ❌
```

### With Memory Coordination
```typescript
// src/lib/customer-memory.ts

// Track onboarding progress
export async function trackOnboardingStep(step: {
  userId: string;
  stepName: string;
  completed: boolean;
  data: any;
}) {
  await memory.storeContext({
    key: `onboarding_${step.userId}_${step.stepName}`,
    value: { ...step, timestamp: new Date() },
    layer: 'long_term', // Persist beyond session
    scope: 'user',
    priority: 'high',
    category: 'onboarding'
  });
}

// Resume onboarding from where they left off
export async function getOnboardingProgress(userId: string) {
  const steps = await memory.searchContext({
    query: `onboarding user:${userId}`,
    scope: 'user',
    category: 'onboarding'
  });

  return {
    completedSteps: steps.filter(s => s.value.completed),
    nextStep: steps.find(s => !s.value.completed),
    progress: calculateProgress(steps)
  };
}
```

### Integration Points

#### A. Smart Onboarding Resume
```typescript
// src/app/dashboard/page.tsx
export default async function Dashboard() {
  const onboarding = await getOnboardingProgress(userId);

  if (onboarding.progress < 100) {
    return <OnboardingResume nextStep={onboarding.nextStep} />;
  }

  // AI knows user completed onboarding
  return <FullDashboard />;
}
```

#### B. Customer Preferences Memory
```typescript
// Store customer preferences permanently
async function saveCustomerPreference(pref: {
  userId: string;
  key: string;
  value: any;
}) {
  await memory.storeContext({
    key: `pref_${pref.userId}_${pref.key}`,
    value: pref.value,
    layer: 'permanent', // Never forget
    scope: 'user',
    priority: 'medium',
    category: 'preferences'
  });
}

// AI chat knows user preferences
async function getAIChatContext(userId: string) {
  const prefs = await memory.searchContext({
    query: `preferences user:${userId}`,
    scope: 'user',
    category: 'preferences'
  });

  return {
    preferredLanguage: prefs.find(p => p.key === 'language')?.value,
    timezone: prefs.find(p => p.key === 'timezone')?.value,
    notificationStyle: prefs.find(p => p.key === 'notifications')?.value
  };
}
```

#### C. Multi-Session AI Conversations
```typescript
// src/components/AIChat.tsx
export function AIChat() {
  const [sessionId, setSessionId] = useState<string>();

  useEffect(() => {
    // Start or resume session
    async function initSession() {
      const existingSession = localStorage.getItem('ai_session_id');

      if (existingSession) {
        // Resume existing session - AI remembers everything
        await memory.resumeSession(existingSession);
        setSessionId(existingSession);
      } else {
        // Start new session
        const session = await memory.startSession({
          sessionId: generateId(),
          tenantId: 'myroofgenius',
          userId: currentUserId
        });
        localStorage.setItem('ai_session_id', session.session_id);
        setSessionId(session.session_id);
      }
    }

    initSession();
  }, []);

  async function sendMessage(message: string) {
    // Add to session memory
    await memory.addMessage({
      sessionId,
      role: 'user',
      content: message
    });

    // Get AI response with FULL conversation history
    const context = await memory.getSessionContext(sessionId);
    const response = await getAIResponse(message, context);

    await memory.addMessage({
      sessionId,
      role: 'assistant',
      content: response
    });
  }
}
```

---

## 3. Codex VSCode Extension Integration

### Current State (No Persistence)
```typescript
// Codex helps you write code in file A
// You switch to file B
// Codex has NO MEMORY of what you just did in file A ❌
// Can't provide contextual suggestions ❌
```

### With Memory Coordination

#### A. VSCode Extension Client
```typescript
// extension/src/memoryClient.ts
import * as vscode from 'vscode';

export class CodexMemoryClient {
  private sessionId: string;
  private apiUrl = 'https://brainops-ai-agents.onrender.com';

  constructor() {
    this.sessionId = vscode.workspace.getConfiguration('codex').get('sessionId')
      || this.createSession();
  }

  async createSession() {
    const response = await fetch(`${this.apiUrl}/memory/session/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: generateId(),
        tenant_id: 'codex_vscode',
        user_id: vscode.env.machineId, // Unique per machine
        initial_context: {
          workspace: vscode.workspace.workspaceFolders?.[0].uri.fsPath,
          language: vscode.env.language
        }
      })
    });

    const { session_id } = await response.json();

    // Persist across VSCode restarts
    await vscode.workspace.getConfiguration('codex').update(
      'sessionId',
      session_id,
      vscode.ConfigurationTarget.Global
    );

    return session_id;
  }

  async trackFileEdit(file: string, content: string) {
    await fetch(`${this.apiUrl}/memory/context/store`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        key: `file_edit_${file}_${Date.now()}`,
        value: {
          file,
          content: content.substring(0, 1000), // First 1KB
          timestamp: new Date(),
          lineCount: content.split('\n').length
        },
        layer: 'session',
        scope: 'session',
        priority: 'medium',
        category: 'code_edit',
        session_id: this.sessionId
      })
    });
  }

  async getRecentContext() {
    const response = await fetch(
      `${this.apiUrl}/memory/session/context/${this.sessionId}`
    );

    const { context } = await response.json();
    return context;
  }

  async addConversationMessage(role: 'user' | 'assistant', content: string) {
    await fetch(`${this.apiUrl}/memory/session/message`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: this.sessionId,
        role,
        content,
        metadata: {
          activeFile: vscode.window.activeTextEditor?.document.fileName,
          cursorPosition: vscode.window.activeTextEditor?.selection.active
        }
      })
    });
  }
}
```

#### B. Context-Aware Code Suggestions
```typescript
// extension/src/codexProvider.ts
export class CodexCompletionProvider implements vscode.InlineCompletionItemProvider {
  private memory: CodexMemoryClient;

  async provideInlineCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position
  ): Promise<vscode.InlineCompletionItem[]> {

    // Get FULL session context
    const context = await this.memory.getRecentContext();

    const prompt = `
      # Recent Files Edited
      ${context.memory.working_memory.recent_files?.join('\n')}

      # Recent Conversation
      ${context.conversation.recent_messages.slice(-5).map(m =>
        `${m.role}: ${m.content}`
      ).join('\n')}

      # Current File
      ${document.fileName}

      # Code Before Cursor
      ${document.getText(new vscode.Range(
        new vscode.Position(Math.max(0, position.line - 10), 0),
        position
      ))}

      Suggest next line of code based on FULL context.
    `;

    const suggestion = await callLLM(prompt);

    return [new vscode.InlineCompletionItem(suggestion)];
  }
}
```

#### C. Cross-Session Learning
```typescript
// extension/src/activation.ts
export async function activate(context: vscode.ExtensionContext) {
  const memory = new CodexMemoryClient();

  // Track every file edit
  vscode.workspace.onDidChangeTextDocument(async (e) => {
    await memory.trackFileEdit(
      e.document.fileName,
      e.document.getText()
    );
  });

  // Track active file switches
  vscode.window.onDidChangeActiveTextEditor(async (editor) => {
    if (editor) {
      await memory.storeContext({
        key: `active_file_${Date.now()}`,
        value: {
          file: editor.document.fileName,
          language: editor.document.languageId,
          lineCount: editor.document.lineCount
        },
        layer: 'session',
        category: 'navigation'
      });
    }
  });

  // Codex chat command with memory
  context.subscriptions.push(
    vscode.commands.registerCommand('codex.chat', async () => {
      const input = await vscode.window.showInputBox({
        prompt: 'Ask Codex anything'
      });

      if (!input) return;

      // Store user message
      await memory.addConversationMessage('user', input);

      // Get context-aware response
      const context = await memory.getRecentContext();
      const response = await getAIResponse(input, context);

      // Store assistant response
      await memory.addConversationMessage('assistant', response);

      // Show response
      vscode.window.showInformationMessage(response);
    })
  );
}
```

---

## 4. Practical Client Library (TypeScript)

Create this in all three projects:

```typescript
// lib/memory-coordination-client.ts

export interface MemoryConfig {
  apiUrl: string;
  tenantId?: string;
  userId?: string;
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
}

export class MemoryClient {
  constructor(private config: MemoryConfig) {}

  async storeContext(entry: Partial<ContextEntry>): Promise<string> {
    const response = await fetch(`${this.config.apiUrl}/memory/context/store`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...entry,
        tenant_id: entry.tenant_id || this.config.tenantId,
        user_id: entry.user_id || this.config.userId,
        source: entry.source || 'client'
      })
    });

    const { entry_id } = await response.json();
    return entry_id;
  }

  async retrieveContext(key: string, scope: string): Promise<any> {
    const response = await fetch(`${this.config.apiUrl}/memory/context/retrieve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        key,
        scope,
        tenant_id: this.config.tenantId,
        user_id: this.config.userId
      })
    });

    return response.json();
  }

  async searchContext(params: {
    query: string;
    scope?: string;
    layer?: string;
    category?: string;
    limit?: number;
  }): Promise<any[]> {
    const response = await fetch(`${this.config.apiUrl}/memory/context/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...params,
        tenant_id: this.config.tenantId
      })
    });

    const { results } = await response.json();
    return results;
  }

  async startSession(sessionId: string): Promise<any> {
    const response = await fetch(`${this.config.apiUrl}/memory/session/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        tenant_id: this.config.tenantId,
        user_id: this.config.userId
      })
    });

    return response.json();
  }

  async resumeSession(sessionId: string): Promise<any> {
    const response = await fetch(
      `${this.config.apiUrl}/memory/session/resume/${sessionId}`,
      { method: 'POST' }
    );

    return response.json();
  }

  async addMessage(sessionId: string, role: string, content: string): Promise<void> {
    await fetch(`${this.config.apiUrl}/memory/session/message`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        role,
        content
      })
    });
  }

  async getSessionContext(sessionId: string): Promise<any> {
    const response = await fetch(
      `${this.config.apiUrl}/memory/session/context/${sessionId}`
    );

    const { context } = await response.json();
    return context;
  }
}
```

---

## 5. Immediate Action Items

### For Weathercraft ERP:
1. Add `memory-coordination-client.ts` to `src/lib/`
2. Add NavigationTracker to root layout
3. Update AI assistant to use context
4. **Result:** AI knows what user is working on

### For MyRoofGenius:
1. Add `memory-coordination-client.ts` to `src/lib/`
2. Track onboarding progress with memory
3. Update AI chat to use sessions
4. **Result:** Conversations persist, AI remembers customers

### For Codex Extension:
1. Add `memoryClient.ts` to extension
2. Track file edits in extension
3. Use context for code suggestions
4. **Result:** Code suggestions are contextual, not generic

---

## 6. Measuring Success

### Before Memory Integration:
- AI gives generic responses ❌
- Users repeat context every time ❌
- No continuity between sessions ❌
- Suggestions are not contextual ❌

### After Memory Integration:
- AI knows full user context ✅
- Conversations pick up where they left off ✅
- Cross-session learning works ✅
- Suggestions are highly relevant ✅

### Metrics to Track:
- **Context hit rate:** % of AI requests with context
- **Session resume rate:** % of sessions resumed vs new
- **User satisfaction:** Survey after using AI
- **Suggestion acceptance:** % of AI suggestions accepted

---

## Next Steps

1. **Copy client library** to each project
2. **Instrument key user flows** (page navigation, chat, file edits)
3. **Update AI prompts** to include context
4. **Measure impact** on user satisfaction

The system is built. Now **use it everywhere**.
