# Memory Coordination Client Libraries
**Make Your Apps Actually Use The Memory System**

---

## Quick Start

### 1. For Weathercraft ERP / MyRoofGenius (Next.js/React)

**Copy this file:**
```bash
cp typescript/memory-client.ts <your-project>/src/lib/memory-client.ts
```

**Use it:**
```typescript
// src/lib/memory.ts
import { MemoryClient } from './memory-client';

export const memory = new MemoryClient({
  apiUrl: 'https://brainops-ai-agents.onrender.com',
  tenantId: 'weathercraft-erp', // or 'myroofgenius'
  userId: getCurrentUserId() // Your auth system
});

// Track user actions automatically
export async function trackPageView(pathname: string) {
  await memory.trackAction('page_view', { pathname });
}

// Get AI context before LLM calls
export async function getAIContext() {
  return await memory.getAIContext();
}
```

**Add to layout:**
```typescript
// app/layout.tsx
'use client';

import { usePathname } from 'next/navigation';
import { useEffect } from 'react';
import { trackPageView } from '@/lib/memory';

export function MemoryTracker() {
  const pathname = usePathname();

  useEffect(() => {
    trackPageView(pathname);
  }, [pathname]);

  return null;
}

// In your layout:
<MemoryTracker />
```

### 2. For Codex VSCode Extension

**Copy this file:**
```bash
cp vscode-extension/codexMemoryExtension.ts <codex-extension>/src/
```

**Activate in extension.ts:**
```typescript
import { CodexMemoryExtension } from './codexMemoryExtension';

export async function activate(context: vscode.ExtensionContext) {
  const codexMemory = new CodexMemoryExtension(context);
  await codexMemory.activate();

  // Before calling LLM:
  async function getCodeSuggestion(userInput: string) {
    // Get full context (files, conversation, etc.)
    const prompt = await codexMemory.buildAIPrompt(userInput);

    // Call LLM with context
    const response = await callLLM(prompt);

    // Track the conversation
    await codexMemory.addConversationMessage({
      role: 'user',
      content: userInput
    });
    await codexMemory.addConversationMessage({
      role: 'assistant',
      content: response
    });

    return response;
  }
}
```

**User Commands (auto-registered):**
- `Codex: Show Memory Stats` - See what's remembered
- `Codex: Clear Memory` - Start fresh
- `Codex: View Current Context` - Debug memory state

---

## What This Actually Does

### Without Memory (Current State):
```
User: "Fix the authentication bug"
AI: "Which bug? What file? What error?"  ❌

User switches files
AI: "I have no idea what you were just working on" ❌

User comes back next day
AI: "I don't remember yesterday's conversation" ❌
```

### With Memory:
```
User: "Fix the authentication bug"
AI: "I see you were just editing auth.ts, got a 401 error on line 42,
     and yesterday we discussed JWT expiration. Let me check..." ✅

User switches files
AI: "Noticed you moved to database.ts. This connects to the auth
     issue since you need to verify tokens in the DB layer." ✅

User comes back next day
AI: "Welcome back! You were fixing auth. The JWT fix is deployed.
     Want to continue with the user permissions feature?" ✅
```

---

## Real-World Examples

### Example 1: Weathercraft ERP - AI Assistant Knows What User Is Doing

```typescript
// When user opens customer page
async function loadCustomer(customerId: string) {
  // Store current context
  await memory.storeContext({
    key: 'current_customer',
    value: { customerId, name: customer.name },
    layer: 'session',
    scope: 'user',
    priority: 'high',
    category: 'current_context'
  });

  // AI now knows user is working with this customer
}

// AI assistant button click
async function getAIHelp(userQuestion: string) {
  const context = await memory.getAIContext();

  const prompt = `
    User is viewing customer: ${context.sessionContext?.memory.working_memory.current_customer?.name}
    Recent actions: ${context.recentActions.slice(0, 5).map(a => a.type).join(', ')}
    User asks: ${userQuestion}

    Provide contextual help.
  `;

  // AI response is contextual, not generic!
  return await callLLM(prompt);
}
```

### Example 2: MyRoofGenius - Onboarding That Resumes

```typescript
// Track onboarding progress
async function completeOnboardingStep(step: string) {
  await memory.storeContext({
    key: `onboarding_step_${step}`,
    value: { completed: true, timestamp: new Date() },
    layer: 'long_term', // Persists beyond session
    scope: 'user',
    priority: 'high',
    category: 'onboarding'
  });
}

// Resume onboarding
async function getOnboardingProgress() {
  const steps = await memory.searchContext({
    query: 'onboarding',
    scope: 'user',
    category: 'onboarding'
  });

  const completed = steps.filter(s => s.value.completed);
  return { completed, nextStep: STEPS[completed.length] };
}

// User logs in next day → goes right to next step ✅
```

### Example 3: Codex - Code Suggestions With Full Context

```typescript
// Codex tracks every file edit automatically
// When user asks for help:

async function provideCodeSuggestion(position: vscode.Position) {
  const context = await codexMemory.getContextForAI();

  const prompt = `
    Files recently edited:
    ${context.recentFiles.map(f => `- ${f.fileName} (${f.language})`).join('\n')}

    Current file: ${context.currentFile?.fileName}
    Cursor at line: ${context.currentFile?.cursorLine}

    Recent conversation:
    ${context.recentConversation.slice(-3).map(m => `${m.role}: ${m.content}`).join('\n')}

    Suggest code for current position based on full context.
  `;

  // Suggestions are now hyper-contextual ✅
  return await callLLM(prompt);
}
```

---

## Implementation Checklist

### For Weathercraft ERP:
- [ ] Copy `typescript/memory-client.ts` to `src/lib/`
- [ ] Add `MemoryTracker` component to root layout
- [ ] Update AI assistant to use `getAIContext()`
- [ ] Track critical user actions (customer views, edits, etc.)
- **Result:** AI assistant knows what user is working on

### For MyRoofGenius:
- [ ] Copy `typescript/memory-client.ts` to `src/lib/`
- [ ] Track onboarding progress with memory
- [ ] Store user preferences permanently
- [ ] Update AI chat to use session context
- **Result:** Conversations persist, onboarding resumes

### For Codex Extension:
- [ ] Copy `vscode-extension/codexMemoryExtension.ts` to `src/`
- [ ] Activate in `extension.ts`
- [ ] Update LLM calls to use `buildAIPrompt()`
- [ ] Track conversations with `addConversationMessage()`
- **Result:** Code suggestions are contextual, not generic

---

## Measuring Success

### Metrics to Track:

1. **Context Hit Rate**
   ```sql
   SELECT
     COUNT(*) FILTER (WHERE hit_cache) * 100.0 / COUNT(*) as hit_rate
   FROM memory_context_access_log;
   ```

2. **Session Resume Rate**
   ```sql
   SELECT
     COUNT(*) FILTER (WHERE status = 'resumed') * 100.0 / COUNT(*)
   FROM memory_session_context;
   ```

3. **AI Response Quality**
   - Before: "I don't have context" (generic responses)
   - After: "Based on your recent edits to auth.ts..." (contextual)

### Expected Improvements:
- **AI accuracy:** +50% (has context vs no context)
- **User frustration:** -80% (no repeating context)
- **Session continuity:** 100% (resume from anywhere)

---

## Troubleshooting

### "Memory API error: 500"
- Check API is deployed: `curl https://brainops-ai-agents.onrender.com/memory/health`
- Verify database connection in health response

### "No context retrieved"
- Check `sessionId` is being stored
- Verify `tenantId` and `userId` are set
- Use `/memory/health` endpoint to check stats

### "Codex commands not showing"
- Add commands to `package.json` (see codexMemoryExtension.ts comments)
- Reload VSCode window after installation

---

## Next Steps

1. **Choose one app** (ERP, MRG, or Codex)
2. **Copy the client library** to your project
3. **Add tracking** to 1-2 key user flows
4. **Test it** - verify context is stored and retrieved
5. **Expand** - add more tracking points once working

**The system is built and deployed. Now integrate it and make your AI actually smart.**

---

## Support

- **API Docs:** https://brainops-ai-agents.onrender.com/docs
- **Health Check:** https://brainops-ai-agents.onrender.com/memory/health
- **Integration Guide:** See `PRACTICAL_INTEGRATION_GUIDE.md`
- **System Docs:** See `MEMORY_COORDINATION_SYSTEM.md`
