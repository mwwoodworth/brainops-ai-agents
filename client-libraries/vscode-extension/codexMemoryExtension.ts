/**
 * Codex Memory Extension for VSCode
 * Enables persistent AI memory across coding sessions
 *
 * Installation for Codex:
 * 1. Copy this file to your Codex extension's src/ directory
 * 2. Import and activate in extension.ts
 * 3. Codex will now remember everything across sessions
 */

import * as vscode from 'vscode';

interface MemoryConfig {
  apiUrl: string;
  sessionId?: string;
}

interface FileContext {
  fileName: string;
  language: string;
  content: string;
  lineCount: number;
  cursorLine: number;
  timestamp: string;
}

interface ConversationMessage {
  role: 'user' | 'assistant';
  content: string;
  metadata?: {
    activeFile?: string;
    cursorPosition?: vscode.Position;
    selectedText?: string;
  };
}

export class CodexMemoryExtension {
  private apiUrl = 'https://brainops-ai-agents.onrender.com';
  private sessionId: string;
  private context: vscode.ExtensionContext;

  constructor(context: vscode.ExtensionContext) {
    this.context = context;
    this.sessionId = this.getOrCreateSessionId();
  }

  /**
   * Activate the extension - call this from extension.ts activate()
   */
  async activate() {
    console.log('Codex Memory Extension activating...');

    // Resume or start session
    await this.resumeOrStartSession();

    // Register event listeners
    this.registerFileEditTracking();
    this.registerActiveFileTracking();
    this.registerConversationTracking();

    // Register commands
    this.registerCommands();

    console.log('Codex Memory Extension activated!');
    vscode.window.showInformationMessage(
      'Codex Memory: Now remembering context across sessions!'
    );
  }

  // ========================================================================
  // SESSION MANAGEMENT
  // ========================================================================

  private getOrCreateSessionId(): string {
    // Try to get existing session from global state
    let sessionId = this.context.globalState.get<string>('codex.sessionId');

    if (!sessionId) {
      // Create new session ID
      sessionId = `codex_${vscode.env.machineId}_${Date.now()}`;
      this.context.globalState.update('codex.sessionId', sessionId);
    }

    return sessionId;
  }

  private async resumeOrStartSession() {
    try {
      // Try to resume existing session
      const response = await fetch(
        `${this.apiUrl}/memory/session/resume/${this.sessionId}`,
        { method: 'POST' }
      );

      if (response.ok) {
        const data = await response.json();
        console.log(`Resumed session: ${data.message_count} messages`);
        vscode.window.showInformationMessage(
          `Codex Memory: Resumed session with ${data.message_count} messages`
        );
      } else {
        // Session doesn't exist, start new one
        await this.startNewSession();
      }
    } catch (error) {
      console.error('Error resuming session:', error);
      await this.startNewSession();
    }
  }

  private async startNewSession() {
    try {
      const workspace = vscode.workspace.workspaceFolders?.[0];

      const response = await fetch(`${this.apiUrl}/memory/session/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: this.sessionId,
          tenant_id: 'codex_vscode',
          user_id: vscode.env.machineId,
          initial_context: {
            workspace: workspace?.uri.fsPath,
            workspaceName: workspace?.name,
            language: vscode.env.language,
            vscodeVersion: vscode.version
          }
        })
      });

      if (response.ok) {
        console.log('Started new session');
        vscode.window.showInformationMessage('Codex Memory: New session started');
      }
    } catch (error) {
      console.error('Error starting session:', error);
    }
  }

  // ========================================================================
  // FILE TRACKING
  // ========================================================================

  private registerFileEditTracking() {
    // Track text document changes
    vscode.workspace.onDidChangeTextDocument(async (e) => {
      // Only track meaningful changes (not just cursor movement)
      if (e.contentChanges.length === 0) return;

      const fileContext = this.getFileContext(e.document);

      await this.storeContext({
        key: `file_edit_${fileContext.fileName}_${Date.now()}`,
        value: {
          ...fileContext,
          changes: e.contentChanges.map(c => ({
            range: {
              start: { line: c.range.start.line, char: c.range.start.character },
              end: { line: c.range.end.line, char: c.range.end.character }
            },
            text: c.text.substring(0, 500) // First 500 chars
          }))
        },
        layer: 'session',
        scope: 'session',
        priority: 'medium',
        category: 'file_edit'
      });
    });

    // Track file saves
    vscode.workspace.onDidSaveTextDocument(async (document) => {
      const fileContext = this.getFileContext(document);

      await this.storeContext({
        key: `file_save_${fileContext.fileName}_${Date.now()}`,
        value: fileContext,
        layer: 'session',
        scope: 'session',
        priority: 'high',
        category: 'file_save'
      });
    });
  }

  private registerActiveFileTracking() {
    // Track when user switches files
    vscode.window.onDidChangeActiveTextEditor(async (editor) => {
      if (!editor) return;

      const fileContext = this.getFileContext(editor.document);

      await this.storeContext({
        key: `active_file_${Date.now()}`,
        value: fileContext,
        layer: 'session',
        scope: 'session',
        priority: 'medium',
        category: 'navigation'
      });

      // Store as current context (for AI to know what user is looking at)
      await this.storeContext({
        key: `current_file`,
        value: fileContext,
        layer: 'ephemeral',
        scope: 'session',
        priority: 'high',
        category: 'current_context'
      });
    });
  }

  // ========================================================================
  // CONVERSATION TRACKING
  // ========================================================================

  private registerConversationTracking() {
    // This will be called by Codex when user/AI exchanges messages
    // Codex should call: codexMemory.addConversationMessage(...)
  }

  async addConversationMessage(message: ConversationMessage) {
    await fetch(`${this.apiUrl}/memory/session/message`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: this.sessionId,
        role: message.role,
        content: message.content,
        metadata: message.metadata || {}
      })
    });
  }

  // ========================================================================
  // CONTEXT RETRIEVAL FOR AI
  // ========================================================================

  /**
   * Get full context for AI prompts
   * Call this before sending prompt to LLM
   */
  async getContextForAI(): Promise<{
    currentFile?: FileContext;
    recentFiles: FileContext[];
    recentConversation: any[];
    workingMemory: Record<string, any>;
  }> {
    try {
      const response = await fetch(
        `${this.apiUrl}/memory/session/context/${this.sessionId}`
      );

      if (!response.ok) {
        return this.getEmptyContext();
      }

      const { context } = await response.json();

      return {
        currentFile: context.memory.working_memory.current_file,
        recentFiles: context.memory.working_memory.recent_files || [],
        recentConversation: context.conversation.recent_messages.slice(-10),
        workingMemory: context.memory.working_memory
      };
    } catch (error) {
      console.error('Error getting AI context:', error);
      return this.getEmptyContext();
    }
  }

  private getEmptyContext() {
    return {
      currentFile: undefined,
      recentFiles: [],
      recentConversation: [],
      workingMemory: {}
    };
  }

  /**
   * Build AI prompt with full context
   */
  async buildAIPrompt(userQuery: string): Promise<string> {
    const context = await this.getContextForAI();

    let prompt = '# Context\n\n';

    // Current file
    if (context.currentFile) {
      prompt += `## Currently Viewing\n`;
      prompt += `File: ${context.currentFile.fileName}\n`;
      prompt += `Language: ${context.currentFile.language}\n`;
      prompt += `Lines: ${context.currentFile.lineCount}\n`;
      prompt += `Cursor: Line ${context.currentFile.cursorLine}\n\n`;
    }

    // Recent files
    if (context.recentFiles.length > 0) {
      prompt += `## Recently Edited Files\n`;
      for (const file of context.recentFiles.slice(0, 5)) {
        prompt += `- ${file.fileName} (${file.language})\n`;
      }
      prompt += '\n';
    }

    // Recent conversation
    if (context.recentConversation.length > 0) {
      prompt += `## Recent Conversation\n`;
      for (const msg of context.recentConversation.slice(-5)) {
        prompt += `${msg.role}: ${msg.content.substring(0, 200)}\n`;
      }
      prompt += '\n';
    }

    // User's current question
    prompt += `# User Question\n\n${userQuery}\n`;

    return prompt;
  }

  // ========================================================================
  // COMMANDS
  // ========================================================================

  private registerCommands() {
    // Command: Get session stats
    this.context.subscriptions.push(
      vscode.commands.registerCommand('codex.memory.stats', async () => {
        const context = await this.getContextForAI();
        const stats = `
Codex Memory Stats:
- Session ID: ${this.sessionId}
- Files edited this session: ${context.recentFiles.length}
- Conversation messages: ${context.recentConversation.length}
- Current file: ${context.currentFile?.fileName || 'None'}
        `.trim();

        vscode.window.showInformationMessage(stats);
      })
    );

    // Command: Clear session (start fresh)
    this.context.subscriptions.push(
      vscode.commands.registerCommand('codex.memory.clear', async () => {
        const confirm = await vscode.window.showWarningMessage(
          'Clear Codex memory? This will end the current session and start fresh.',
          'Yes',
          'No'
        );

        if (confirm === 'Yes') {
          await this.endSession();
          this.sessionId = `codex_${vscode.env.machineId}_${Date.now()}`;
          this.context.globalState.update('codex.sessionId', this.sessionId);
          await this.startNewSession();
          vscode.window.showInformationMessage('Codex memory cleared!');
        }
      })
    );

    // Command: View recent context
    this.context.subscriptions.push(
      vscode.commands.registerCommand('codex.memory.viewContext', async () => {
        const context = await this.getContextForAI();

        const document = await vscode.workspace.openTextDocument({
          content: JSON.stringify(context, null, 2),
          language: 'json'
        });

        await vscode.window.showTextDocument(document);
      })
    );
  }

  // ========================================================================
  // UTILITIES
  // ========================================================================

  private getFileContext(document: vscode.TextDocument): FileContext {
    const editor = vscode.window.activeTextEditor;
    const cursorLine = editor?.selection.active.line || 0;

    return {
      fileName: document.fileName,
      language: document.languageId,
      content: document.getText().substring(0, 10000), // First 10KB
      lineCount: document.lineCount,
      cursorLine,
      timestamp: new Date().toISOString()
    };
  }

  private async storeContext(entry: any) {
    try {
      await fetch(`${this.apiUrl}/memory/context/store`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...entry,
          session_id: this.sessionId,
          source: 'codex_vscode'
        })
      });
    } catch (error) {
      console.error('Error storing context:', error);
    }
  }

  private async endSession() {
    try {
      await fetch(`${this.apiUrl}/memory/session/end/${this.sessionId}`, {
        method: 'POST'
      });
    } catch (error) {
      console.error('Error ending session:', error);
    }
  }
}

// ========================================================================
// INTEGRATION WITH CODEX EXTENSION
// ========================================================================

/**
 * Add this to your extension.ts activate() function:
 *
 * import { CodexMemoryExtension } from './codexMemoryExtension';
 *
 * export async function activate(context: vscode.ExtensionContext) {
 *   // Initialize memory extension
 *   const codexMemory = new CodexMemoryExtension(context);
 *   await codexMemory.activate();
 *
 *   // Store reference for other parts of extension
 *   context.workspaceState.update('codexMemory', codexMemory);
 *
 *   // When Codex generates completions/responses:
 *   async function getCodeSuggestion(userInput: string) {
 *     // Get context for AI
 *     const prompt = await codexMemory.buildAIPrompt(userInput);
 *
 *     // Call LLM with contextual prompt
 *     const response = await callLLM(prompt);
 *
 *     // Track conversation
 *     await codexMemory.addConversationMessage({
 *       role: 'user',
 *       content: userInput,
 *       metadata: {
 *         activeFile: vscode.window.activeTextEditor?.document.fileName
 *       }
 *     });
 *
 *     await codexMemory.addConversationMessage({
 *       role: 'assistant',
 *       content: response
 *     });
 *
 *     return response;
 *   }
 * }
 */

// ========================================================================
// PACKAGE.JSON ADDITIONS
// ========================================================================

/**
 * Add these commands to package.json:
 *
 * "contributes": {
 *   "commands": [
 *     {
 *       "command": "codex.memory.stats",
 *       "title": "Codex: Show Memory Stats"
 *     },
 *     {
 *       "command": "codex.memory.clear",
 *       "title": "Codex: Clear Memory (Start Fresh)"
 *     },
 *     {
 *       "command": "codex.memory.viewContext",
 *       "title": "Codex: View Current Context"
 *     }
 *   ]
 * }
 */
