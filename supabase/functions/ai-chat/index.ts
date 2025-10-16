// AI Chat Edge Function
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient, type SupabaseClient } from 'https://esm.sh/@supabase/supabase-js@2.39.1'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
} as const

interface ChatRequest {
  message: string
  conversation_id?: string
  user_id?: string
  context?: Record<string, unknown>
}

interface AiMessageRow {
  id: string
  conversation_id: string
  role: 'user' | 'assistant'
  content: string
  created_at: string
  model_used?: string | null
  tokens_used?: number | null
  user_id?: string | null
  metadata?: Record<string, unknown> | null
}

interface OpenAIChatCompletionResponse {
  choices?: Array<{
    message?: {
      content?: string
    }
  }>
  usage?: {
    total_tokens?: number
  }
}

const getEnvVar = (key: string): string => {
  const value = Deno.env.get(key)
  if (!value) {
    throw new Error(`Missing required environment variable: ${key}`)
  }
  return value
}

const parseRequest = async (req: Request): Promise<ChatRequest> => {
  const body = await req.json().catch(() => null)
  if (!body || typeof body !== 'object') {
    throw new Error('Invalid request body. Expected JSON object.')
  }

  const { message, conversation_id, user_id, context } = body as Record<string, unknown>

  if (typeof message !== 'string' || message.trim().length === 0) {
    throw new Error('The "message" field is required and must be a non-empty string.')
  }

  if (conversation_id !== undefined && typeof conversation_id !== 'string') {
    throw new Error('The "conversation_id" field must be a string when provided.')
  }

  if (user_id !== undefined && typeof user_id !== 'string') {
    throw new Error('The "user_id" field must be a string when provided.')
  }

  if (context !== undefined && (typeof context !== 'object' || context === null || Array.isArray(context))) {
    throw new Error('The "context" field must be an object when provided.')
  }

  return {
    message,
    conversation_id,
    user_id,
    context: context as Record<string, unknown> | undefined,
  }
}

const getSupabaseClient = (): SupabaseClient => {
  const supabaseUrl = getEnvVar('SUPABASE_URL')
  const supabaseKey = getEnvVar('SUPABASE_ANON_KEY')
  return createClient(supabaseUrl, supabaseKey)
}

type MessageInsertPayload = Pick<AiMessageRow, 'conversation_id' | 'role' | 'content' | 'created_at'> &
  Partial<Pick<AiMessageRow, 'user_id' | 'metadata' | 'model_used' | 'tokens_used'>>

const storeMessage = async (
  supabase: SupabaseClient,
  payload: MessageInsertPayload,
): Promise<AiMessageRow> => {
  const response = await supabase
    .from<AiMessageRow>('ai_messages')
    .insert(payload)
    .select()
    .single()

  if (response.error) {
    throw new Error(`Failed to store message: ${response.error.message}`)
  }

  if (!response.data) {
    throw new Error('Message storage returned no data.')
  }

  return response.data
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const { message, conversation_id, user_id, context } = await parseRequest(req)
    const conversationId = conversation_id && conversation_id.trim().length > 0
      ? conversation_id
      : crypto.randomUUID()

    const supabase = getSupabaseClient()

    const userMessage = await storeMessage(supabase, {
      conversation_id: conversationId,
      role: 'user',
      content: message,
      created_at: new Date().toISOString(),
      user_id: user_id ?? null,
      metadata: context ?? null,
    })

    const openAiKey = getEnvVar('OPENAI_API_KEY')

    const openAiResponse = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${openAiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-4',
        messages: [
          {
            role: 'system',
            content: 'You are a helpful AI assistant for BrainOps, specializing in business automation and AI solutions.'
          },
          {
            role: 'user',
            content: message
          }
        ],
        temperature: 0.7,
        max_tokens: 500
      }),
    })

    if (!openAiResponse.ok) {
      const errorText = await openAiResponse.text()
      throw new Error(`OpenAI API request failed: ${errorText}`)
    }

    const completion = await openAiResponse.json() as OpenAIChatCompletionResponse
    const aiResponse = completion.choices?.[0]?.message?.content?.trim()

    if (!aiResponse) {
      throw new Error('OpenAI returned an empty response.')
    }

    const assistantMessage = await storeMessage(supabase, {
      conversation_id: userMessage.conversation_id,
      role: 'assistant',
      content: aiResponse,
      model_used: 'gpt-4',
      tokens_used: completion.usage?.total_tokens ?? null,
      created_at: new Date().toISOString(),
      user_id: user_id ?? null,
      metadata: context ?? null,
    })

    return new Response(
      JSON.stringify({
        success: true,
        conversation_id: userMessage.conversation_id,
        response: aiResponse,
        message_id: assistantMessage.id
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200,
      }
    )
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred.'
    return new Response(
      JSON.stringify({
        success: false,
        error: errorMessage
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 400,
      }
    )
  }
})
