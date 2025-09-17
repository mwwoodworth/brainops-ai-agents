// AI Chat Edge Function
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.39.1'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface ChatRequest {
  message: string
  conversation_id?: string
  user_id?: string
  context?: Record<string, any>
}

serve(async (req) => {
  // Handle CORS
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const { message, conversation_id, user_id, context } = await req.json() as ChatRequest

    // Initialize Supabase client
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!
    const supabaseKey = Deno.env.get('SUPABASE_ANON_KEY')!
    const supabase = createClient(supabaseUrl, supabaseKey)

    // Store the message
    const { data: messageData, error: messageError } = await supabase
      .from('ai_messages')
      .insert({
        conversation_id: conversation_id || crypto.randomUUID(),
        role: 'user',
        content: message,
        created_at: new Date().toISOString()
      })
      .select()
      .single()

    if (messageError) throw messageError

    // Get OpenAI API key from environment
    const openAiKey = Deno.env.get('OPENAI_API_KEY')!

    // Call OpenAI API
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

    const completion = await openAiResponse.json()
    const aiResponse = completion.choices[0].message.content

    // Store AI response
    const { data: aiMessageData, error: aiMessageError } = await supabase
      .from('ai_messages')
      .insert({
        conversation_id: messageData.conversation_id,
        role: 'assistant',
        content: aiResponse,
        model_used: 'gpt-4',
        tokens_used: completion.usage?.total_tokens || 0,
        created_at: new Date().toISOString()
      })
      .select()
      .single()

    if (aiMessageError) throw aiMessageError

    // Return the response
    return new Response(
      JSON.stringify({
        success: true,
        conversation_id: messageData.conversation_id,
        response: aiResponse,
        message_id: aiMessageData.id
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200,
      }
    )
  } catch (error) {
    return new Response(
      JSON.stringify({
        success: false,
        error: error.message
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 400,
      }
    )
  }
})