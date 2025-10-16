// AI Analyze Edge Function - Analyzes data and provides insights
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient, type SupabaseClient } from 'https://esm.sh/@supabase/supabase-js@2.39.1'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
} as const

type DataType = 'customer' | 'revenue' | 'performance' | 'general'

interface AnalyzeRequest {
  data_type: DataType
  entity_id?: string
  time_period?: string
  metrics?: string[]
}

interface CustomerRow {
  id?: string
  company_name?: string | null
  last_contact?: string | null
}

interface CustomerInteractionRow {
  sentiment_score?: number | null
  intent?: string | null
  interaction_type?: string | null
  customer_id?: string | null
}

interface RevenueRow {
  amount?: number | string | null
  status?: string | null
  revenue_type?: string | null
}

interface ModelRow {
  model_type?: string | null
  accuracy?: number | null
  f1_score?: number | null
}

interface OpenAIChatCompletionResponse {
  choices?: Array<{
    message?: {
      content?: string
    }
  }>
}

const getEnvVar = (key: string): string => {
  const value = Deno.env.get(key)
  if (!value) {
    throw new Error(`Missing required environment variable: ${key}`)
  }
  return value
}

const getSupabaseClient = (): SupabaseClient => {
  const supabaseUrl = getEnvVar('SUPABASE_URL')
  const supabaseKey = getEnvVar('SUPABASE_ANON_KEY')
  return createClient(supabaseUrl, supabaseKey)
}

const parseAnalyzeRequest = async (req: Request): Promise<AnalyzeRequest> => {
  const body = await req.json().catch(() => null)
  if (!body || typeof body !== 'object') {
    throw new Error('Invalid request body. Expected JSON object.')
  }

  const { data_type, entity_id, time_period, metrics } = body as Record<string, unknown>

  if (data_type !== 'customer' && data_type !== 'revenue' && data_type !== 'performance' && data_type !== 'general') {
    throw new Error('The "data_type" field is required and must be one of customer, revenue, performance, or general.')
  }

  if (entity_id !== undefined && typeof entity_id !== 'string') {
    throw new Error('The "entity_id" field must be a string when provided.')
  }

  if (time_period !== undefined && typeof time_period !== 'string') {
    throw new Error('The "time_period" field must be a string when provided.')
  }

  if (metrics !== undefined && !Array.isArray(metrics)) {
    throw new Error('The "metrics" field must be an array when provided.')
  }

  return {
    data_type,
    entity_id,
    time_period: typeof time_period === 'string' && time_period.trim().length > 0 ? time_period : '30d',
    metrics: Array.isArray(metrics) ? metrics.map(String) : undefined,
  }
}

const safeNumber = (value: unknown): number => {
  const num = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(num) ? num : 0
}

const average = (values: number[]): number => {
  if (values.length === 0) {
    return 0
  }
  const sum = values.reduce((acc, value) => acc + value, 0)
  return sum / values.length
}

const getTopItems = (items: Array<string | null | undefined>, limit = 3): Record<string, number> => {
  const counts = new Map<string, number>()
  for (const item of items) {
    if (!item) {
      continue
    }
    counts.set(item, (counts.get(item) ?? 0) + 1)
  }

  return Object.fromEntries(
    Array.from(counts.entries())
      .sort(([, a], [, b]) => b - a)
      .slice(0, limit),
  )
}

const groupBySum = (items: RevenueRow[], groupKey: keyof RevenueRow, sumKey: keyof RevenueRow): Record<string, number> => {
  const groups = new Map<string, number>()
  for (const item of items) {
    const rawKey = item[groupKey]
    const key = typeof rawKey === 'string' && rawKey.trim().length > 0 ? rawKey : 'unknown'
    const currentTotal = groups.get(key) ?? 0
    groups.set(key, currentTotal + safeNumber(item[sumKey]))
  }
  return Object.fromEntries(groups.entries())
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const { data_type, entity_id, time_period } = await parseAnalyzeRequest(req)
    const supabase = getSupabaseClient()

    let analysis: Record<string, unknown> = {}
    const insights: string[] = []

    switch (data_type) {
      case 'customer': {
        const customerResponse = entity_id
          ? await supabase
            .from<CustomerRow>('customers')
            .select('id, company_name, last_contact')
            .eq('id', entity_id)
            .limit(1)
          : await supabase
            .from<CustomerRow>('customers')
            .select('id, company_name, last_contact')
            .limit(100)

        if (customerResponse.error) {
          throw new Error(`Failed to fetch customers: ${customerResponse.error.message}`)
        }

        const customers = customerResponse.data ?? []

        const interactionsQuery = supabase
          .from<CustomerInteractionRow>('ai_customer_interactions')
          .select('sentiment_score, intent, interaction_type, customer_id')
          .gte('created_at', new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString())

        if (entity_id) {
          interactionsQuery.eq('customer_id', entity_id)
        }

        const interactionsResponse = await interactionsQuery

        if (interactionsResponse.error) {
          throw new Error(`Failed to fetch interactions: ${interactionsResponse.error.message}`)
        }

        const interactions = interactionsResponse.data ?? []
        const sentimentValues = interactions.map((interaction) => safeNumber(interaction.sentiment_score))
        const avgSentiment = average(sentimentValues)

        analysis = {
          total_customers: customers.length,
          avg_sentiment: avgSentiment.toFixed(2),
          top_intents: getTopItems(interactions.map((interaction) => interaction.intent)),
          interaction_types: getTopItems(interactions.map((interaction) => interaction.interaction_type)),
        }

        if (avgSentiment > 0.5) {
          insights.push('Customer sentiment is positive - maintain current engagement strategies')
        } else if (avgSentiment < -0.2) {
          insights.push('Customer sentiment needs improvement - review recent interactions')
        }
        break
      }

      case 'revenue': {
        const revenueResponse = await supabase
          .from<RevenueRow>('ai_revenue_tracking')
          .select('amount, status, revenue_type, created_at')
          .gte('created_at', new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString())

        if (revenueResponse.error) {
          throw new Error(`Failed to fetch revenue data: ${revenueResponse.error.message}`)
        }

        const revenueData = revenueResponse.data ?? []
        const totalRevenue = revenueData.reduce((acc, record) => acc + safeNumber(record.amount), 0)
        const avgDealSize = revenueData.length > 0 ? totalRevenue / revenueData.length : 0

        analysis = {
          total_revenue: totalRevenue.toFixed(2),
          deals_count: revenueData.length,
          avg_deal_size: avgDealSize.toFixed(2),
          revenue_by_type: groupBySum(revenueData, 'revenue_type', 'amount'),
        }

        if (totalRevenue > 100_000) {
          insights.push('Strong revenue performance - consider scaling successful strategies')
        }
        if (avgDealSize < 5_000) {
          insights.push('Average deal size is low - focus on upselling and premium offerings')
        }
        break
      }

      case 'performance': {
        const modelResponse = await supabase
          .from<ModelRow>('ai_trained_models')
          .select('model_type, accuracy, f1_score, created_at')
          .order('created_at', { ascending: false })
          .limit(10)

        if (modelResponse.error) {
          throw new Error(`Failed to fetch model performance: ${modelResponse.error.message}`)
        }

        const models = modelResponse.data ?? []
        const accuracyValues = models.map((model) => safeNumber(model.accuracy))
        const avgAccuracy = average(accuracyValues)

        const bestModel = models.reduce<ModelRow | null>((best, current) => {
          if (!best || safeNumber(current.accuracy) > safeNumber(best.accuracy)) {
            return current
          }
          return best
        }, null)

        analysis = {
          avg_model_accuracy: avgAccuracy.toFixed(3),
          models_count: models.length,
          best_performing_model: bestModel?.model_type ?? 'none',
        }

        if (avgAccuracy < 0.8) {
          insights.push('Model accuracy below target - consider retraining with more data')
        }
        if (avgAccuracy > 0.9) {
          insights.push('Excellent model performance - systems are learning effectively')
        }
        break
      }

      case 'general':
      default: {
        const interactionsCountResponse = await supabase
          .from('ai_customer_interactions')
          .select('*', { count: 'exact', head: true })

        if (interactionsCountResponse.error) {
          throw new Error(`Failed to count customer interactions: ${interactionsCountResponse.error.message}`)
        }

        const modelsCountResponse = await supabase
          .from('ai_trained_models')
          .select('*', { count: 'exact', head: true })

        if (modelsCountResponse.error) {
          throw new Error(`Failed to count trained models: ${modelsCountResponse.error.message}`)
        }

        analysis = {
          total_interactions: interactionsCountResponse.count ?? 0,
          total_models: modelsCountResponse.count ?? 0,
          system_health: 'operational',
        }

        insights.push('System is operational and collecting data')
        break
      }
    }

    const openAiKey = getEnvVar('OPENAI_API_KEY')
    const insightPrompt = `Analyze this business data and provide 2-3 actionable insights.\n\nContext: ${data_type} analysis for ${time_period} period.\n\nData:${JSON.stringify(analysis, null, 2)}`

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
            content: 'You are a business analyst AI providing actionable insights from data.'
          },
          {
            role: 'user',
            content: insightPrompt
          }
        ],
        temperature: 0.3,
        max_tokens: 300
      }),
    })

    if (!openAiResponse.ok) {
      const errorText = await openAiResponse.text()
      throw new Error(`OpenAI API request failed: ${errorText}`)
    }

    const completion = await openAiResponse.json() as OpenAIChatCompletionResponse
    const aiInsights = completion.choices?.[0]?.message?.content?.trim()

    if (!aiInsights) {
      throw new Error('OpenAI returned an empty insight response.')
    }

    const insertInsightResponse = await supabase
      .from('ai_learning_insights')
      .insert({
        insight_type: 'analysis',
        category: data_type,
        insight: aiInsights,
        confidence: 0.85,
        impact_score: 0.7,
        metadata: { analysis, time_period },
        created_at: new Date().toISOString(),
      })

    if (insertInsightResponse.error) {
      throw new Error(`Failed to store insights: ${insertInsightResponse.error.message}`)
    }

    return new Response(
      JSON.stringify({
        success: true,
        analysis,
        insights: [...insights, aiInsights],
        timestamp: new Date().toISOString()
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
