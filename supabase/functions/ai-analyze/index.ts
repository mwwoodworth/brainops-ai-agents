// AI Analyze Edge Function - Analyzes data and provides insights
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.39.1'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface AnalyzeRequest {
  data_type: 'customer' | 'revenue' | 'performance' | 'general'
  entity_id?: string
  time_period?: string
  metrics?: string[]
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const { data_type, entity_id, time_period = '30d', metrics } = await req.json() as AnalyzeRequest

    const supabaseUrl = Deno.env.get('SUPABASE_URL')!
    const supabaseKey = Deno.env.get('SUPABASE_ANON_KEY')!
    const supabase = createClient(supabaseUrl, supabaseKey)

    let analysis: any = {}
    let insights: string[] = []

    switch (data_type) {
      case 'customer':
        // Analyze customer data
        const { data: customerData } = await supabase
          .from('customers')
          .select('*')
          .eq(entity_id ? 'id' : '', entity_id || '')
          .limit(entity_id ? 1 : 100)

        const { data: interactions } = await supabase
          .from('ai_customer_interactions')
          .select('sentiment_score, intent, interaction_type')
          .eq(entity_id ? 'customer_id' : '', entity_id || '')
          .gte('created_at', new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString())

        const avgSentiment = interactions?.reduce((acc, i) => acc + (i.sentiment_score || 0), 0) / (interactions?.length || 1)

        analysis = {
          total_customers: customerData?.length || 0,
          avg_sentiment: avgSentiment.toFixed(2),
          top_intents: getTopItems(interactions?.map(i => i.intent) || []),
          interaction_types: getTopItems(interactions?.map(i => i.interaction_type) || [])
        }

        if (avgSentiment > 0.5) {
          insights.push("Customer sentiment is positive - maintain current engagement strategies")
        } else if (avgSentiment < -0.2) {
          insights.push("Customer sentiment needs improvement - review recent interactions")
        }
        break

      case 'revenue':
        // Analyze revenue data
        const { data: revenueData } = await supabase
          .from('ai_revenue_tracking')
          .select('amount, status, revenue_type')
          .gte('created_at', new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString())

        const totalRevenue = revenueData?.reduce((acc, r) => acc + (Number(r.amount) || 0), 0) || 0
        const avgDealSize = totalRevenue / (revenueData?.length || 1)

        analysis = {
          total_revenue: totalRevenue.toFixed(2),
          deals_count: revenueData?.length || 0,
          avg_deal_size: avgDealSize.toFixed(2),
          revenue_by_type: groupBySum(revenueData || [], 'revenue_type', 'amount')
        }

        if (totalRevenue > 100000) {
          insights.push("Strong revenue performance - consider scaling successful strategies")
        }
        if (avgDealSize < 5000) {
          insights.push("Average deal size is low - focus on upselling and premium offerings")
        }
        break

      case 'performance':
        // Analyze system performance
        const { data: modelData } = await supabase
          .from('ai_trained_models')
          .select('model_type, accuracy, f1_score')
          .order('created_at', { ascending: false })
          .limit(10)

        const avgAccuracy = modelData?.reduce((acc, m) => acc + (m.accuracy || 0), 0) / (modelData?.length || 1)

        analysis = {
          avg_model_accuracy: avgAccuracy.toFixed(3),
          models_count: modelData?.length || 0,
          best_performing_model: modelData?.sort((a, b) => (b.accuracy || 0) - (a.accuracy || 0))[0]?.model_type || 'none'
        }

        if (avgAccuracy < 0.8) {
          insights.push("Model accuracy below target - consider retraining with more data")
        }
        if (avgAccuracy > 0.9) {
          insights.push("Excellent model performance - systems are learning effectively")
        }
        break

      default:
        // General analysis
        const { count: totalInteractions } = await supabase
          .from('ai_customer_interactions')
          .select('*', { count: 'exact', head: true })

        const { count: totalModels } = await supabase
          .from('ai_trained_models')
          .select('*', { count: 'exact', head: true })

        analysis = {
          total_interactions: totalInteractions || 0,
          total_models: totalModels || 0,
          system_health: 'operational'
        }

        insights.push("System is operational and collecting data")
        break
    }

    // Generate AI insights using OpenAI
    const openAiKey = Deno.env.get('OPENAI_API_KEY')!
    const insightPrompt = `
      Analyze this business data and provide 2-3 actionable insights:
      ${JSON.stringify(analysis, null, 2)}

      Context: ${data_type} analysis for ${time_period} period

      Provide specific, actionable recommendations.
    `

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

    const completion = await openAiResponse.json()
    const aiInsights = completion.choices[0].message.content

    // Store the analysis
    await supabase
      .from('ai_learning_insights')
      .insert({
        insight_type: 'analysis',
        category: data_type,
        insight: aiInsights,
        confidence: 0.85,
        impact_score: 0.7,
        metadata: { analysis, time_period }
      })

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

function getTopItems(items: any[], limit = 3): Record<string, number> {
  const counts: Record<string, number> = {}
  items.forEach(item => {
    if (item) counts[item] = (counts[item] || 0) + 1
  })
  return Object.fromEntries(
    Object.entries(counts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, limit)
  )
}

function groupBySum(items: any[], groupKey: string, sumKey: string): Record<string, number> {
  const groups: Record<string, number> = {}
  items.forEach(item => {
    const key = item[groupKey] || 'unknown'
    groups[key] = (groups[key] || 0) + (Number(item[sumKey]) || 0)
  })
  return groups
}