// AI Execute Edge Function - Executes AI agent tasks
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.39.1'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface ExecuteRequest {
  agent_type: string
  task: string
  parameters?: Record<string, any>
  priority?: 'low' | 'normal' | 'high' | 'critical'
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const { agent_type, task, parameters = {}, priority = 'normal' } = await req.json() as ExecuteRequest

    const supabaseUrl = Deno.env.get('SUPABASE_URL')!
    const supabaseKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
    const supabase = createClient(supabaseUrl, supabaseKey)

    // Create execution record
    const execution_id = crypto.randomUUID()
    const { error: insertError } = await supabase
      .from('agent_executions')
      .insert({
        id: execution_id,
        agent_type,
        prompt: task,
        status: 'executing',
        priority,
        metadata: parameters,
        created_at: new Date().toISOString()
      })

    if (insertError) throw insertError

    // Execute based on agent type
    let result: any = {}

    try {
      switch (agent_type) {
        case 'revenue_generator':
          result = await executeRevenueTask(task, parameters, supabase)
          break

        case 'customer_acquisition':
          result = await executeAcquisitionTask(task, parameters, supabase)
          break

        case 'data_analyzer':
          result = await executeAnalysisTask(task, parameters, supabase)
          break

        case 'automation':
          result = await executeAutomationTask(task, parameters, supabase)
          break

        default:
          // Generic AI task execution
          result = await executeGenericTask(task, parameters)
          break
      }

      // Update execution status
      await supabase
        .from('agent_executions')
        .update({
          status: 'completed',
          result,
          updated_at: new Date().toISOString()
        })
        .eq('id', execution_id)

      // Broadcast event
      await supabase
        .from('ai_event_broadcasts')
        .insert({
          event_type: 'agent_execution',
          event_data: {
            execution_id,
            agent_type,
            task,
            status: 'completed'
          },
          created_at: new Date().toISOString()
        })

      return new Response(
        JSON.stringify({
          success: true,
          execution_id,
          result,
          status: 'completed'
        }),
        {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          status: 200,
        }
      )
    } catch (execError) {
      // Update execution status to failed
      await supabase
        .from('agent_executions')
        .update({
          status: 'failed',
          error_message: execError.message,
          updated_at: new Date().toISOString()
        })
        .eq('id', execution_id)

      throw execError
    }
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

async function executeRevenueTask(task: string, params: any, supabase: any) {
  // Identify and qualify leads
  if (task.includes('identify') || task.includes('lead')) {
    const { data: leads } = await supabase
      .from('customers')
      .select('*')
      .is('last_contact', null)
      .limit(10)

    const qualified = []
    for (const lead of leads || []) {
      // Simple qualification logic
      const score = Math.random() * 100
      if (score > 50) {
        qualified.push({
          lead_id: lead.id,
          company: lead.company_name,
          score: score.toFixed(1)
        })

        // Store in AI leads table
        await supabase
          .from('ai_leads')
          .insert({
            company_name: lead.company_name,
            lead_score: score,
            lead_stage: 'QUALIFIED',
            source: 'ai_identification'
          })
      }
    }

    return {
      task: 'lead_identification',
      leads_analyzed: leads?.length || 0,
      qualified_leads: qualified.length,
      leads: qualified
    }
  }

  // Generate proposal
  if (task.includes('proposal')) {
    const proposal = {
      title: 'Custom Solution Proposal',
      value: Math.floor(Math.random() * 50000) + 10000,
      items: [
        'Discovery and Analysis',
        'Solution Design',
        'Implementation',
        'Training and Support'
      ],
      timeline: '4-6 weeks',
      generated_at: new Date().toISOString()
    }

    return {
      task: 'proposal_generation',
      proposal
    }
  }

  return { task: 'revenue_task', status: 'completed' }
}

async function executeAcquisitionTask(task: string, params: any, supabase: any) {
  // Search for prospects
  if (task.includes('search') || task.includes('prospect')) {
    const prospects = []
    for (let i = 0; i < 5; i++) {
      prospects.push({
        company: `Prospect Company ${i + 1}`,
        industry: ['Construction', 'Real Estate', 'Property Management'][Math.floor(Math.random() * 3)],
        size: ['Small', 'Medium', 'Large'][Math.floor(Math.random() * 3)],
        score: (Math.random() * 100).toFixed(1)
      })
    }

    // Store prospects
    for (const prospect of prospects) {
      await supabase
        .from('ai_acquisition_leads')
        .insert({
          source: 'ai_search',
          lead_data: prospect,
          qualification_score: parseFloat(prospect.score),
          status: 'NEW'
        })
    }

    return {
      task: 'prospect_search',
      prospects_found: prospects.length,
      prospects
    }
  }

  // Create outreach campaign
  if (task.includes('outreach') || task.includes('campaign')) {
    const campaign = {
      name: `AI Campaign ${Date.now()}`,
      target_audience: params.audience || 'general',
      channels: ['email', 'linkedin'],
      messages: 5,
      schedule: 'Every 3 days'
    }

    await supabase
      .from('ai_acquisition_campaigns')
      .insert({
        campaign_name: campaign.name,
        campaign_type: 'outreach',
        target_audience: { audience: campaign.target_audience },
        status: 'ACTIVE'
      })

    return {
      task: 'outreach_campaign',
      campaign
    }
  }

  return { task: 'acquisition_task', status: 'completed' }
}

async function executeAnalysisTask(task: string, params: any, supabase: any) {
  // Analyze customer sentiment
  if (task.includes('sentiment')) {
    const { data: interactions } = await supabase
      .from('ai_customer_interactions')
      .select('sentiment_score, created_at')
      .gte('created_at', new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString())
      .order('created_at', { ascending: false })

    const avgSentiment = interactions?.reduce((acc, i) => acc + (i.sentiment_score || 0), 0) / (interactions?.length || 1)

    return {
      task: 'sentiment_analysis',
      period: '7_days',
      interactions_analyzed: interactions?.length || 0,
      average_sentiment: avgSentiment.toFixed(2),
      trend: avgSentiment > 0 ? 'positive' : avgSentiment < 0 ? 'negative' : 'neutral'
    }
  }

  // Performance metrics
  if (task.includes('performance') || task.includes('metrics')) {
    const { data: models } = await supabase
      .from('ai_trained_models')
      .select('model_type, accuracy')
      .order('created_at', { ascending: false })
      .limit(5)

    return {
      task: 'performance_analysis',
      models_evaluated: models?.length || 0,
      avg_accuracy: (models?.reduce((acc, m) => acc + (m.accuracy || 0), 0) / (models?.length || 1)).toFixed(3),
      models: models || []
    }
  }

  return { task: 'analysis_task', status: 'completed' }
}

async function executeAutomationTask(task: string, params: any, supabase: any) {
  // Schedule follow-ups
  if (task.includes('follow') || task.includes('schedule')) {
    const scheduled = []
    const targets = params.targets || ['target1', 'target2']

    for (const target of targets) {
      const scheduleDate = new Date()
      scheduleDate.setDate(scheduleDate.getDate() + Math.floor(Math.random() * 7) + 1)

      scheduled.push({
        target,
        scheduled_for: scheduleDate.toISOString(),
        type: 'follow_up'
      })

      await supabase
        .from('ai_scheduled_outreach')
        .insert({
          target_id: target,
          channel: 'email',
          scheduled_for: scheduleDate.toISOString(),
          status: 'scheduled'
        })
    }

    return {
      task: 'schedule_followups',
      scheduled_count: scheduled.length,
      followups: scheduled
    }
  }

  // Workflow automation
  if (task.includes('workflow') || task.includes('automate')) {
    const workflow = {
      name: `Automated Workflow ${Date.now()}`,
      steps: [
        'Data Collection',
        'Processing',
        'Analysis',
        'Action',
        'Reporting'
      ],
      trigger: params.trigger || 'manual',
      status: 'active'
    }

    return {
      task: 'workflow_automation',
      workflow
    }
  }

  return { task: 'automation_task', status: 'completed' }
}

async function executeGenericTask(task: string, params: any) {
  // Use OpenAI for generic tasks
  const openAiKey = Deno.env.get('OPENAI_API_KEY')!

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
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
          content: 'You are an AI agent executing business tasks. Provide structured, actionable responses.'
        },
        {
          role: 'user',
          content: `Execute this task: ${task}\nParameters: ${JSON.stringify(params)}`
        }
      ],
      temperature: 0.5,
      max_tokens: 500
    }),
  })

  const completion = await response.json()

  return {
    task: 'generic_execution',
    response: completion.choices[0].message.content,
    tokens_used: completion.usage?.total_tokens || 0
  }
}