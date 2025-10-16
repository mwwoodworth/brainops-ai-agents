// AI Execute Edge Function - Executes AI agent tasks
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient, type SupabaseClient } from 'https://esm.sh/@supabase/supabase-js@2.39.1'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
} as const

type PriorityLevel = 'low' | 'normal' | 'high' | 'critical'

type AgentType = 'revenue_generator' | 'customer_acquisition' | 'data_analyzer' | 'automation' | string

interface ExecuteRequest {
  agent_type: AgentType
  task: string
  parameters?: Record<string, unknown>
  priority?: PriorityLevel
}

interface CustomerRow {
  id?: string | number
  company_name?: string | null
  last_contact?: string | null
}

interface InteractionRow {
  sentiment_score?: number | null
  created_at?: string | null
}

interface ModelRow {
  model_type?: string | null
  accuracy?: number | null
  created_at?: string | null
}

interface OpenAIChatCompletionResponse {
  choices?: Array<{
    message?: { content?: string }
  }>
  usage?: {
    total_tokens?: number
  }
}

type JsonRecord = Record<string, unknown>

const getEnvVar = (key: string): string => {
  const value = Deno.env.get(key)
  if (!value) {
    throw new Error(`Missing required environment variable: ${key}`)
  }
  return value
}

const getSupabaseClient = (): SupabaseClient => {
  const supabaseUrl = getEnvVar('SUPABASE_URL')
  const serviceRoleKey = getEnvVar('SUPABASE_SERVICE_ROLE_KEY')
  return createClient(supabaseUrl, serviceRoleKey)
}

const parseExecuteRequest = async (req: Request): Promise<ExecuteRequest> => {
  const body = await req.json().catch(() => null)
  if (!body || typeof body !== 'object') {
    throw new Error('Invalid request body. Expected JSON object.')
  }

  const { agent_type, task, parameters, priority } = body as Record<string, unknown>

  if (typeof agent_type !== 'string' || agent_type.trim().length === 0) {
    throw new Error('The "agent_type" field is required and must be a non-empty string.')
  }

  if (typeof task !== 'string' || task.trim().length === 0) {
    throw new Error('The "task" field is required and must be a non-empty string.')
  }

  if (priority !== undefined && priority !== 'low' && priority !== 'normal' && priority !== 'high' && priority !== 'critical') {
    throw new Error('The "priority" field must be one of low, normal, high, or critical when provided.')
  }

  if (parameters !== undefined && (typeof parameters !== 'object' || parameters === null || Array.isArray(parameters))) {
    throw new Error('The "parameters" field must be an object when provided.')
  }

  return {
    agent_type,
    task,
    parameters: parameters as JsonRecord | undefined,
    priority: (priority as PriorityLevel | undefined) ?? 'normal',
  }
}

const safeString = (value: unknown, fallback = ''): string => {
  if (typeof value === 'string') {
    return value
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }
  return fallback
}

const safeNumber = (value: unknown): number => {
  const num = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(num) ? num : 0
}

const createExecutionRecord = async (
  supabase: SupabaseClient,
  agentType: AgentType,
  task: string,
  priority: PriorityLevel,
  parameters: JsonRecord,
): Promise<string> => {
  const executionId = crypto.randomUUID()
  const insertResponse = await supabase
    .from('agent_executions')
    .insert({
      id: executionId,
      agent_type: agentType,
      prompt: task,
      status: 'executing',
      priority,
      metadata: parameters,
      created_at: new Date().toISOString(),
    })

  if (insertResponse.error) {
    throw new Error(`Failed to create execution record: ${insertResponse.error.message}`)
  }

  return executionId
}

const updateExecutionRecord = async (
  supabase: SupabaseClient,
  executionId: string,
  payload: Record<string, unknown>,
) => {
  const response = await supabase
    .from('agent_executions')
    .update({
      ...payload,
      updated_at: new Date().toISOString(),
    })
    .eq('id', executionId)

  if (response.error) {
    throw new Error(`Failed to update execution record: ${response.error.message}`)
  }
}

const broadcastExecutionEvent = async (
  supabase: SupabaseClient,
  executionId: string,
  agentType: AgentType,
  task: string,
  status: string,
) => {
  const response = await supabase
    .from('ai_event_broadcasts')
    .insert({
      event_type: 'agent_execution',
      event_data: {
        execution_id: executionId,
        agent_type: agentType,
        task,
        status,
      },
      created_at: new Date().toISOString(),
    })

  if (response.error) {
    throw new Error(`Failed to broadcast execution event: ${response.error.message}`)
  }
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const { agent_type, task, parameters = {}, priority = 'normal' } = await parseExecuteRequest(req)
    const supabase = getSupabaseClient()

    const executionId = await createExecutionRecord(supabase, agent_type, task, priority, parameters)

    try {
      const result = await executeAgentTask(agent_type, task, parameters, supabase)

      await updateExecutionRecord(supabase, executionId, {
        status: 'completed',
        result,
      })

      await broadcastExecutionEvent(supabase, executionId, agent_type, task, 'completed')

      return new Response(
        JSON.stringify({
          success: true,
          execution_id: executionId,
          result,
          status: 'completed',
        }),
        {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          status: 200,
        },
      )
    } catch (taskError) {
      const errorMessage = taskError instanceof Error ? taskError.message : 'Agent task failed.'
      await updateExecutionRecord(supabase, executionId, {
        status: 'failed',
        error_message: errorMessage,
      })

      await broadcastExecutionEvent(supabase, executionId, agent_type, task, 'failed')

      throw taskError
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred.'
    return new Response(
      JSON.stringify({
        success: false,
        error: errorMessage,
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 400,
      },
    )
  }
})

const executeAgentTask = async (
  agentType: AgentType,
  task: string,
  parameters: JsonRecord,
  supabase: SupabaseClient,
): Promise<JsonRecord> => {
  switch (agentType) {
    case 'revenue_generator':
      return executeRevenueTask(task, parameters, supabase)
    case 'customer_acquisition':
      return executeAcquisitionTask(task, parameters, supabase)
    case 'data_analyzer':
      return executeAnalysisTask(task, parameters, supabase)
    case 'automation':
      return executeAutomationTask(task, parameters, supabase)
    default:
      return executeGenericTask(task, parameters)
  }
}

const executeRevenueTask = async (
  task: string,
  params: JsonRecord,
  supabase: SupabaseClient,
): Promise<JsonRecord> => {
  if (task.includes('identify') || task.includes('lead')) {
    const leadsResponse = await supabase
      .from<CustomerRow>('customers')
      .select('id, company_name, last_contact')
      .is('last_contact', null)
      .limit(10)

    if (leadsResponse.error) {
      throw new Error(`Failed to fetch customers: ${leadsResponse.error.message}`)
    }

    const leads = leadsResponse.data ?? []
    const qualified: Array<{ lead_id: string; company: string; score: string }> = []

    for (const lead of leads) {
      const leadId = safeString(lead?.id, '')
      if (!leadId) {
        continue
      }

      const company = safeString(lead?.company_name, 'Unknown Company')
      const score = Math.random() * 100

      if (score > 50) {
        qualified.push({
          lead_id: leadId,
          company,
          score: score.toFixed(1),
        })

        const insertLeadResponse = await supabase
          .from('ai_leads')
          .insert({
            company_name: company,
            lead_score: score,
            lead_stage: 'QUALIFIED',
            source: 'ai_identification',
          })

        if (insertLeadResponse.error) {
          throw new Error(`Failed to store qualified lead: ${insertLeadResponse.error.message}`)
        }
      }
    }

    return {
      task: 'lead_identification',
      leads_analyzed: leads.length,
      qualified_leads: qualified.length,
      leads: qualified,
    }
  }

  if (task.includes('proposal')) {
    const proposal = {
      title: 'Custom Solution Proposal',
      value: Math.floor(Math.random() * 50_000) + 10_000,
      items: [
        'Discovery and Analysis',
        'Solution Design',
        'Implementation',
        'Training and Support',
      ],
      timeline: '4-6 weeks',
      generated_at: new Date().toISOString(),
      notes: params.notes ?? null,
    }

    return {
      task: 'proposal_generation',
      proposal,
    }
  }

  return { task: 'revenue_task', status: 'completed' }
}

const executeAcquisitionTask = async (
  task: string,
  params: JsonRecord,
  supabase: SupabaseClient,
): Promise<JsonRecord> => {
  if (task.includes('search') || task.includes('prospect')) {
    const industries = ['Construction', 'Real Estate', 'Property Management']
    const sizes = ['Small', 'Medium', 'Large']
    const prospects = Array.from({ length: 5 }, (_, index) => {
      const score = Number((Math.random() * 100).toFixed(1))
      return {
        company: `Prospect Company ${index + 1}`,
        industry: industries[Math.floor(Math.random() * industries.length)],
        size: sizes[Math.floor(Math.random() * sizes.length)],
        score,
      }
    })

    for (const prospect of prospects) {
      const insertProspectResponse = await supabase
        .from('ai_acquisition_leads')
        .insert({
          source: 'ai_search',
          lead_data: prospect,
          qualification_score: prospect.score,
          status: 'NEW',
        })

      if (insertProspectResponse.error) {
        throw new Error(`Failed to store acquisition lead: ${insertProspectResponse.error.message}`)
      }
    }

    return {
      task: 'prospect_search',
      prospects_found: prospects.length,
      prospects,
    }
  }

  if (task.includes('outreach') || task.includes('campaign')) {
    const campaign = {
      name: `AI Campaign ${Date.now()}`,
      target_audience: safeString(params.audience, 'general'),
      channels: ['email', 'linkedin'],
      messages: 5,
      schedule: 'Every 3 days',
    }

    const insertCampaignResponse = await supabase
      .from('ai_acquisition_campaigns')
      .insert({
        campaign_name: campaign.name,
        campaign_type: 'outreach',
        target_audience: { audience: campaign.target_audience },
        status: 'ACTIVE',
      })

    if (insertCampaignResponse.error) {
      throw new Error(`Failed to store outreach campaign: ${insertCampaignResponse.error.message}`)
    }

    return {
      task: 'outreach_campaign',
      campaign,
    }
  }

  return { task: 'acquisition_task', status: 'completed' }
}

const executeAnalysisTask = async (
  task: string,
  _params: JsonRecord,
  supabase: SupabaseClient,
): Promise<JsonRecord> => {
  if (task.includes('sentiment')) {
    const interactionsResponse = await supabase
      .from<InteractionRow>('ai_customer_interactions')
      .select('sentiment_score, created_at')
      .gte('created_at', new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString())
      .order('created_at', { ascending: false })

    if (interactionsResponse.error) {
      throw new Error(`Failed to fetch customer interactions: ${interactionsResponse.error.message}`)
    }

    const interactions: InteractionRow[] = interactionsResponse.data ?? []
    const sentiments = interactions.map((record: InteractionRow) => safeNumber(record.sentiment_score))
    const totalSentiment = sentiments.reduce((acc: number, value: number) => acc + value, 0)
    const avgSentiment = interactions.length > 0 ? totalSentiment / interactions.length : 0

    return {
      task: 'sentiment_analysis',
      period: '7_days',
      interactions_analyzed: interactions.length,
      average_sentiment: avgSentiment.toFixed(2),
      trend: avgSentiment > 0 ? 'positive' : avgSentiment < 0 ? 'negative' : 'neutral',
    }
  }

  if (task.includes('performance') || task.includes('metrics')) {
    const modelsResponse = await supabase
      .from<ModelRow>('ai_trained_models')
      .select('model_type, accuracy, created_at')
      .order('created_at', { ascending: false })
      .limit(5)

    if (modelsResponse.error) {
      throw new Error(`Failed to fetch trained models: ${modelsResponse.error.message}`)
    }

    const models: ModelRow[] = modelsResponse.data ?? []
    const accuracyValues = models.map((model: ModelRow) => safeNumber(model.accuracy))
    const avgAccuracy = models.length > 0
      ? accuracyValues.reduce((acc: number, value: number) => acc + value, 0) / models.length
      : 0

    return {
      task: 'performance_analysis',
      models_evaluated: models.length,
      avg_accuracy: avgAccuracy.toFixed(3),
      models,
    }
  }

  return { task: 'analysis_task', status: 'completed' }
}

const executeAutomationTask = async (
  task: string,
  params: JsonRecord,
  supabase: SupabaseClient,
): Promise<JsonRecord> => {
  if (task.includes('follow') || task.includes('schedule')) {
    const targets = Array.isArray(params.targets) ? params.targets : ['target1', 'target2']
    const scheduled: Array<{ target: string; scheduled_for: string; type: string }> = []

    for (const target of targets) {
      const targetId = safeString(target, '')
      if (!targetId) {
        continue
      }

      const scheduleDate = new Date()
      scheduleDate.setDate(scheduleDate.getDate() + Math.floor(Math.random() * 7) + 1)

      const record = {
        target: targetId,
        scheduled_for: scheduleDate.toISOString(),
        type: 'follow_up',
      }

      const insertResponse = await supabase
        .from('ai_scheduled_outreach')
        .insert({
          target_id: targetId,
          channel: 'email',
          scheduled_for: record.scheduled_for,
          status: 'scheduled',
        })

      if (insertResponse.error) {
        throw new Error(`Failed to schedule outreach: ${insertResponse.error.message}`)
      }

      scheduled.push(record)
    }

    return {
      task: 'schedule_followups',
      scheduled_count: scheduled.length,
      followups: scheduled,
    }
  }

  if (task.includes('workflow') || task.includes('automate')) {
    const workflow = {
      name: `Automated Workflow ${Date.now()}`,
      steps: [
        'Data Collection',
        'Processing',
        'Analysis',
        'Action',
        'Reporting',
      ],
      trigger: safeString(params.trigger, 'manual'),
      status: 'active',
    }

    return {
      task: 'workflow_automation',
      workflow,
    }
  }

  return { task: 'automation_task', status: 'completed' }
}

const executeGenericTask = async (task: string, params: JsonRecord): Promise<JsonRecord> => {
  const openAiKey = getEnvVar('OPENAI_API_KEY')

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

  if (!response.ok) {
    const errorText = await response.text()
    throw new Error(`OpenAI API request failed: ${errorText}`)
  }

  const completion = await response.json() as OpenAIChatCompletionResponse
  const message = completion.choices?.[0]?.message?.content?.trim()

  if (!message) {
    throw new Error('OpenAI returned an empty response for the generic task.')
  }

  return {
    task: 'generic_execution',
    response: message,
    tokens_used: completion.usage?.total_tokens ?? null,
  }
}
