declare module "https://deno.land/std@0.168.0/http/server.ts" {
  export function serve(
    handler: (request: Request) => Response | Promise<Response>,
    options?: { addr?: string }
  ): void
}

declare module 'https://esm.sh/@supabase/supabase-js@2.39.1' {
  export interface SupabaseClient {
    from<T>(table: string): any
  }

  export function createClient(supabaseUrl: string, supabaseKey: string): SupabaseClient
}
