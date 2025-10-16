interface DenoEnv {
  get(key: string): string | undefined
}

interface DenoNamespace {
  env: DenoEnv
}

declare const Deno: DenoNamespace
