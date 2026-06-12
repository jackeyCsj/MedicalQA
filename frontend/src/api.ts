export type User = {
  username: string
  is_admin: boolean
}

export type StreamEvent =
  | { event: 'status'; data: string }
  | { event: 'debug'; data: Record<string, unknown> }
  | { event: 'token'; data: string }
  | { event: 'done'; data: Record<string, unknown> }
  | { event: 'error'; data: { message: string } }

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...(init.headers ?? {})
    }
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: '请求失败' }))
    throw new Error(payload.detail ?? '请求失败')
  }
  return response.json() as Promise<T>
}

export function me() {
  return apiFetch<User>('/api/auth/me')
}

export function login(username: string, password: string) {
  return apiFetch<{ user: User }>('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify({ username, password })
  })
}

export function register(username: string, password: string) {
  return apiFetch<{ ok: boolean }>('/api/auth/register', {
    method: 'POST',
    body: JSON.stringify({ username, password })
  })
}

export function logout() {
  return apiFetch<{ ok: boolean }>('/api/auth/logout', { method: 'POST' })
}

export function models() {
  return apiFetch<{ provider: string; models: string[] }>('/api/models')
}

function parseSseChunk(buffer: string): { events: StreamEvent[]; rest: string } {
  const parts = buffer.split('\n\n')
  const rest = parts.pop() ?? ''
  const events = parts
    .map((part) => {
      const eventLine = part.split('\n').find((line) => line.startsWith('event: '))
      const dataLine = part.split('\n').find((line) => line.startsWith('data: '))
      if (!eventLine || !dataLine) return null
      const event = eventLine.slice('event: '.length) as StreamEvent['event']
      const data = JSON.parse(dataLine.slice('data: '.length))
      return { event, data } as StreamEvent
    })
    .filter((event): event is StreamEvent => event !== null)
  return { events, rest }
}

export async function streamChat(
  message: string,
  model: string | null,
  onEvent: (event: StreamEvent) => void
) {
  const response = await fetch(`${API_BASE}/api/chat/stream`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, model })
  })
  if (!response.ok || !response.body) {
    const payload = await response.json().catch(() => ({ detail: '请求失败' }))
    throw new Error(payload.detail ?? '请求失败')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const parsed = parseSseChunk(buffer)
    buffer = parsed.rest
    parsed.events.forEach(onEvent)
  }
}
