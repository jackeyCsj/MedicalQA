/**
 * 前端 API 客户端。
 *
 * 封装所有与后端的 HTTP 通信，包括：
 * - 认证相关（me、login、register、logout）
 * - 模型列表查询
 * - SSE 流式聊天（核心功能）
 * - 会话 CRUD（列表、消息历史、删除）
 */

/** 用户信息 */
export type User = {
  username: string
  is_admin: boolean
}

/** 会话摘要（用于侧边栏列表展示） */
export type ConversationSummary = {
  id: number
  title: string
  phase: string        // "collecting" | "clarifying" | "answered"
  updated_at: string | null
}

/** 会话消息 */
export type ConversationMessage = {
  id: number
  role: 'user' | 'assistant'
  content: string
  created_at: string | null
}

/**
 * SSE 事件联合类型。
 *
 * 与后端 _sse() 函数的事件类型一一对应：
 * - status: 进度提示文本
 * - debug: 调试信息（关键词、检索结果、prompt 等）
 * - token: LLM 输出的文本片段
 * - action: 路由决策结果（ask=追问 / answer=最终回答）
 * - done: 完成事件，含 conversation_id 和 diagnostic_state
 * - error: 错误信息
 */
export type StreamEvent =
  | { event: 'status'; data: string }
  | { event: 'debug'; data: Record<string, unknown> }
  | { event: 'token'; data: string }
  | { event: 'action'; data: { type: 'ask' | 'answer' } }
  | { event: 'done'; data: Record<string, unknown> }
  | { event: 'error'; data: { message: string } }

// 后端 API 基础地址（可通过 VITE_API_BASE 环境变量覆盖）
const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

/**
 * 通用 API 请求函数。
 * 自动携带 credentials: 'include'（发送 session cookie）和 Content-Type。
 */
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

// ============================================================
//  认证相关 API
// ============================================================

/** 获取当前登录用户信息（未登录时抛异常） */
export function me() {
  return apiFetch<User>('/api/auth/me')
}

/** 登录 */
export function login(username: string, password: string) {
  return apiFetch<{ user: User }>('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify({ username, password })
  })
}

/** 注册 */
export function register(username: string, password: string) {
  return apiFetch<{ ok: boolean }>('/api/auth/register', {
    method: 'POST',
    body: JSON.stringify({ username, password })
  })
}

/** 登出 */
export function logout() {
  return apiFetch<{ ok: boolean }>('/api/auth/logout', { method: 'POST' })
}

// ============================================================
//  模型相关 API
// ============================================================

/** 获取当前 LLM 提供商和可用模型列表 */
export function models() {
  return apiFetch<{ provider: string; models: string[] }>('/api/models')
}

// ============================================================
//  会话相关 API
// ============================================================

/** 获取用户的会话列表（按更新时间倒序） */
export function listConversations() {
  return apiFetch<ConversationSummary[]>('/api/conversations')
}

/** 获取会话的全部消息（按创建时间正序） */
export function getConversationMessages(convId: number) {
  return apiFetch<ConversationMessage[]>(`/api/conversations/${convId}/messages`)
}

/** 删除会话及其所有关联数据 */
export function deleteConversation(convId: number) {
  return apiFetch<{ ok: boolean }>(`/api/conversations/${convId}`, { method: 'DELETE' })
}

// ============================================================
//  SSE 流式聊天
// ============================================================

/**
 * 解析 SSE 数据流中的事件。
 *
 * SSE 格式：event: xxx\ndata: json\n\n
 * 由于 TCP 分包，一个 chunk 可能包含多个完整事件 + 一个不完整事件的前半部分。
 * 所以用 \n\n 分割后，最后一段作为 rest 返回，下次拼接。
 */
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

/**
 * SSE 流式聊天。
 *
 * 向后端发送用户消息，通过 ReadableStream 逐块读取 SSE 事件，
 * 解析后通过 onEvent 回调逐个通知调用方。
 *
 * @param message - 用户消息文本
 * @param model - LLM 模型名称（null 时使用默认）
 * @param finalEntityTopK - 送入 prompt 的实体数量
 * @param onEvent - 事件回调函数
 * @param conversationId - 会话 ID（null 表示新会话）
 */
export async function streamChat(
  message: string,
  model: string | null,
  finalEntityTopK: number,
  onEvent: (event: StreamEvent) => void,
  conversationId?: number | null
) {
  const body: Record<string, unknown> = { message, model, final_entity_top_k: finalEntityTopK }
  if (conversationId != null) {
    body.conversation_id = conversationId
  }

  const response = await fetch(`${API_BASE}/api/chat/stream`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  })
  if (!response.ok || !response.body) {
    const payload = await response.json().catch(() => ({ detail: '请求失败' }))
    throw new Error(payload.detail ?? '请求失败')
  }

  // 逐块读取 SSE 流
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const parsed = parseSseChunk(buffer)
    buffer = parsed.rest  // 保留不完整的事件片段，下次拼接
    parsed.events.forEach(onEvent)
  }
}
