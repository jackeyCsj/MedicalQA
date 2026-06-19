<script setup lang="ts">
/**
 * App.vue — 主组件
 *
 * 负责整个应用的 UI 和交互逻辑，包括：
 * 1. 认证面板（登录/注册）
 * 2. 聊天布局（侧边栏 + 聊天区）
 * 3. 侧边栏：新会话、历史会话列表、会话切换/删除
 * 4. 聊天区：消息列表、模型选择、检索范围调节、输入框
 * 5. SSE 流式消息接收和渲染（支持 Markdown）
 */
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
// @ts-expect-error markdown-it uses CommonJS typings in this Vite setup.
import MarkdownIt from 'markdown-it'
import { login, logout, me, models, register, streamChat, listConversations, getConversationMessages, deleteConversation, type StreamEvent, type User, type ConversationSummary } from './api'

/** 单条消息的数据结构 */
type Message = {
  role: 'user' | 'assistant'
  content: string
  debug?: Record<string, unknown>[]  // 调试信息（仅管理员可见）
}

// ============================================================
//  认证状态
// ============================================================
const user = ref<User | null>(null)           // 当前登录用户（null 表示未登录）
const authMode = ref<'login' | 'register'>('login')  // 认证面板模式
const username = ref('')
const password = ref('')
const authError = ref('')                     // 认证错误提示
const authBusy = ref(false)                   // 认证请求进行中

// ============================================================
//  模型和检索配置
// ============================================================
const availableModels = ref<string[]>([])     // 可用 LLM 模型列表
const selectedModel = ref<string | null>(null) // 当前选中的模型
const finalEntityTopK = ref(5)                // 送入 prompt 的实体数量（3-10）
const modelMenuOpen = ref(false)              // 模型下拉菜单是否展开
const modelPickerRef = ref<HTMLElement | null>(null)  // 模型选择器 DOM 引用（用于点击外部关闭）
const theme = ref<'light' | 'dark'>('light')  // 主题模式

// ============================================================
//  聊天状态
// ============================================================
const input = ref('')                         // 输入框文本
const statusText = ref('')                    // SSE status 事件的进度提示
const sending = ref(false)                    // 是否正在发送/等待响应
const messages = ref<Message[]>([])           // 当前会话的消息列表
const messagesRef = ref<HTMLElement | null>(null)  // 消息容器 DOM 引用（用于自动滚动）

// 检索范围选项（3-10，步进 1）
const minFinalEntityTopK = 3
const maxFinalEntityTopK = 10
const finalEntityOptions = Array.from(
  { length: maxFinalEntityTopK - minFinalEntityTopK + 1 },
  (_, index) => minFinalEntityTopK + index
)

// Markdown 渲染器（将 LLM 输出的 Markdown 转为 HTML）
const markdown = new MarkdownIt({
  breaks: true,    // 换行符转 <br>
  linkify: true,   // 自动识别 URL
  html: false       // 禁止原始 HTML（安全考虑）
})

// ============================================================
//  计算属性
// ============================================================
const isAdmin = computed(() => user.value?.is_admin === true)
const appThemeClass = computed(() => `theme-${theme.value}`)
const hasInput = computed(() => input.value.trim().length > 0)
const canSend = computed(() => hasInput.value && !sending.value)
const selectedModelLabel = computed(() => selectedModel.value || '选择模型')
const themeButtonLabel = computed(() => (theme.value === 'light' ? '夜间模式' : '亮色模式'))

// ============================================================
//  会话管理状态
// ============================================================
const conversations = ref<ConversationSummary[]>([])  // 历史会话列表
const activeConversationId = ref<number | null>(null) // 当前活跃的会话 ID（null 表示新会话）
const loadingConversation = ref(false)                // 正在加载历史会话

/** 滚动消息容器到底部（新消息到达时自动滚动） */
function scrollToBottom() {
  nextTick(() => {
    if (messagesRef.value) {
      messagesRef.value.scrollTop = messagesRef.value.scrollHeight
    }
  })
}

/** 点击文档任意位置时关闭模型下拉菜单 */
function handleDocumentPointerDown(event: PointerEvent) {
  if (!modelMenuOpen.value) return
  // 如果点击的是模型选择器内部，不关闭
  if (modelPickerRef.value?.contains(event.target as Node)) return
  modelMenuOpen.value = false
}

/** 将助手消息的 Markdown 内容渲染为 HTML */
function renderAssistantContent(content: string) {
  return markdown.render(content || '')
}

watch(statusText, scrollToBottom)

onMounted(async () => {
  document.addEventListener('pointerdown', handleDocumentPointerDown)
  try {
    user.value = await me()
  } catch {
    user.value = null
  }
  await loadModels()
  if (user.value) {
    await loadConversations()
  }
})

onBeforeUnmount(() => {
  document.removeEventListener('pointerdown', handleDocumentPointerDown)
})

async function loadModels() {
  try {
    const result = await models()
    availableModels.value = result.models ?? []
    if (!selectedModel.value && availableModels.value.length) {
      selectedModel.value = availableModels.value[0]
    }
  } catch (e) {
    console.error('Failed to load models:', e)
    availableModels.value = []
  }
}

async function submitAuth() {
  authBusy.value = true
  authError.value = ''
  try {
    if (authMode.value === 'register') {
      await register(username.value, password.value)
      authMode.value = 'login'
      authError.value = '注册成功，请登录。'
    } else {
      const result = await login(username.value, password.value)
      user.value = result.user
      username.value = ''
      password.value = ''
    }
  } catch (error) {
    authError.value = error instanceof Error ? error.message : '请求失败'
  } finally {
    authBusy.value = false
  }
}

async function submitLogout() {
  await logout().catch(() => undefined)
  user.value = null
  messages.value = []
  username.value = ''
  password.value = ''
  authError.value = ''
  modelMenuOpen.value = false
}

/**
 * 发送消息并处理 SSE 流式响应。
 *
 * 流程：
 * 1. 清空输入框，添加用户消息和空的助手消息占位
 * 2. 调用 streamChat() 发起 SSE 请求
 * 3. 逐个处理事件：status→更新进度、debug→记录调试、token→追加内容、done→保存会话ID
 * 4. 完成后刷新历史会话列表
 */
async function sendMessage() {
  const text = input.value.trim()
  if (!text || sending.value) return
  input.value = ''
  statusText.value = ''
  sending.value = true
  messages.value.push({ role: 'user', content: text })
  const assistant: Message = { role: 'assistant', content: '', debug: [] }
  messages.value.push(assistant)
  scrollToBottom()

  try {
    await streamChat(
      text,
      selectedModel.value,
      finalEntityTopK.value,
      (event: StreamEvent) => {
        if (event.event === 'status') {
          // 进度提示（如"正在提取关键词..."）
          statusText.value = event.data
        } else if (event.event === 'debug') {
          // 调试信息（仅管理员可见）
          assistant.debug?.push(event.data)
        } else if (event.event === 'token') {
          // LLM 输出的文本片段，逐个追加
          assistant.content += event.data
          scrollToBottom()
        } else if (event.event === 'error') {
          // 错误信息
          assistant.content += `\n${event.data.message}`
        } else if (event.event === 'done') {
          // 完成事件：保存 conversation_id，刷新会话列表
          statusText.value = ''
          const doneData = event.data as Record<string, unknown>
          if (doneData.conversation_id != null) {
            activeConversationId.value = doneData.conversation_id as number
            void loadConversations()
          }
        }
      },
      activeConversationId.value  // 传入当前会话 ID（null 表示新会话）
    )
  } catch (error) {
    assistant.content = error instanceof Error ? error.message : '发送失败'
  } finally {
    sending.value = false
    statusText.value = ''
  }
}

function toggleTheme() {
  theme.value = theme.value === 'light' ? 'dark' : 'light'
}

function selectModel(model: string) {
  selectedModel.value = model
  modelMenuOpen.value = false
}

/** 开始新会话：清空消息列表，重置会话 ID */
function startNewChat() {
  messages.value = []
  statusText.value = ''
  input.value = ''
  modelMenuOpen.value = false
  activeConversationId.value = null
}

/** 从后端加载用户的会话列表（用于侧边栏展示） */
async function loadConversations() {
  try {
    conversations.value = await listConversations()
  } catch (e) {
    console.error('Failed to load conversations:', e)
  }
}

/**
 * 打开历史会话：加载该会话的全部消息并展示。
 * 点击侧边栏的历史会话项时触发。
 */
async function openConversation(convId: number) {
  if (sending.value || loadingConversation.value) return
  loadingConversation.value = true
  try {
    const msgs = await getConversationMessages(convId)
    // 将后端消息格式转为前端 Message 格式
    messages.value = msgs.map(m => ({ role: m.role, content: m.content }))
    activeConversationId.value = convId
    scrollToBottom()
  } catch (e) {
    console.error('Failed to open conversation:', e)
  } finally {
    loadingConversation.value = false
  }
}

/**
 * 删除会话：调用后端 API 删除，如果删除的是当前活跃会话则清空消息。
 */
async function deleteConversationItem(convId: number) {
  try {
    await deleteConversation(convId)
    if (activeConversationId.value === convId) {
      messages.value = []
      activeConversationId.value = null
    }
    await loadConversations()
  } catch (e) {
    console.error('Failed to delete conversation:', e)
  }
}

function adjustFinalEntityTopK(delta: number) {
  const nextValue = finalEntityTopK.value + delta
  finalEntityTopK.value = Math.min(maxFinalEntityTopK, Math.max(minFinalEntityTopK, nextValue))
}

function handleInputKeydown(event: KeyboardEvent) {
  if (event.key !== 'Enter' || event.shiftKey || event.isComposing) return
  event.preventDefault()
  void sendMessage()
}
</script>

<template>
  <main :class="['app-shell', appThemeClass]">
    <section v-if="!user" class="auth-panel">
      <div class="auth-card">
        <h1>医疗问答系统</h1>
        <p>登录后开始基于知识图谱的医疗问答。</p>
        <div class="segmented">
          <button :class="{ active: authMode === 'login' }" @click="authMode = 'login'">登录</button>
          <button :class="{ active: authMode === 'register' }" @click="authMode = 'register'">注册</button>
        </div>
        <form @submit.prevent="submitAuth">
          <!-- 隐藏输入框欺骗浏览器自动填充 -->
          <input type="text" style="display:none" />
          <input type="password" style="display:none" />
          <label>
            用户名
            <input v-model="username" autocomplete="off" />
          </label>
          <label>
            密码
            <input v-model="password" type="password" autocomplete="new-password" />
          </label>
          <button class="primary" :disabled="authBusy">
            {{ authMode === 'login' ? '登录' : '注册' }}
          </button>
        </form>
        <p v-if="authError" class="auth-error">{{ authError }}</p>
      </div>
    </section>

    <section v-else class="chat-layout">
      <aside class="sidebar">
        <div class="brand-block">
          <h2>医疗问答系统</h2>
          <button class="new-chat-button" type="button" @click="startNewChat">新会话</button>
        </div>
        <nav class="history-panel" aria-label="历史会话">
          <div class="history-heading">历史会话</div>
          <article
            v-for="item in conversations"
            :key="item.id"
            :class="['history-item', { active: item.id === activeConversationId }]"
            @click="openConversation(item.id)"
          >
            <span class="history-title">{{ item.title }}</span>
            <span v-if="item.phase === 'clarifying'" class="phase-badge">追问中</span>
            <div class="history-actions">
              <button type="button" class="delete-btn" title="删除" @click.stop="deleteConversationItem(item.id)">删除</button>
            </div>
          </article>
          <p v-if="!conversations.length" class="history-note">暂无历史会话。</p>
        </nav>
      </aside>

      <section class="chat-panel">
        <header class="chat-header">
          <button
            class="icon-button"
            type="button"
            :aria-label="themeButtonLabel"
            :title="themeButtonLabel"
            @click="toggleTheme"
          >
            <svg
              v-if="theme === 'light'"
              aria-hidden="true"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <path d="M12 3a6 6 0 0 0 9 7.5A9 9 0 1 1 12 3Z" />
            </svg>
            <svg
              v-else
              aria-hidden="true"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <circle cx="12" cy="12" r="4" />
              <path d="M12 2v2" />
              <path d="M12 20v2" />
              <path d="m4.93 4.93 1.41 1.41" />
              <path d="m17.66 17.66 1.41 1.41" />
              <path d="M2 12h2" />
              <path d="M20 12h2" />
              <path d="m6.34 17.66-1.41 1.41" />
              <path d="m19.07 4.93-1.41 1.41" />
            </svg>
          </button>
          <div class="account-area">
            <span class="username">{{ user.username }}</span>
            <button class="ghost-button" type="button" @click="submitLogout">退出登录</button>
          </div>
        </header>

        <div class="messages" ref="messagesRef">
          <article v-for="(message, index) in messages" :key="index" :class="['message', message.role]">
            <div
              v-if="message.role === 'assistant'"
              class="bubble markdown-body"
              v-html="renderAssistantContent(message.content || '...')"
            />
            <div v-else class="bubble">{{ message.content || '...' }}</div>
            <details v-if="isAdmin && message.debug?.length" class="debug">
              <summary>调试信息</summary>
              <pre>{{ JSON.stringify(message.debug, null, 2) }}</pre>
            </details>
          </article>
          <div v-if="statusText" class="status">{{ statusText }}</div>
        </div>
        <form class="composer" @submit.prevent="sendMessage">
          <div class="composer-controls">
            <div class="model-picker" ref="modelPickerRef">
              <button class="control-button" type="button" @click="modelMenuOpen = !modelMenuOpen">
                <span>模型</span>
                <strong>{{ selectedModelLabel }}</strong>
              </button>
              <div v-if="modelMenuOpen" class="model-menu">
                <button
                  v-for="model in availableModels"
                  :key="model"
                  type="button"
                  :class="{ selected: model === selectedModel }"
                  @click="selectModel(model)"
                >
                  {{ model }}
                </button>
                <span v-if="!availableModels.length" class="model-empty">加载中...</span>
              </div>
            </div>
            <div class="range-stepper" role="group" aria-label="检索范围">
              <span class="range-heading">
                <span>检索范围</span>
              </span>
              <div class="range-stepper-controls">
                <button
                  class="stepper-button"
                  type="button"
                  :disabled="finalEntityTopK <= minFinalEntityTopK"
                  aria-label="减少检索范围"
                  @click="adjustFinalEntityTopK(-1)"
                >
                  <span aria-hidden="true">-</span>
                </button>
                <label class="range-picker">
                  <select
                    v-model.number="finalEntityTopK"
                    class="range-select"
                    aria-label="选择检索范围"
                  >
                    <option
                      v-for="value in finalEntityOptions"
                      :key="value"
                      :value="value"
                    >
                      {{ value }}
                    </option>
                  </select>
                </label>
                <button
                  class="stepper-button"
                  type="button"
                  :disabled="finalEntityTopK >= maxFinalEntityTopK"
                  aria-label="增加检索范围"
                  @click="adjustFinalEntityTopK(1)"
                >
                  <span aria-hidden="true">+</span>
                </button>
              </div>
            </div>
          </div>
          <div class="input-row">
            <textarea
              v-model="input"
              rows="2"
              placeholder="输入你的问题"
              @keydown="handleInputKeydown"
            />
            <button
              :class="['icon-button', 'send-button', { ready: canSend }]"
              :disabled="!canSend"
              type="submit"
              aria-label="发送"
              title="发送"
            >
              <svg
                aria-hidden="true"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="1.9"
                stroke-linecap="round"
                stroke-linejoin="round"
              >
                <path d="M21 3 10 14" />
                <path d="m21 3-7 18-4-7-7-4Z" />
              </svg>
            </button>
          </div>
        </form>
      </section>
    </section>
  </main>
</template>
