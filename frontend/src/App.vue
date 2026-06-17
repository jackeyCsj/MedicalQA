<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
// @ts-expect-error markdown-it uses CommonJS typings in this Vite setup.
import MarkdownIt from 'markdown-it'
import { login, logout, me, models, register, streamChat, type StreamEvent, type User } from './api'

type Message = {
  role: 'user' | 'assistant'
  content: string
  debug?: Record<string, unknown>[]
}

const user = ref<User | null>(null)
const authMode = ref<'login' | 'register'>('login')
const username = ref('')
const password = ref('')
const authError = ref('')
const authBusy = ref(false)

const availableModels = ref<string[]>([])
const selectedModel = ref<string | null>(null)
const finalEntityTopK = ref(5)
const modelMenuOpen = ref(false)
const modelPickerRef = ref<HTMLElement | null>(null)
const theme = ref<'light' | 'dark'>('light')
const input = ref('')
const statusText = ref('')
const sending = ref(false)
const messages = ref<Message[]>([])
const messagesRef = ref<HTMLElement | null>(null)
const minFinalEntityTopK = 3
const maxFinalEntityTopK = 10
const finalEntityOptions = Array.from(
  { length: maxFinalEntityTopK - minFinalEntityTopK + 1 },
  (_, index) => minFinalEntityTopK + index
)
const markdown = new MarkdownIt({
  breaks: true,
  linkify: true,
  html: false
})

const isAdmin = computed(() => user.value?.is_admin === true)
const appThemeClass = computed(() => `theme-${theme.value}`)
const hasInput = computed(() => input.value.trim().length > 0)
const canSend = computed(() => hasInput.value && !sending.value)
const selectedModelLabel = computed(() => selectedModel.value || '选择模型')
const themeButtonLabel = computed(() => (theme.value === 'light' ? '夜间模式' : '亮色模式'))
const historyItems = [
  { id: 'current', title: '当前问答' },
  { id: 'placeholder-1', title: '历史会话占位' }
]

function scrollToBottom() {
  nextTick(() => {
    if (messagesRef.value) {
      messagesRef.value.scrollTop = messagesRef.value.scrollHeight
    }
  })
}

function handleDocumentPointerDown(event: PointerEvent) {
  if (!modelMenuOpen.value) return
  if (modelPickerRef.value?.contains(event.target as Node)) return
  modelMenuOpen.value = false
}

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
    await streamChat(text, selectedModel.value, finalEntityTopK.value, (event: StreamEvent) => {
      if (event.event === 'status') {
        statusText.value = event.data
      } else if (event.event === 'debug') {
        assistant.debug?.push(event.data)
      } else if (event.event === 'token') {
        assistant.content += event.data
        scrollToBottom()
      } else if (event.event === 'error') {
        assistant.content += `\n${event.data.message}`
      } else if (event.event === 'done') {
        statusText.value = ''
      }
    })
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

function startNewChat() {
  messages.value = []
  statusText.value = ''
  input.value = ''
  modelMenuOpen.value = false
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
          <article v-for="item in historyItems" :key="item.id" class="history-item">
            <span>{{ item.title }}</span>
            <div class="history-actions">
              <button type="button" disabled title="后续接入改名">改名</button>
              <button type="button" disabled title="后续接入删除">删除</button>
            </div>
          </article>
          <p class="history-note">后续接入真实会话记录。</p>
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
