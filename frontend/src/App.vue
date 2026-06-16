<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
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
const input = ref('')
const statusText = ref('')
const sending = ref(false)
const messages = ref<Message[]>([])
const messagesRef = ref<HTMLElement | null>(null)

const isAdmin = computed(() => user.value?.is_admin === true)

function scrollToBottom() {
  nextTick(() => {
    if (messagesRef.value) {
      messagesRef.value.scrollTop = messagesRef.value.scrollHeight
    }
  })
}

watch(statusText, scrollToBottom)

onMounted(async () => {
  try {
    user.value = await me()
  } catch {
    user.value = null
  }
  await loadModels()
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

  try {
    await streamChat(text, selectedModel.value, (event: StreamEvent) => {
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
</script>

<template>
  <main class="app-shell">
    <section v-if="!user" class="auth-panel">
      <div class="auth-card">
        <h1>MedicalQA</h1>
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
        <h2>MedicalQA</h2>
        <div class="user-box">
          <span>{{ user.username }}</span>
          <small>{{ user.is_admin ? '管理员' : '用户' }}</small>
        </div>
        <label>
          模型
          <select v-model="selectedModel">
            <option v-if="!availableModels.length" :value="null" disabled>加载中...</option>
            <option v-for="model in availableModels" :key="model" :value="model">{{ model }}</option>
          </select>
        </label>
        <button @click="submitLogout">退出登录</button>
      </aside>

      <section class="chat-panel">
        <div class="messages" ref="messagesRef">
          <article v-for="(message, index) in messages" :key="index" :class="['message', message.role]">
            <div class="bubble">{{ message.content || '...' }}</div>
            <details v-if="isAdmin && message.debug?.length" class="debug">
              <summary>调试信息</summary>
              <pre>{{ JSON.stringify(message.debug, null, 2) }}</pre>
            </details>
          </article>
          <div v-if="statusText" class="status">{{ statusText }}</div>
        </div>
        <form class="composer" @submit.prevent="sendMessage">
          <textarea v-model="input" rows="2" placeholder="描述你的问题，例如：我牙齿上长黑斑怎么回事" />
          <button class="primary" :disabled="sending">发送</button>
        </form>
      </section>
    </section>
  </main>
</template>
