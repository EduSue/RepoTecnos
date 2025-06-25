"use client"
import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { ArrowLeft, MessageSquare, Send, Bot, User, Brain } from "lucide-react"

interface TransformerModuleProps {
  onBack: () => void
}

interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
}

export default function TransformerModule({ onBack }: TransformerModuleProps) {
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: "1",
      role: "assistant",
      content: "¡Hola! Soy un asistente de IA basado en transformers. ¿En qué puedo ayudarte hoy?",
      timestamp: new Date(),
    },
  ])
  const [inputMessage, setInputMessage] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [chatMessages])

  const sendMessage = async () => {
    if (!inputMessage.trim()) return

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      content: inputMessage,
      timestamp: new Date(),
    }

    setChatMessages((prev) => [...prev, userMessage])
    const currentMessage = inputMessage
    setInputMessage("")
    setIsLoading(true)

    try {
      // Llamar a la API proporcionada
      const response = await fetch("https://catemop447.app.n8n.cloud/webhook/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: currentMessage,
        }),
      })

      // Leer la respuesta como texto primero
      const raw = await response.text()
      let data: any = raw

      try {
        if (raw) {
          data = JSON.parse(raw)
        }
      } catch {
        // Si no es JSON válido, usar el texto crudo
        data = raw
      }

      // Mostrar la respuesta completa en consola
      console.log("Respuesta completa de la API:", data)

      // Extraer el mensaje de respuesta basado en la estructura observada
      let responseContent = "Lo siento, no pude procesar tu mensaje."

      if (Array.isArray(data) && data.length > 0) {
        // La API devuelve un array con objetos
        const firstItem = data[0]
        if (firstItem && typeof firstItem === "object") {
          // Buscar la propiedad 'output' primero (estructura observada)
          if (firstItem.output) {
            responseContent = firstItem.output
          } else if (firstItem.message) {
            responseContent = firstItem.message
          } else if (firstItem.response) {
            responseContent = firstItem.response
          } else if (firstItem.text) {
            responseContent = firstItem.text
          } else if (firstItem.content) {
            responseContent = firstItem.content
          } else {
            // Si no encontramos un campo conocido, usar el primer valor string
            const firstStringValue = Object.values(firstItem).find((value) => typeof value === "string")
            if (firstStringValue) {
              responseContent = firstStringValue
            }
          }
        }
      } else if (data && typeof data === "object" && !Array.isArray(data)) {
        // Si es un objeto directo (no array)
        if (data.output) {
          responseContent = data.output
        } else if (data.message) {
          responseContent = data.message
        } else if (data.response) {
          responseContent = data.response
        } else if (data.text) {
          responseContent = data.text
        } else if (data.content) {
          responseContent = data.content
        } else {
          // Si no encontramos un campo conocido, usar el primer valor string
          const firstStringValue = Object.values(data).find((value) => typeof value === "string")
          if (firstStringValue) {
            responseContent = firstStringValue
          }
        }
      } else if (typeof data === "string" && data.trim()) {
        // Si es texto plano
        responseContent = data.trim()
      }

      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: responseContent,
        timestamp: new Date(),
      }

      setChatMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      console.error("Error al llamar a la API:", error)

      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "Lo siento, hubo un error al procesar tu mensaje. Por favor, inténtalo de nuevo.",
        timestamp: new Date(),
      }

      setChatMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Button variant="ghost" size="sm" onClick={onBack} className="mr-4">
                <ArrowLeft className="h-4 w-4 mr-2" />
                <span className="hidden sm:inline">Volver al inicio</span>
              </Button>
              <div>
                <h1 className="text-xl lg:text-2xl font-bold text-gray-900">Chat con Transformers</h1>
                <p className="text-sm lg:text-base text-gray-600 hidden sm:block">
                  Conversa con un modelo transformer preentrenado
                </p>
              </div>
            </div>
            <Badge variant="outline" className="bg-purple-50 text-purple-700">
              <Brain className="h-4 w-4 mr-1" />
              Transformer Activo
            </Badge>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Card className="h-[600px] lg:h-[700px] flex flex-col">
          <CardHeader className="pb-4">
            <CardTitle className="flex items-center text-lg">
              <MessageSquare className="h-5 w-5 mr-2" />
              Chat con IA
            </CardTitle>
            <CardDescription className="text-sm">
              Conversa con el modelo transformer. Las respuestas se muestran en consola para debugging.
            </CardDescription>
          </CardHeader>

          <CardContent className="flex-1 flex flex-col min-h-0">
            {/* Messages */}
            <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-2">
              {chatMessages.map((message) => (
                <div key={message.id} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                  <div
                    className={`max-w-[85%] lg:max-w-[80%] p-3 rounded-lg ${
                      message.role === "user" ? "bg-blue-500 text-white" : "bg-gray-100 text-gray-900"
                    }`}
                  >
                    <div className="flex items-center mb-1">
                      {message.role === "user" ? <User className="h-4 w-4 mr-2" /> : <Bot className="h-4 w-4 mr-2" />}
                      <span className="text-xs opacity-75">{message.timestamp.toLocaleTimeString()}</span>
                    </div>
                    <p className="text-sm lg:text-base whitespace-pre-wrap">{message.content}</p>
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 p-3 rounded-lg max-w-[80%]">
                    <div className="flex items-center">
                      <Bot className="h-4 w-4 mr-2" />
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                        <div
                          className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                          style={{ animationDelay: "0.1s" }}
                        />
                        <div
                          className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                          style={{ animationDelay: "0.2s" }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="flex space-x-2">
              <Input
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="Escribe tu mensaje..."
                onKeyPress={(e) => e.key === "Enter" && !e.shiftKey && sendMessage()}
                disabled={isLoading}
                className="text-sm lg:text-base"
              />
              <Button onClick={sendMessage} disabled={isLoading || !inputMessage.trim()} size="sm" className="px-4">
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </CardContent>
        </Card>

        
      </main>
    </div>
  )
}
