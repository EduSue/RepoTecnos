"use client"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Brain, Camera, MessageSquare, ArrowRight, Zap, Eye, Cpu, BookOpen, Github } from "lucide-react"
import NetworkModule from "@/components/network-module"
import ImageRecognitionModule from "@/components/image-recognition-module"
import TransformerModule from "@/components/transformer-module"

type ModuleType = "home" | "neural-network" | "image-recognition" | "transformer"

export default function Home() {
  const [currentModule, setCurrentModule] = useState<ModuleType>("home")

  const modules = [
    {
      id: "neural-network" as ModuleType,
      title: "Redes Neuronales",
      subtitle: "Propagación hacia adelante y hacia atrás",
      description:
        "Construye y entrena redes neuronales desde cero. Visualiza el proceso completo de propagación hacia adelante y retropropagación con cálculos matemáticos detallados.",
      icon: Brain,
      color: "bg-blue-500",
      features: [
        "Constructor visual de redes",
        "Propagación hacia adelante",
        "Algoritmo de retropropagación",
        "Visualización de cálculos matemáticos",
        "Ajuste de hiperparámetros",
      ],
      status: "Completado",
    },
    {
      id: "image-recognition" as ModuleType,
      title: "Reconocimiento de Imágenes",
      subtitle: "Redes Neuronales Convolucionales (CNN)",
      description:
        "Implementa sistemas de reconocimiento de imágenes usando CNNs. Compatible con dispositivos móviles para captura y clasificación en tiempo real.",
      icon: Camera,
      color: "bg-green-500",
      features: [
        "Modelo CNN CIFAR-10",
        "Captura desde dispositivo móvil",
        "Clasificación en tiempo real",
        "Visualización de arquitectura",
        "Métricas de rendimiento",
      ],
      status: "Completado",
    },
    {
      id: "transformer" as ModuleType,
      title: "Chat con Transformers",
      subtitle: "Modelos preentrenados y Chat IA",
      description:
        "Conversa con modelos transformer preentrenados. Integración con API externa para procesamiento de lenguaje natural avanzado.",
      icon: MessageSquare,
      color: "bg-purple-500",
      features: [
        "Chat IA interactivo",
        "Integración con API externa",
        "Respuestas en tiempo real",
        "Debugging en consola",
        "Interfaz responsive",
      ],
      status: "Completado",
    },
  ]

  const renderHome = () => (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
            <div>
              <h1 className="text-2xl lg:text-3xl font-bold text-gray-900">Sistema Experimental de Machine Learning</h1>
              <p className="text-base lg:text-lg text-gray-600 mt-2">Métodos de Machine Learning y Redes Neuronales</p>
            </div>
            <div className="flex items-center space-x-4">
              <Badge variant="outline" className="text-sm">
                <Cpu className="h-4 w-4 mr-1" />
                Experimental
              </Badge>
              <Button variant="outline" size="sm" className="hidden sm:flex">
                <Github className="h-4 w-4 mr-2" />
                Documentación
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Introduction */}
        <div className="text-center mb-16">
          <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-4">
            Explora los Fundamentos del Machine Learning
          </h2>
          <p className="text-lg lg:text-xl text-gray-600 max-w-3xl mx-auto">
            Un sistema completo para experimentar con diferentes técnicas de machine learning, desde redes neuronales
            básicas hasta modelos transformer avanzados.
          </p>
        </div>

        {/* Modules Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-16">
          {modules.map((module) => {
            const IconComponent = module.icon
            return (
              <Card
                key={module.id}
                className="relative overflow-hidden hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1"
              >
                <CardHeader className="pb-4">
                  <div className="flex items-center justify-between mb-4">
                    <div className={`${module.color} p-3 rounded-lg text-white`}>
                      <IconComponent className="h-6 w-6" />
                    </div>
                    <Badge variant="default" className="text-xs bg-green-500">
                      {module.status}
                    </Badge>
                  </div>
                  <CardTitle className="text-lg lg:text-xl font-bold">{module.title}</CardTitle>
                  <CardDescription className="text-sm font-medium text-blue-600">{module.subtitle}</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-600 mb-6 leading-relaxed text-sm lg:text-base">{module.description}</p>

                  <div className="space-y-2 mb-6">
                    <h4 className="font-semibold text-sm text-gray-900">Características:</h4>
                    <ul className="space-y-1">
                      {module.features.map((feature, index) => (
                        <li key={index} className="flex items-center text-xs lg:text-sm text-gray-600">
                          <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mr-2" />
                          {feature}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <Button onClick={() => setCurrentModule(module.id)} className="w-full group text-sm">
                    Abrir Módulo
                    <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform" />
                  </Button>
                </CardContent>
              </Card>
            )
          })}
        </div>

        {/* Features Overview */}
        <div className="bg-white rounded-2xl shadow-lg p-6 lg:p-8 mb-16">
          <h3 className="text-xl lg:text-2xl font-bold text-gray-900 mb-8 text-center">Características del Sistema</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="bg-blue-100 p-4 rounded-full w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                <Eye className="h-8 w-8 text-blue-600" />
              </div>
              <h4 className="font-semibold text-base lg:text-lg mb-2">Visualización Interactiva</h4>
              <p className="text-gray-600 text-sm lg:text-base">
                Observa en tiempo real cómo funcionan los algoritmos de machine learning
              </p>
            </div>
            <div className="text-center">
              <div className="bg-green-100 p-4 rounded-full w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                <Zap className="h-8 w-8 text-green-600" />
              </div>
              <h4 className="font-semibold text-base lg:text-lg mb-2">Experimentación Práctica</h4>
              <p className="text-gray-600 text-sm lg:text-base">
                Modifica parámetros y observa inmediatamente los resultados
              </p>
            </div>
            <div className="text-center">
              <div className="bg-purple-100 p-4 rounded-full w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                <BookOpen className="h-8 w-8 text-purple-600" />
              </div>
              <h4 className="font-semibold text-base lg:text-lg mb-2">Aprendizaje Guiado</h4>
              <p className="text-gray-600 text-sm lg:text-base">
                Explicaciones detalladas de cada proceso y cálculo matemático
              </p>
            </div>
          </div>
        </div>

        {/* Quick Start */}
        <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl p-6 lg:p-8 text-white text-center">
          <h3 className="text-xl lg:text-2xl font-bold mb-4">¿Listo para comenzar?</h3>
          <p className="text-base lg:text-lg mb-6 opacity-90">
            Comienza con el módulo de Redes Neuronales para entender los fundamentos
          </p>
          <Button
            onClick={() => setCurrentModule("neural-network")}
            size="lg"
            variant="secondary"
            className="bg-white text-blue-600 hover:bg-gray-100"
          >
            Comenzar con Redes Neuronales
            <ArrowRight className="ml-2 h-5 w-5" />
          </Button>
        </div>
      </main>
    </div>
  )

  const renderModule = () => {
    switch (currentModule) {
      case "neural-network":
        return <NetworkModule onBack={() => setCurrentModule("home")} />
      case "image-recognition":
        return <ImageRecognitionModule onBack={() => setCurrentModule("home")} />
      case "transformer":
        return <TransformerModule onBack={() => setCurrentModule("home")} />
      default:
        return renderHome()
    }
  }

  return renderModule()
}
