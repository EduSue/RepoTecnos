"use client"
import { NetworkProvider } from "./network-context"
import NetworkCanvas from "./network-canvas"
import ConfigPanel from "./config-panel"
import PropertiesPanel from "./properties-panel"
import MathSteps from "./math-steps"
import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Calculator, ArrowLeft, Menu, X } from "lucide-react"

interface NetworkModuleProps {
  onBack: () => void
}

export default function NetworkModule({ onBack }: NetworkModuleProps) {
  const [showMathSteps, setShowMathSteps] = useState(false)
  const [showConfigPanel, setShowConfigPanel] = useState(false)
  const [showPropertiesPanel, setShowPropertiesPanel] = useState(false)
  const mathStepsRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (showMathSteps && mathStepsRef.current) {
      setTimeout(() => {
        mathStepsRef.current?.scrollIntoView({ behavior: "smooth" })
      }, 100)
    }
  }, [showMathSteps])

  return (
    <NetworkProvider>
      <div className="flex h-screen bg-gray-50 text-gray-900">
        {/* Mobile Menu Buttons */}
        <div className="lg:hidden fixed top-4 left-4 z-50 flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowConfigPanel(!showConfigPanel)}
            className="bg-white shadow-md"
          >
            <Menu className="h-4 w-4" />
          </Button>
        </div>

        <div className="lg:hidden fixed top-4 right-4 z-50">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowPropertiesPanel(!showPropertiesPanel)}
            className="bg-white shadow-md"
          >
            <Menu className="h-4 w-4" />
          </Button>
        </div>

        {/* Left Configuration Panel */}
        <div
          className={`${
            showConfigPanel ? "translate-x-0" : "-translate-x-full"
          } lg:translate-x-0 fixed lg:relative z-40 transition-transform duration-300 ease-in-out lg:block`}
        >
          <div className="lg:hidden absolute top-4 right-4 z-50">
            <Button variant="ghost" size="sm" onClick={() => setShowConfigPanel(false)}>
              <X className="h-4 w-4" />
            </Button>
          </div>
          <ConfigPanel />
        </div>

        {/* Main Network Visualization */}
        <main className="flex-1 p-2 lg:p-4 overflow-auto flex flex-col min-w-0">
          <div className="bg-white rounded-xl shadow-md flex-1 flex flex-col min-h-0">
            <div className="p-2 lg:p-4 border-b flex flex-col lg:flex-row justify-between items-start lg:items-center gap-2">
              <div className="flex items-center w-full lg:w-auto">
                <Button variant="ghost" size="sm" onClick={onBack} className="mr-2 lg:mr-4 flex-shrink-0">
                  <ArrowLeft className="h-4 w-4 mr-1 lg:mr-2" />
                  <span className="hidden sm:inline">Volver</span>
                </Button>
                <div className="min-w-0 flex-1">
                  <h1 className="text-lg lg:text-2xl font-bold truncate">Constructor de Redes Neuronales</h1>
                  <p className="text-sm lg:text-base text-gray-500 hidden sm:block">
                    Construye y personaliza redes neuronales de forma visual
                  </p>
                </div>
              </div>
              <Button
                variant="outline"
                onClick={() => setShowMathSteps(!showMathSteps)}
                className="flex items-center gap-2 w-full lg:w-auto text-sm"
                size="sm"
              >
                <Calculator className="h-4 w-4" />
                {showMathSteps ? "Ocultar Pasos" : "Mostrar Pasos"}
              </Button>
            </div>
            <div className="flex-1 overflow-hidden min-h-0">
              <NetworkCanvas />
            </div>
            {showMathSteps && (
              <div ref={mathStepsRef} className="max-h-96 lg:max-h-none">
                <MathSteps />
              </div>
            )}
          </div>
        </main>

        {/* Right Properties Panel */}
        <div
          className={`${
            showPropertiesPanel ? "translate-x-0" : "translate-x-full"
          } lg:translate-x-0 fixed lg:relative z-40 transition-transform duration-300 ease-in-out lg:block`}
        >
          <div className="lg:hidden absolute top-4 left-4 z-50">
            <Button variant="ghost" size="sm" onClick={() => setShowPropertiesPanel(false)}>
              <X className="h-4 w-4" />
            </Button>
          </div>
          <PropertiesPanel />
        </div>

        {/* Overlay for mobile */}
        {(showConfigPanel || showPropertiesPanel) && (
          <div
            className="lg:hidden fixed inset-0 bg-black bg-opacity-50 z-30"
            onClick={() => {
              setShowConfigPanel(false)
              setShowPropertiesPanel(false)
            }}
          />
        )}
      </div>
    </NetworkProvider>
  )
}
