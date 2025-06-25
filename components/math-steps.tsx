"use client"

import { useState } from "react"
import { useNetwork, type MathStep } from "./network-context"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { ChevronDown, ChevronRight, X, ChevronUp, ChevronDownIcon } from "lucide-react"
import { Progress } from "@/components/ui/progress"

export default function MathSteps() {
  const { network, clearMathSteps } = useNetwork()
  const { forwardPropagationSteps, backpropagationSteps, currentEpoch, totalEpochs, trainingError, layers } = network

  // Obtener la tasa de aprendizaje del contexto (podría venir de config-panel.tsx)
  const learningRate = 0.01 // Valor por defecto

  const [expandedSteps, setExpandedSteps] = useState<Record<string, boolean>>({})
  const [currentEpochView, setCurrentEpochView] = useState(1)
  const [expandedEpochs, setExpandedEpochs] = useState<Record<number, boolean>>({ 1: true })

  // Función para alternar la expansión de un paso
  const toggleStep = (stepId: string) => {
    setExpandedSteps((prev) => ({
      ...prev,
      [stepId]: !prev[stepId],
    }))
  }

  // Función para alternar la expansión de una época
  const toggleEpoch = (epochNumber: number) => {
    setExpandedEpochs((prev) => ({
      ...prev,
      [epochNumber]: !prev[epochNumber],
    }))
  }

  // Función para renderizar un paso matemático
  const renderMathStep = (step: MathStep, index: number, prefix = "") => {
    const stepId = `${prefix}-${index}`
    const isExpanded = expandedSteps[stepId] || false

    return (
      <div key={stepId} className="mb-2">
        <div
          className="flex items-start p-2 bg-gray-50 rounded-md hover:bg-gray-100 cursor-pointer"
          onClick={() => (step.substeps && step.substeps.length > 0 ? toggleStep(stepId) : null)}
        >
          {step.substeps && step.substeps.length > 0 ? (
            <div className="mt-1 mr-2">
              {isExpanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
            </div>
          ) : (
            <div className="w-6" />
          )}

          <div className="flex-1">
            <div className="font-medium">{step.description}</div>
            <div className="text-sm font-mono mt-1">{step.formula}</div>
            <div className="text-sm text-blue-600 mt-1">
              Resultado: {typeof step.result === "number" ? step.result.toFixed(6) : step.result}
            </div>
          </div>
        </div>

        {isExpanded && step.substeps && (
          <div className="ml-6 mt-2 pl-2 border-l-2 border-gray-200">
            {step.substeps.map((substep, subIndex) => renderMathStep(substep, subIndex, `${stepId}-sub`))}
          </div>
        )}
      </div>
    )
  }

  // Renderizar los pasos de propagación hacia adelante
  const renderForwardPropagation = () => {
    if (!forwardPropagationSteps) {
      return (
        <div className="text-center py-8 text-gray-500">
          <p>No hay cálculos de propagación hacia adelante disponibles.</p>
          <p className="mt-2">Haz clic en "Calcular Salida" para ver los pasos matemáticos.</p>
        </div>
      )
    }

    return (
      <div className="space-y-6">
        {forwardPropagationSteps.layerSteps.map((layerStep, layerIndex) => (
          <Card key={`layer-${layerIndex}`}>
            <CardHeader className="pb-2">
              <CardTitle>
                {layerStep.layerType === "input"
                  ? "Capa de Entrada"
                  : layerStep.layerType === "output"
                    ? "Capa de Salida"
                    : `Capa Oculta ${layerStep.layerId.split("-").pop()}`}
              </CardTitle>
              <CardDescription>
                {layerStep.layerType === "input" ? "Valores de entrada (no hay cálculos)" : "Cálculo de activaciones"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {layerStep.neurons.map((neuron, neuronIndex) => (
                <div key={`neuron-${neuronIndex}`} className="mb-4">
                  <h4 className="text-sm font-semibold mb-2 pb-1 border-b">
                    Neurona {neuron.neuronId.split("-").pop()}
                  </h4>
                  {neuron.steps.map((step, stepIndex) =>
                    renderMathStep(step, stepIndex, `forward-${layerIndex}-${neuronIndex}`),
                  )}
                </div>
              ))}
            </CardContent>
          </Card>
        ))}
      </div>
    )
  }

  // Renderizar la secuencia completa de entrenamiento para todas las épocas
  const renderTrainingSequence = () => {
    if (!backpropagationSteps || backpropagationSteps.epochSteps.length === 0) {
      return (
        <div className="text-center py-8 text-gray-500">
          <p>No hay cálculos de retropropagación disponibles.</p>
          <p className="mt-2">Haz clic en "Entrenar Red" para ver los pasos matemáticos.</p>
        </div>
      )
    }

    return (
      <div className="space-y-6">
        <div className="bg-blue-50 p-4 rounded-md mb-6">
          <h3 className="text-lg font-semibold mb-2">Proceso de Entrenamiento</h3>
          <p className="text-sm text-gray-600">
            El entrenamiento se realiza en {backpropagationSteps.epochSteps.length} épocas con una tasa de aprendizaje
            de {learningRate}. Cada época sigue estos pasos:
          </p>
          <ol className="list-decimal list-inside mt-2 space-y-1 text-sm">
            <li>Propagación hacia adelante: calcular las salidas con los pesos y sesgos actuales</li>
            <li>Cálculo del error: comparar las salidas con los valores esperados</li>
            <li>Retropropagación: calcular los gradientes del error</li>
            <li>Actualización de pesos y sesgos: ajustar los parámetros de la red</li>
            <li>Nueva propagación hacia adelante: verificar los resultados con los nuevos parámetros</li>
          </ol>
        </div>

        <div className="mb-4">
          <Progress value={(currentEpoch / totalEpochs) * 100} className="h-2" />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>Época 1</span>
            <span>Época {Math.floor(totalEpochs / 2)}</span>
            <span>Época {totalEpochs}</span>
          </div>
        </div>

        {/* Renderizar cada época */}
        {backpropagationSteps.epochSteps.map((epochStep, epochIndex) => {
          const epochNumber = epochIndex + 1
          const isExpanded = expandedEpochs[epochNumber] || false
          const initialError = epochStep.error
          const nextEpochError =
            epochIndex < backpropagationSteps.epochSteps.length - 1
              ? backpropagationSteps.epochSteps[epochIndex + 1].error
              : null

          return (
            <Card
              key={`epoch-${epochNumber}`}
              className={`border-l-4 ${isExpanded ? "border-l-blue-500" : "border-l-gray-300"}`}
            >
              <CardHeader
                className={`cursor-pointer ${isExpanded ? "bg-blue-50" : "hover:bg-gray-50"}`}
                onClick={() => toggleEpoch(epochNumber)}
              >
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg flex items-center">
                    <span className="mr-2">Época {epochNumber}</span>
                    {nextEpochError !== null && (
                      <span
                        className={`text-sm font-normal ${nextEpochError < initialError ? "text-green-600" : "text-red-600"}`}
                      >
                        {nextEpochError < initialError
                          ? `Error ↓ ${(initialError - nextEpochError).toFixed(6)}`
                          : `Error ↑ ${(nextEpochError - initialError).toFixed(6)}`}
                      </span>
                    )}
                  </CardTitle>
                  <div className="flex items-center">
                    <span className="text-sm mr-4">Error: {initialError.toFixed(6)}</span>
                    {isExpanded ? <ChevronUp className="h-5 w-5" /> : <ChevronDownIcon className="h-5 w-5" />}
                  </div>
                </div>
              </CardHeader>

              {isExpanded && (
                <CardContent>
                  <div className="space-y-6">
                    {/* PASO 1: Propagación hacia adelante inicial */}
                    <div className="border-l-4 border-blue-400 pl-4">
                      <h3 className="text-md font-semibold mb-3 flex items-center">
                        <div className="bg-blue-100 text-blue-800 rounded-full w-6 h-6 flex items-center justify-center font-bold mr-2">
                          1
                        </div>
                        Propagación Hacia Adelante (Época {epochNumber})
                      </h3>

                      {epochStep.forwardPropagationSteps ? (
                        <Tabs defaultValue="input" className="mt-2">
                          <TabsList className="mb-2">
                            <TabsTrigger value="input">Capa de Entrada</TabsTrigger>
                            {epochStep.forwardPropagationSteps.layerSteps
                              .filter((layer) => layer.layerType === "hidden")
                              .map((layer, idx) => (
                                <TabsTrigger key={`hidden-${idx}`} value={`hidden-${idx}`}>
                                  Capa Oculta {idx + 1}
                                </TabsTrigger>
                              ))}
                            <TabsTrigger value="output">Capa de Salida</TabsTrigger>
                          </TabsList>

                          <TabsContent value="input">
                            {epochStep.forwardPropagationSteps.layerSteps
                              .find((layer) => layer.layerType === "input")
                              ?.neurons.map((neuron, neuronIndex) => (
                                <div key={`neuron-${neuronIndex}`} className="mb-2">
                                  <h4 className="text-sm font-semibold mb-1 pb-1 border-b">
                                    Neurona {neuron.neuronId.split("-").pop()}
                                  </h4>
                                  {neuron.steps.map((step, stepIndex) =>
                                    renderMathStep(step, stepIndex, `forward-input-${epochNumber}-${neuronIndex}`),
                                  )}
                                </div>
                              ))}
                          </TabsContent>

                          {epochStep.forwardPropagationSteps.layerSteps
                            .filter((layer) => layer.layerType === "hidden")
                            .map((layer, layerIdx) => (
                              <TabsContent key={`hidden-content-${layerIdx}`} value={`hidden-${layerIdx}`}>
                                {layer.neurons.map((neuron, neuronIndex) => (
                                  <div key={`neuron-${neuronIndex}`} className="mb-2">
                                    <h4 className="text-sm font-semibold mb-1 pb-1 border-b">
                                      Neurona {neuron.neuronId.split("-").pop()}
                                    </h4>
                                    {neuron.steps.map((step, stepIndex) =>
                                      renderMathStep(
                                        step,
                                        stepIndex,
                                        `forward-hidden-${epochNumber}-${layerIdx}-${neuronIndex}`,
                                      ),
                                    )}
                                  </div>
                                ))}
                              </TabsContent>
                            ))}

                          <TabsContent value="output">
                            {epochStep.forwardPropagationSteps.layerSteps
                              .find((layer) => layer.layerType === "output")
                              ?.neurons.map((neuron, neuronIndex) => (
                                <div key={`neuron-${neuronIndex}`} className="mb-2">
                                  <h4 className="text-sm font-semibold mb-1 pb-1 border-b">
                                    Neurona {neuron.neuronId.split("-").pop()}
                                  </h4>
                                  {neuron.steps.map((step, stepIndex) =>
                                    renderMathStep(step, stepIndex, `forward-output-${epochNumber}-${neuronIndex}`),
                                  )}
                                </div>
                              ))}
                          </TabsContent>
                        </Tabs>
                      ) : (
                        <div className="text-sm text-gray-500">
                          No hay datos disponibles para la propagación hacia adelante.
                        </div>
                      )}
                    </div>

                    {/* PASO 2: Cálculo del Error */}
                    <div className="border-l-4 border-red-400 pl-4">
                      <h3 className="text-md font-semibold mb-3 flex items-center">
                        <div className="bg-red-100 text-red-800 rounded-full w-6 h-6 flex items-center justify-center font-bold mr-2">
                          2
                        </div>
                        Cálculo del Error (Época {epochNumber})
                      </h3>

                      <div className="bg-white p-3 rounded-md border">
                        <div className="grid grid-cols-3 gap-4 font-medium text-sm bg-gray-50 p-2 rounded-md">
                          <div>Neurona</div>
                          <div>Valor Esperado</div>
                          <div>Valor Obtenido</div>
                        </div>

                        {epochStep.outputGradients
                          .filter((step) => step.description.includes("Error para"))
                          .map((step, index) => {
                            const neuronLabel = step.description.replace("Error para ", "")
                            const expectedValue = step.formula.match(/0\.5 \* \(([^)]+) - /)?.[1] || "N/A"
                            const obtainedValue = step.formula.match(/ - ([^)]+)\)\^2/)?.[1] || "N/A"

                            return (
                              <div key={index} className="grid grid-cols-3 gap-4 text-sm border-t pt-2">
                                <div>{neuronLabel}</div>
                                <div>{expectedValue}</div>
                                <div>{obtainedValue}</div>
                              </div>
                            )
                          })}

                        <div className="mt-4 pt-2 border-t">
                          <div className="font-medium">Error total: {epochStep.error.toFixed(6)}</div>
                          <p className="text-xs text-gray-600 mt-1">E = 0.5 * Σ(esperado - obtenido)²</p>
                        </div>
                      </div>
                    </div>

                    {/* PASO 3: Retropropagación (cálculo de gradientes) */}
                    <div className="border-l-4 border-purple-400 pl-4">
                      <h3 className="text-md font-semibold mb-3 flex items-center">
                        <div className="bg-purple-100 text-purple-800 rounded-full w-6 h-6 flex items-center justify-center font-bold mr-2">
                          3
                        </div>
                        Retropropagación - Regla de la Cadena (Época {epochNumber})
                      </h3>

                      <div className="bg-purple-50 p-3 rounded-md border border-purple-100 mb-4">
                        <h4 className="font-medium mb-2">Explicación de la Regla de la Cadena</h4>
                        <p className="text-sm text-gray-700">
                          La retropropagación utiliza la regla de la cadena del cálculo para propagar el error desde la
                          salida hacia atrás:
                        </p>
                        <ol className="list-decimal list-inside mt-2 space-y-1 text-sm text-gray-700">
                          <li>Para la capa de salida: δ = -(esperado - obtenido) * f'(z)</li>
                          <li>Para las capas ocultas: δ = (Σ δ_siguiente * w) * f'(z)</li>
                          <li>Donde f'(z) es la derivada de la función de activación</li>
                        </ol>
                      </div>

                      <Tabs defaultValue="output" className="mt-2">
                        <TabsList className="mb-2">
                          <TabsTrigger value="output">Capa de Salida</TabsTrigger>
                          <TabsTrigger value="hidden">Capas Ocultas</TabsTrigger>
                        </TabsList>

                        <TabsContent value="output">
                          <div className="space-y-2">
                            <p className="text-sm text-gray-600 mb-2">
                              <strong>Regla de la cadena para la capa de salida:</strong> Calculamos el gradiente del
                              error con respecto a la salida y lo multiplicamos por la derivada de la función de
                              activación.
                            </p>
                            {epochStep.outputGradients
                              .filter((step) => step.description.includes("Gradiente para"))
                              .map((step, index) => (
                                <div key={index} className="bg-white p-3 rounded-md border mb-3">
                                  <h5 className="font-medium text-purple-700 mb-2">{step.description}</h5>
                                  <div className="grid grid-cols-1 gap-2">
                                    <div className="bg-purple-50 p-2 rounded-md">
                                      <div className="font-medium text-sm">Fórmula:</div>
                                      <div className="font-mono text-sm">{step.formula}</div>
                                    </div>
                                    {step.substeps &&
                                      step.substeps.map((substep, subIndex) => (
                                        <div key={subIndex} className="bg-gray-50 p-2 rounded-md">
                                          <div className="font-medium text-sm">{substep.description}:</div>
                                          <div className="font-mono text-sm">{substep.formula}</div>
                                          <div className="text-sm text-blue-600">
                                            Resultado:{" "}
                                            {typeof substep.result === "number"
                                              ? substep.result.toFixed(6)
                                              : substep.result}
                                          </div>
                                        </div>
                                      ))}
                                    <div className="bg-green-50 p-2 rounded-md">
                                      <div className="font-medium text-sm">Resultado final (delta):</div>
                                      <div className="text-green-700 font-medium">
                                        {typeof step.result === "number" ? step.result.toFixed(6) : step.result}
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              ))}
                          </div>
                        </TabsContent>

                        <TabsContent value="hidden">
                          <div className="space-y-2">
                            <p className="text-sm text-gray-600 mb-2">
                              <strong>Regla de la cadena para las capas ocultas:</strong> Propagamos el error desde la
                              capa siguiente, ponderado por los pesos de las conexiones, y multiplicamos por la derivada
                              de la función de activación.
                            </p>
                            {epochStep.hiddenGradients.map((step, index) => (
                              <div key={index} className="bg-white p-3 rounded-md border mb-3">
                                <h5 className="font-medium text-purple-700 mb-2">{step.description}</h5>
                                <div className="grid grid-cols-1 gap-2">
                                  <div className="bg-purple-50 p-2 rounded-md">
                                    <div className="font-medium text-sm">Fórmula:</div>
                                    <div className="font-mono text-sm">{step.formula}</div>
                                  </div>
                                  {step.substeps &&
                                    step.substeps.map((substep, subIndex) => (
                                      <div key={subIndex} className="bg-gray-50 p-2 rounded-md">
                                        <div className="font-medium text-sm">{substep.description}:</div>
                                        <div className="font-mono text-sm">{substep.formula}</div>
                                        <div className="text-sm text-blue-600">
                                          Resultado:{" "}
                                          {typeof substep.result === "number"
                                            ? substep.result.toFixed(6)
                                            : substep.result}
                                        </div>
                                        {substep.substeps && (
                                          <div className="ml-4 mt-2 pl-2 border-l-2 border-gray-200">
                                            {substep.substeps.map((subsubstep, subsubIndex) => (
                                              <div key={subsubIndex} className="mb-2">
                                                <div className="text-xs">{subsubstep.description}:</div>
                                                <div className="font-mono text-xs">{subsubstep.formula}</div>
                                                <div className="text-xs text-blue-600">
                                                  ={" "}
                                                  {typeof subsubstep.result === "number"
                                                    ? subsubstep.result.toFixed(6)
                                                    : subsubstep.result}
                                                </div>
                                              </div>
                                            ))}
                                          </div>
                                        )}
                                      </div>
                                    ))}
                                  <div className="bg-green-50 p-2 rounded-md">
                                    <div className="font-medium text-sm">Resultado final (delta):</div>
                                    <div className="text-green-700 font-medium">
                                      {typeof step.result === "number" ? step.result.toFixed(6) : step.result}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </TabsContent>
                      </Tabs>
                    </div>

                    {/* PASO 4: Actualización de pesos y sesgos */}
                    <div className="border-l-4 border-green-400 pl-4">
                      <h3 className="text-md font-semibold mb-3 flex items-center">
                        <div className="bg-green-100 text-green-800 rounded-full w-6 h-6 flex items-center justify-center font-bold mr-2">
                          4
                        </div>
                        Actualización de Pesos y Sesgos (Época {epochNumber})
                      </h3>

                      <div className="bg-green-50 p-3 rounded-md border border-green-100 mb-4">
                        <h4 className="font-medium mb-2">Fórmulas de Actualización</h4>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <p className="font-medium">Actualización de pesos:</p>
                            <p className="font-mono">w_nuevo = w_anterior - η * δ * a</p>
                            <p className="text-gray-600 mt-1">
                              Donde η es la tasa de aprendizaje ({learningRate}), δ es el gradiente calculado en el paso
                              anterior, y a es la activación de la neurona de origen.
                            </p>
                          </div>
                          <div>
                            <p className="font-medium">Actualización de sesgos:</p>
                            <p className="font-mono">b_nuevo = b_anterior - η * δ</p>
                            <p className="text-gray-600 mt-1">
                              Donde η es la tasa de aprendizaje ({learningRate}) y δ es el gradiente calculado en el
                              paso anterior.
                            </p>
                          </div>
                        </div>
                      </div>

                      <Tabs defaultValue="weights" className="mt-2">
                        <TabsList className="mb-2">
                          <TabsTrigger value="weights">Actualización de Pesos</TabsTrigger>
                          <TabsTrigger value="biases">Actualización de Sesgos</TabsTrigger>
                        </TabsList>

                        <TabsContent value="weights">
                          <div className="space-y-2">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                              {epochStep.weightUpdates.map((step, index) => (
                                <div key={index} className="bg-white p-3 rounded-md border">
                                  <h5 className="font-medium text-green-700 mb-2">{step.description}</h5>
                                  <div className="grid grid-cols-1 gap-2">
                                    <div className="bg-green-50 p-2 rounded-md">
                                      <div className="font-medium text-sm">Fórmula:</div>
                                      <div className="font-mono text-sm">{step.formula}</div>
                                    </div>
                                    {step.substeps &&
                                      step.substeps.map((substep, subIndex) => (
                                        <div key={subIndex} className="bg-gray-50 p-2 rounded-md">
                                          <div className="font-medium text-sm">{substep.description}:</div>
                                          <div className="font-mono text-sm">{substep.formula}</div>
                                        </div>
                                      ))}
                                    <div className="bg-blue-50 p-2 rounded-md">
                                      <div className="font-medium text-sm">Cambio en el peso:</div>
                                      <div className="text-blue-700 font-medium">
                                        {typeof step.result === "number" ? step.result.toFixed(6) : step.result}
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        </TabsContent>

                        <TabsContent value="biases">
                          <div className="space-y-2">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                              {epochStep.biasUpdates.map((step, index) => (
                                <div key={index} className="bg-white p-3 rounded-md border">
                                  <h5 className="font-medium text-green-700 mb-2">{step.description}</h5>
                                  <div className="grid grid-cols-1 gap-2">
                                    <div className="bg-green-50 p-2 rounded-md">
                                      <div className="font-medium text-sm">Fórmula:</div>
                                      <div className="font-mono text-sm">{step.formula}</div>
                                    </div>
                                    {step.substeps &&
                                      step.substeps.map((substep, subIndex) => (
                                        <div key={subIndex} className="bg-gray-50 p-2 rounded-md">
                                          <div className="font-medium text-sm">{substep.description}:</div>
                                          <div className="font-mono text-sm">{substep.formula}</div>
                                        </div>
                                      ))}
                                    <div className="bg-blue-50 p-2 rounded-md">
                                      <div className="font-medium text-sm">Cambio en el sesgo:</div>
                                      <div className="text-blue-700 font-medium">
                                        {typeof step.result === "number" ? step.result.toFixed(6) : step.result}
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        </TabsContent>
                      </Tabs>
                    </div>

                    {/* PASO 5: Resultados después de la actualización */}
                    <div className="border-l-4 border-blue-400 pl-4">
                      <h3 className="text-md font-semibold mb-3 flex items-center">
                        <div className="bg-blue-100 text-blue-800 rounded-full w-6 h-6 flex items-center justify-center font-bold mr-2">
                          5
                        </div>
                        Resultados Después de la Actualización (Época {epochNumber})
                      </h3>

                      <div className="bg-white p-3 rounded-md border">
                        {epochIndex < backpropagationSteps.epochSteps.length - 1 ? (
                          <>
                            <div className="grid grid-cols-3 gap-4 font-medium text-sm bg-gray-50 p-2 rounded-md">
                              <div>Neurona</div>
                              <div>Valor Esperado</div>
                              <div>Nuevo Valor</div>
                            </div>

                            {/* Mostrar los resultados para cada neurona de salida */}
                            {epochStep.outputGradients
                              .filter((step) => step.description.includes("Error para"))
                              .map((step, index) => {
                                const neuronLabel = step.description.replace("Error para ", "")
                                const expectedValue = step.formula.match(/0\.5 \* \(([^)]+) - /)?.[1] || "N/A"

                                // Obtener el valor de la siguiente época
                                const nextEpoch = backpropagationSteps.epochSteps[epochIndex + 1]
                                const nextValue =
                                  nextEpoch?.outputGradients
                                    .find((s) => s.description === step.description)
                                    ?.formula.match(/ - ([^)]+)\)\^2/)?.[1] || "N/A"

                                return (
                                  <div key={index} className="grid grid-cols-3 gap-4 text-sm border-t pt-2">
                                    <div>{neuronLabel}</div>
                                    <div>{expectedValue}</div>
                                    <div className="text-blue-600 font-medium">{nextValue}</div>
                                  </div>
                                )
                              })}

                            <div className="mt-4 pt-2 border-t">
                              <div className="font-medium">
                                Nuevo error: {backpropagationSteps.epochSteps[epochIndex + 1].error.toFixed(6)}
                                {backpropagationSteps.epochSteps[epochIndex + 1].error < epochStep.error ? (
                                  <span className="text-green-600 ml-2">
                                    ↓{" "}
                                    {(epochStep.error - backpropagationSteps.epochSteps[epochIndex + 1].error).toFixed(
                                      6,
                                    )}
                                  </span>
                                ) : (
                                  <span className="text-red-600 ml-2">
                                    ↑{" "}
                                    {(backpropagationSteps.epochSteps[epochIndex + 1].error - epochStep.error).toFixed(
                                      6,
                                    )}
                                  </span>
                                )}
                              </div>
                              <p className="text-xs text-gray-600 mt-1">
                                Estos son los nuevos valores después de actualizar los pesos y sesgos. La siguiente
                                época comenzará con estos valores.
                              </p>
                            </div>
                          </>
                        ) : (
                          // Para la última época, mostrar los resultados finales
                          <>
                            <div className="grid grid-cols-3 gap-4 font-medium text-sm bg-gray-50 p-2 rounded-md">
                              <div>Neurona</div>
                              <div>Valor Esperado</div>
                              <div>Valor Final</div>
                            </div>

                            {epochStep.outputGradients
                              .filter((step) => step.description.includes("Error para"))
                              .map((step, index) => {
                                const neuronLabel = step.description.replace("Error para ", "")
                                const expectedValue = step.formula.match(/0\.5 \* \(([^)]+) - /)?.[1] || "N/A"
                                const finalValue = step.formula.match(/ - ([^)]+)\)\^2/)?.[1] || "N/A"

                                return (
                                  <div key={index} className="grid grid-cols-3 gap-4 text-sm border-t pt-2">
                                    <div>{neuronLabel}</div>
                                    <div>{expectedValue}</div>
                                    <div className="text-green-600 font-medium">{finalValue}</div>
                                  </div>
                                )
                              })}

                            <div className="mt-4 pt-2 border-t">
                              <div className="font-medium">Error final: {epochStep.error.toFixed(6)}</div>
                              <p className="text-xs text-gray-600 mt-1">
                                Entrenamiento completado. Estos son los valores finales de la red después de{" "}
                                {backpropagationSteps.epochSteps.length} épocas.
                              </p>
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                </CardContent>
              )}
            </Card>
          )
        })}

        {/* Resumen final del entrenamiento */}
        <Card className="mt-6 border-green-200 bg-green-50">
          <CardHeader className="pb-2">
            <CardTitle className="text-green-700">Resumen del Entrenamiento</CardTitle>
            <CardDescription>
              Resultados después de {backpropagationSteps.epochSteps.length} épocas de entrenamiento
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-white p-3 rounded-md border">
                  <h4 className="font-medium mb-2">Error</h4>
                  <div className="flex justify-between items-center">
                    <div>
                      <div className="text-sm text-gray-600">Inicial:</div>
                      <div className="text-lg">{backpropagationSteps.epochSteps[0].error.toFixed(6)}</div>
                    </div>
                    <div className="text-2xl text-gray-300">→</div>
                    <div>
                      <div className="text-sm text-gray-600">Final:</div>
                      <div className="text-lg text-green-600">
                        {backpropagationSteps.epochSteps[backpropagationSteps.epochSteps.length - 1].error.toFixed(6)}
                      </div>
                    </div>
                  </div>
                  <div className="mt-2 text-sm text-green-600">
                    Reducción:{" "}
                    {(
                      backpropagationSteps.epochSteps[0].error -
                      backpropagationSteps.epochSteps[backpropagationSteps.epochSteps.length - 1].error
                    ).toFixed(6)}
                    (
                    {(
                      (1 -
                        backpropagationSteps.epochSteps[backpropagationSteps.epochSteps.length - 1].error /
                          backpropagationSteps.epochSteps[0].error) *
                      100
                    ).toFixed(2)}
                    %)
                  </div>
                </div>

                <div className="bg-white p-3 rounded-md border">
                  <h4 className="font-medium mb-2">Parámetros</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Épocas:</span>
                      <span className="font-medium">{backpropagationSteps.epochSteps.length}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Tasa de aprendizaje:</span>
                      <span className="font-medium">{learningRate}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Neuronas de entrada:</span>
                      <span className="font-medium">
                        {network.layers.find((l) => l.type === "input")?.neurons.length || 0}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Neuronas ocultas:</span>
                      <span className="font-medium">
                        {network.layers
                          .filter((l) => l.type === "hidden")
                          .reduce((acc, layer) => acc + layer.neurons.length, 0)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Neuronas de salida:</span>
                      <span className="font-medium">
                        {network.layers.find((l) => l.type === "output")?.neurons.length || 0}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white p-3 rounded-md border">
                <h4 className="font-medium mb-2">Comparación de Valores</h4>
                <div className="grid grid-cols-3 gap-4 font-medium text-sm bg-gray-50 p-2 rounded-md">
                  <div>Neurona</div>
                  <div>Valor Esperado</div>
                  <div>Valor Final</div>
                </div>

                {network.layers
                  .find((l) => l.type === "output")
                  ?.neurons.map((neuron, index) => {
                    const expectedValue =
                      backpropagationSteps.epochSteps[0].outputGradients
                        .find((step) => step.description.includes(`Error para ${neuron.label}`))
                        ?.formula.match(/0\.5 \* \(([^)]+) - /)?.[1] || "N/A"

                    return (
                      <div key={neuron.id} className="grid grid-cols-3 gap-4 text-sm border-t pt-2">
                        <div>{neuron.label}</div>
                        <div>{expectedValue}</div>
                        <div className="font-medium text-green-700">{neuron.activationValue.toFixed(6)}</div>
                      </div>
                    )
                  })}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  // Renderizar el gráfico de error
  const renderErrorGraph = () => {
    if (!trainingError || trainingError.length === 0) {
      return (
        <div className="text-center py-8 text-gray-500">
          <p>No hay datos de error disponibles.</p>
          <p className="mt-2">Entrena la red para ver la evolución del error.</p>
        </div>
      )
    }

    const maxError = Math.max(...trainingError)
    const minError = Math.min(...trainingError)
    const range = maxError - minError

    return (
      <div className="h-64 mt-4">
        <div className="flex h-full">
          <div className="w-12 h-full flex flex-col justify-between text-xs text-gray-500">
            {[...Array(5)].map((_, i) => {
              const value = maxError - (range * i) / 4
              return <div key={i}>{value.toFixed(4)}</div>
            })}
          </div>
          <div className="flex-1 h-full relative border-l border-b">
            {trainingError.map((error, index) => {
              const normalizedError = range === 0 ? 0 : (maxError - error) / range
              const height = `${normalizedError * 100}%`
              const width = `${100 / trainingError.length}%`
              const left = `${(index / trainingError.length) * 100}%`

              return (
                <div
                  key={index}
                  className="absolute bottom-0 bg-blue-500 hover:bg-blue-600 transition-colors"
                  style={{
                    height,
                    width,
                    left,
                    minWidth: "2px",
                  }}
                  title={`Época ${index + 1}: Error = ${error.toFixed(6)}`}
                />
              )
            })}
          </div>
        </div>
        <div className="flex mt-2">
          <div className="w-12"></div>
          <div className="flex-1 flex justify-between text-xs text-gray-500">
            <div>1</div>
            <div>{Math.floor(trainingError.length / 2)}</div>
            <div>{trainingError.length}</div>
          </div>
        </div>
        <div className="flex justify-center mt-2 text-sm text-gray-500">Época</div>
      </div>
    )
  }

  // Modificar el estilo del contenedor para asegurar que tenga scroll adecuado
  return (
    <div className="bg-white border-t border-gray-200 p-4 h-[500px] overflow-y-auto">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold">Pasos Matemáticos</h2>
        <Button variant="ghost" size="sm" onClick={clearMathSteps}>
          <X className="h-4 w-4 mr-2" />
          Cerrar
        </Button>
      </div>

      <Separator className="my-4" />

      <Tabs defaultValue="sequence">
        <TabsList className="mb-4 sticky top-0 bg-white z-10">
          <TabsTrigger value="sequence">Secuencia de Entrenamiento</TabsTrigger>
          <TabsTrigger value="forward">Propagación Hacia Adelante</TabsTrigger>
          <TabsTrigger value="error">Gráfico de Error</TabsTrigger>
        </TabsList>

        <TabsContent value="sequence" className="mt-0">
          {renderTrainingSequence()}
        </TabsContent>

        <TabsContent value="forward" className="mt-0">
          {renderForwardPropagation()}
        </TabsContent>

        <TabsContent value="error" className="mt-0">
          {renderErrorGraph()}
        </TabsContent>
      </Tabs>
    </div>
  )
}
