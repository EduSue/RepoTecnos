"use client"

import { useNetwork, type ActivationFunction } from "./network-context"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import { PlusCircle, Layers, Download, RefreshCw, Shuffle, Brain, X } from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useState, useEffect } from "react"
import { Progress } from "@/components/ui/progress"

export default function ConfigPanel() {
  const {
    network,
    addLayer,
    removeLayer,
    addNeuron,
    updateLayer,
    exportNetwork: exportNetworkContext,
    setInputNeurons,
    setOutputNeurons,
    updateNeuron,
    resetNetwork,
    randomizeNetwork,
    runForwardPropagation,
    runBackpropagation,
    stopTraining,
    updateNetworkFromTraining,
  } = useNetwork()

  const [inputCount, setInputCount] = useState(network.layers.find((l) => l.type === "input")?.neurons.length || 2)
  const [outputCount, setOutputCount] = useState(network.layers.find((l) => l.type === "output")?.neurons.length || 1)

  // Nuevos estados para los campos adicionales
  const [expectedOutput, setExpectedOutput] = useState<number[]>([0.5])
  const [epochs, setEpochs] = useState(100)
  const [learningRate, setLearningRate] = useState(0.01)

  // Add this constant for activation functions
  const activationFunctions: { value: ActivationFunction; label: string }[] = [
    { value: "sigmoid", label: "Sigmoid" },
    { value: "relu", label: "ReLU" },
    { value: "tanh", label: "Tanh" },
    { value: "linear", label: "Linear" },
    { value: "leakyRelu", label: "Leaky ReLU" },
    { value: "swish", label: "Swish" },
  ]

  // Update input/output counts when network changes
  useEffect(() => {
    const inputLayer = network.layers.find((l) => l.type === "input")
    const outputLayer = network.layers.find((l) => l.type === "output")

    if (inputLayer) {
      setInputCount(inputLayer.neurons.length)
    }

    if (outputLayer) {
      setOutputCount(outputLayer.neurons.length)
      // Solo inicializar el array de salidas esperadas si su longitud no coincide con la capa de salida
      if (expectedOutput.length !== outputLayer.neurons.length) {
        setExpectedOutput(Array(outputLayer.neurons.length).fill(0.5))
      }
    }
  }, [network])

  const handleInputCountChange = (value: string) => {
    const count = Number.parseInt(value)
    if (!isNaN(count) && count > 0 && count <= 10) {
      setInputCount(count)
      setInputNeurons(count)
    }
  }

  const handleOutputCountChange = (value: string) => {
    const count = Number.parseInt(value)
    if (!isNaN(count) && count > 0 && count <= 10) {
      setOutputCount(count)
      setOutputNeurons(count)
      // Actualizar el array de salidas esperadas con el nuevo tamaño
      setExpectedOutput(Array(count).fill(0.5))
    }
  }

  const handleInputValueChange = (neuronId: string, value: string) => {
    const newValue = Number.parseFloat(value) || 0

    // Buscar la neurona específica
    const neuron = network.layers.find((l) => l.type === "input")?.neurons.find((n) => n.id === neuronId)

    if (neuron) {
      // Actualizar el valor de activación de la neurona
      updateNeuron({
        ...neuron,
        activationValue: newValue,
      })
    }
  }

  // Función para manejar cambios en los valores esperados de salida
  const handleExpectedOutputChange = (index: number, value: string) => {
    const newValue = Number.parseFloat(value) || 0
    const newExpectedOutput = [...expectedOutput]
    newExpectedOutput[index] = newValue
    setExpectedOutput(newExpectedOutput)
    console.log("Valor esperado actualizado:", newExpectedOutput)
  }

  // Función para limpiar toda la red
  const handleResetNetwork = () => {
    if (confirm("¿Estás seguro de que quieres reiniciar toda la red? Se perderán todos los valores.")) {
      resetNetwork()
    }
  }

  // Función para randomizar la red
  const handleRandomizeNetwork = () => {
    if (confirm("¿Estás seguro de que quieres generar valores aleatorios para toda la red?")) {
      randomizeNetwork()
    }
  }

  // Función para actualizar los valores de salida (esta función se llamaría cuando lleguen resultados de la API)
  const updateOutputValues = (results: number[]) => {
    // Actualizar los valores de activación de las neuronas de salida
    const outputLayer = network.layers.find((l) => l.type === "output")
    if (outputLayer) {
      outputLayer.neurons.forEach((neuron, index) => {
        if (index < results.length) {
          updateNeuron({
            ...neuron,
            activationValue: results[index],
          })
        }
      })
    }
  }

  // Función para procesar la respuesta de la API de entrenamiento
  const processTrainingResponse = (data: any) => {
    if (data && data.history && data.history.length > 0) {
      // Obtener la última época del historial
      const lastEpoch = data.history[data.history.length - 1]

      // Actualizar la red con los valores de la última época
      updateNetworkFromTraining(lastEpoch)

      // Actualizar los valores de salida
      if (lastEpoch.output && lastEpoch.output.length > 0) {
        updateOutputValues(lastEpoch.output[0])
      }

      alert(`Entrenamiento completado. Red actualizada con los valores de la época ${lastEpoch.epoch}.`)
    }
  }

  const handleAddNeuronToLayer = (layerId: string) => {
    addNeuron(layerId)
  }

  const handleRemoveLayer = (layerId: string) => {
    removeLayer(layerId)
  }

  const exportNetwork = () => {
    const inputLayer = network.layers.find((l) => l.type === "input")
    const outputLayer = network.layers.find((l) => l.type === "output")
    const hiddenLayers = network.layers.filter((l) => l.type === "hidden")

    if (!inputLayer || !outputLayer) return

    // Crear los valores de entrada (x) a partir de los valores de activación de las neuronas de entrada
    const inputData = {
      x: inputLayer.neurons.map((n) => n.activationValue),
    }

    // Crear array para todas las capas ocultas
    const hiddenLayersData = []

    // Crear la primera capa oculta con los pesos desde la capa de entrada
    const firstHiddenLayer = hiddenLayers[0]
    if (firstHiddenLayer) {
      // Crear matriz de pesos para la primera capa oculta
      const firstLayerWeights: number[][] = []
      inputLayer.neurons.forEach((source) => {
        const row: number[] = []
        firstHiddenLayer.neurons.forEach((target) => {
          const connection = network.connections.find((c) => c.sourceId === source.id && c.targetId === target.id)
          row.push(connection ? connection.weight : 0)
        })
        firstLayerWeights.push(row)
      })

      // Añadir la primera capa oculta al array de capas ocultas
      hiddenLayersData.push({
        w: firstLayerWeights,
        b: firstHiddenLayer.neurons.map((n) => n.bias),
        activation: firstHiddenLayer.activationFunction,
      })
    }

    // Crear capas ocultas (a partir de la segunda)
    for (let idx = 1; idx < hiddenLayers.length; idx++) {
      const layer = hiddenLayers[idx]
      const prevLayer = hiddenLayers[idx - 1]

      // Crear matriz de pesos para esta capa
      const weights: number[][] = []
      prevLayer.neurons.forEach((source) => {
        const row: number[] = []
        layer.neurons.forEach((target) => {
          const connection = network.connections.find((c) => c.sourceId === source.id && c.targetId === target.id)
          row.push(connection ? connection.weight : 0)
        })
        weights.push(row)
      })

      hiddenLayersData.push({
        w: weights,
        b: layer.neurons.map((n) => n.bias),
        activation: layer.activationFunction,
      })
    }

    // Crear capa de salida
    const lastHiddenLayer = hiddenLayers.length > 0 ? hiddenLayers[hiddenLayers.length - 1] : inputLayer
    const outputWeights: number[][] = []

    lastHiddenLayer.neurons.forEach((source) => {
      const row: number[] = []
      outputLayer.neurons.forEach((target) => {
        const connection = network.connections.find((c) => c.sourceId === source.id && c.targetId === target.id)
        row.push(connection ? connection.weight : 0)
      })
      outputWeights.push(row)
    })

    const outputData = {
      w: outputWeights,
      b: outputLayer.neurons.map((n) => n.bias),
      activation: outputLayer.activationFunction,
      y: outputLayer.neurons.map((n) => n.activationValue),
    }

    // Crear la estructura final de la red con los nuevos campos
    const networkData = {
      input: inputData,
      hidden_layers: hiddenLayersData,
      output: outputData,
      y_expected: expectedOutput,
      epochs: epochs,
      learning_rate: learningRate,
    }

    // Imprimir en la consola
    console.log(JSON.stringify(networkData, null, 2))

    // También mostrar una alerta para hacerlo más visible para el usuario
    alert("Red neuronal exportada a la consola (F12)")

    // INTEGRACIÓN CON API: Este es el objeto JSON que puedes enviar a tu API
    /*
      Ejemplo de cómo enviar este objeto a tu API:
      
      fetch('https://tu-api.com/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(networkData),
      })
      .then(response => response.json())
      .then(data => {
        // RECEPCIÓN DE RESULTADOS: Aquí es donde recibirías los resultados de tu API
        processTrainingResponse(data);
      })
      .catch(error => {
        console.error('Error al llamar a la API:', error);
      });
    */

    return networkData // Devolver el objeto para posible uso en otras funciones
  }

  // Modificar la función handleTrainNetwork para asegurar que se llame correctamente
  const handleTrainNetwork = () => {
    // Mostrar una alerta para informar al usuario sobre el proceso
    if (epochs > 100) {
      if (
        !confirm(
          `Estás a punto de iniciar un entrenamiento con ${epochs} épocas, lo que puede tardar un tiempo. ¿Deseas continuar?`,
        )
      ) {
        return
      }
    }

    // Iniciar el entrenamiento con los valores esperados, épocas y tasa de aprendizaje
    // Usamos directamente expectedOutput sin modificarlo
    runBackpropagation(expectedOutput, epochs, learningRate)
  }

  const handleStopTraining = () => {
    stopTraining()
  }

  return (
    <div className="w-full lg:w-80 bg-white border-r border-gray-200 p-2 lg:p-4 flex flex-col h-screen overflow-auto">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg lg:text-xl font-bold">Configuración</h2>
        <div className="flex gap-1 lg:gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleRandomizeNetwork}
            className="flex items-center gap-1 text-blue-500 hover:text-blue-700 text-xs lg:text-sm"
          >
            <Shuffle className="h-3 w-3" />
            <span className="hidden sm:inline">Aleatorio</span>
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleResetNetwork}
            className="flex items-center gap-1 text-red-500 hover:text-red-700 text-xs lg:text-sm"
          >
            <RefreshCw className="h-3 w-3" />
            <span className="hidden sm:inline">Limpiar</span>
          </Button>
        </div>
      </div>

      <Separator className="my-4" />

      {/* Input/Output Configuration */}
      <div className="mb-6">
        <h3 className="text-base lg:text-lg font-semibold mb-2">Estructura de la Red</h3>

        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="input-count" className="text-sm">
              Neuronas de Entrada
            </Label>
            <Input
              id="input-count"
              type="number"
              min="1"
              max="10"
              value={inputCount}
              onChange={(e) => handleInputCountChange(e.target.value)}
              className="text-sm"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="output-count" className="text-sm">
              Neuronas de Salida
            </Label>
            <Input
              id="output-count"
              type="number"
              min="1"
              max="10"
              value={outputCount}
              onChange={(e) => handleOutputCountChange(e.target.value)}
              className="text-sm"
            />
          </div>

          <Button
            variant="outline"
            onClick={addLayer}
            className="w-full flex items-center justify-center gap-2 text-sm"
          >
            <PlusCircle className="h-4 w-4" />
            Añadir Capa Oculta
          </Button>
        </div>
      </div>

      <Separator className="my-4" />

      {/* Input Values */}
      <div className="mb-6">
        <h3 className="text-base lg:text-lg font-semibold mb-2">Valores de Entrada (X)</h3>
        <div className="space-y-3">
          {network.layers
            .find((l) => l.type === "input")
            ?.neurons.map((neuron, index) => (
              <div key={neuron.id} className="space-y-1">
                <Label htmlFor={`input-value-${index}`} className="text-sm">
                  {neuron.label}
                </Label>
                <Input
                  id={`input-value-${index}`}
                  type="number"
                  step="0.1"
                  value={neuron.activationValue}
                  onChange={(e) => handleInputValueChange(neuron.id, e.target.value)}
                  className="text-sm"
                />
              </div>
            ))}
        </div>
      </div>

      <Separator className="my-4" />

      {/* Expected Output Values */}
      <div className="mb-6">
        <h3 className="text-base lg:text-lg font-semibold mb-2">Valores Esperados (Y)</h3>
        <div className="space-y-3">
          {expectedOutput.map((value, index) => (
            <div key={`expected-${index}`} className="space-y-1">
              <Label htmlFor={`expected-value-${index}`} className="text-sm">
                Y{index + 1} Esperado
              </Label>
              <Input
                id={`expected-value-${index}`}
                type="number"
                step="0.1"
                value={value}
                onChange={(e) => handleExpectedOutputChange(index, e.target.value)}
                className="text-sm"
              />
            </div>
          ))}
        </div>
      </div>

      {/* Training Parameters */}
      <div className="mb-6">
        <h3 className="text-base lg:text-lg font-semibold mb-2">Parámetros de Entrenamiento</h3>
        <div className="space-y-3">
          <div className="space-y-1">
            <Label htmlFor="epochs" className="text-sm">
              Épocas
            </Label>
            <Input
              id="epochs"
              type="number"
              min="1"
              step="1"
              value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value) || 100)}
              className="text-sm"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="learning-rate" className="text-sm">
              Tasa de Aprendizaje
            </Label>
            <Input
              id="learning-rate"
              type="number"
              min="0.001"
              max="1"
              step="0.001"
              value={learningRate}
              onChange={(e) => setLearningRate(Number(e.target.value) || 0.01)}
              className="text-sm"
            />
          </div>
        </div>
      </div>

      <Separator className="my-4" />

      {/* Output Values (Results) */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-base lg:text-lg font-semibold">Resultados</h3>
          <div className="flex gap-1 lg:gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleTrainNetwork}
              className="text-xs flex items-center gap-1"
              disabled={network.trainingInProgress}
            >
              <Brain className="h-3 w-3" />
              <span className="hidden sm:inline">Entrenar</span>
            </Button>
            {network.trainingInProgress && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleStopTraining}
                className="text-xs flex items-center gap-1 text-red-500"
              >
                <X className="h-3 w-3" />
                <span className="hidden sm:inline">Detener</span>
              </Button>
            )}
          </div>
        </div>
        <div className="space-y-3 bg-gray-50 p-3 rounded-md">
          {network.layers
            .find((l) => l.type === "output")
            ?.neurons.map((neuron, index) => (
              <div key={neuron.id} className="space-y-1">
                <div className="flex justify-between items-center">
                  <Label htmlFor={`output-value-${index}`} className="text-sm">
                    {neuron.label}
                  </Label>
                  <span className="text-xs text-gray-500">Resultado</span>
                </div>
                <Input
                  id={`output-value-${index}`}
                  type="number"
                  step="0.0001"
                  value={neuron.activationValue}
                  readOnly
                  className={`bg-white opacity-90 text-sm ${network.trainingComplete ? "border-green-500 font-medium" : ""}`}
                />
                {network.trainingComplete && index < expectedOutput.length && (
                  <div className="flex justify-between text-xs mt-1">
                    <span className="text-blue-600">Esperado: {expectedOutput[index].toFixed(4)}</span>
                    <span className="text-green-600">
                      Error: {(0.5 * Math.pow(expectedOutput[index] - neuron.activationValue, 2)).toFixed(6)}
                    </span>
                  </div>
                )}
              </div>
            ))}

          {network.trainingInProgress && (
            <div className="mt-4 space-y-2">
              <div className="flex justify-between text-xs">
                <span>Entrenando...</span>
                <span>
                  Época {network.currentEpoch} de {network.totalEpochs}
                </span>
              </div>
              <Progress value={(network.currentEpoch / network.totalEpochs) * 100} />
              <div className="text-xs text-gray-600 mt-1">
                La red está ajustando los pesos y sesgos para minimizar el error.
              </div>
            </div>
          )}

          {network.trainingComplete && (
            <div className="mt-4 space-y-2 bg-green-50 p-3 rounded-md border border-green-200">
              <div className="text-sm font-medium text-green-700">Entrenamiento completado</div>
              <div className="flex justify-between text-xs text-green-600">
                <span>Épocas: {network.totalEpochs}</span>
                <span>Tasa: {learningRate}</span>
              </div>
              <div className="text-xs text-green-600">
                Error final:{" "}
                {network.trainingError.length > 0
                  ? network.trainingError[network.trainingError.length - 1].toFixed(6)
                  : "N/A"}
              </div>
            </div>
          )}

          {!network.trainingInProgress && !network.trainingComplete && (
            <p className="text-xs text-gray-500 mt-2">
              Haz clic en "Entrenar" para ejecutar la retropropagación con los valores esperados configurados.
            </p>
          )}
        </div>
      </div>

      <Separator className="my-4" />

      {/* Network Structure */}
      <div className="mb-6">
        <h3 className="text-base lg:text-lg font-semibold mb-2">Capas</h3>

        <div className="space-y-2 mt-4">
          {network.layers.map((layer) => (
            <div key={layer.id} className="p-2 bg-gray-50 rounded-md">
              <div className="flex items-center">
                <Layers className="h-4 w-4 mr-2 text-gray-500" />
                <span className="font-medium text-sm">
                  {layer.type === "input"
                    ? "Entrada"
                    : layer.type === "output"
                      ? "Salida"
                      : `Oculta ${layer.id.split("-").pop()}`}
                </span>
                <span className="ml-auto text-xs text-gray-500">{layer.neurons.length}</span>
              </div>

              {/* Mostrar la función de activación para capas ocultas y de salida */}
              {(layer.type === "hidden" || layer.type === "output") && (
                <div className="mt-2">
                  <Label htmlFor={`layer-${layer.id}-activation`} className="text-xs">
                    Función de Activación
                  </Label>
                  <Select
                    value={layer.activationFunction}
                    onValueChange={(value: ActivationFunction) => {
                      updateLayer({
                        ...layer,
                        activationFunction: value as ActivationFunction,
                      })
                    }}
                  >
                    <SelectTrigger id={`layer-${layer.id}-activation`} className="h-7 text-xs">
                      <SelectValue placeholder="Seleccionar función" />
                    </SelectTrigger>
                    <SelectContent>
                      {activationFunctions.map((fn) => (
                        <SelectItem key={fn.value} value={fn.value}>
                          {fn.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}

              <div className="flex mt-2 gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="flex-1 text-xs"
                  onClick={() => handleAddNeuronToLayer(layer.id)}
                >
                  Añadir
                </Button>

                {layer.type === "hidden" && (
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1 text-xs text-red-500 hover:text-red-700"
                    onClick={() => handleRemoveLayer(layer.id)}
                  >
                    Eliminar
                  </Button>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      <Separator className="my-4" />

      {/* Export Button */}
      <div className="mt-auto space-y-2">
        <Button onClick={exportNetwork} className="w-full flex items-center justify-center gap-2 text-sm">
          <Download className="h-4 w-4" />
          Exportar Red
        </Button>
        <p className="text-xs text-gray-500 mt-2 text-center">La estructura se exportará a la consola (F12)</p>
      </div>
    </div>
  )
}
