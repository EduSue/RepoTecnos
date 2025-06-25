"use client"

import type React from "react"
import { createContext, useContext, useState } from "react"

export type ActivationFunction = "sigmoid" | "relu" | "tanh" | "linear" | "leakyRelu" | "swish"

// Añadir nuevos tipos para los pasos matemáticos
export interface MathStep {
  description: string
  formula: string
  result: string | number
  substeps?: MathStep[]
}

export interface ForwardPropagationSteps {
  layerSteps: {
    layerId: string
    layerType: string
    neurons: {
      neuronId: string
      steps: MathStep[]
    }[]
  }[]
}

// Actualizar la interfaz BackpropagationSteps para incluir los pasos de propagación hacia adelante
export interface BackpropagationSteps {
  epochSteps: {
    epoch: number
    error: number
    outputGradients: MathStep[]
    hiddenGradients: MathStep[]
    weightUpdates: MathStep[]
    biasUpdates: MathStep[]
    forwardPropagationSteps?: ForwardPropagationSteps
  }[]
}

export interface Neuron {
  id: string
  layerId: string
  label: string
  bias: number
  activationValue: number
  activationFunction: ActivationFunction
  x?: number
  y?: number
}

export interface Connection {
  id: string
  sourceId: string
  targetId: string
  weight: number
}

export interface Layer {
  id: string
  type: "input" | "hidden" | "output"
  neurons: Neuron[]
  activationFunction: ActivationFunction
  index: number
}

// Actualizar la interfaz NetworkState para incluir los pasos matemáticos
export interface NetworkState {
  layers: Layer[]
  connections: Connection[]
  selectedNeuron: Neuron | null
  selectedConnection: Connection | null
  selectedLayer: Layer | null
  forwardPropagationSteps: ForwardPropagationSteps | null
  backpropagationSteps: BackpropagationSteps | null
  currentEpoch: number
  totalEpochs: number
  trainingInProgress: boolean
  trainingComplete: boolean
  trainingError: number[]
}

// Actualizar la interfaz NetworkContextType para incluir las nuevas funciones
interface NetworkContextType {
  network: NetworkState
  addLayer: () => void
  removeLayer: (layerId: string) => void
  addNeuron: (layerId: string) => void
  removeNeuron: (neuronId: string) => void
  updateNeuron: (neuron: Neuron) => void
  updateConnection: (connection: Connection) => void
  updateLayer: (layer: Layer) => void
  selectNeuron: (neuron: Neuron | null) => void
  selectConnection: (connection: Connection | null) => void
  selectLayer: (layer: Layer | null) => void
  exportNetwork: () => void
  setInputNeurons: (count: number) => void
  setOutputNeurons: (count: number) => void
  resetNetwork: () => void
  randomizeNetwork: () => void
  runForwardPropagation: () => void
  runBackpropagation: (expectedOutputs: number[], epochs: number, learningRate: number) => void
  stopTraining: () => void
  clearMathSteps: () => void
}

// Estado inicial de la red para poder reiniciarla
const initialNetworkState: NetworkState = {
  layers: [
    {
      id: "input-layer",
      type: "input",
      neurons: [
        {
          id: "input-1",
          layerId: "input-layer",
          label: "X1",
          bias: 0,
          activationValue: 0,
          activationFunction: "linear",
        },
        {
          id: "input-2",
          layerId: "input-layer",
          label: "X2",
          bias: 0,
          activationValue: 0,
          activationFunction: "linear",
        },
      ],
      activationFunction: "linear",
      index: 0,
    },
    {
      id: "hidden-layer-1",
      type: "hidden",
      neurons: [
        {
          id: "hidden-1-1",
          layerId: "hidden-layer-1",
          label: "H11",
          bias: 0,
          activationValue: 0,
          activationFunction: "sigmoid",
        },
        {
          id: "hidden-1-2",
          layerId: "hidden-layer-1",
          label: "H12",
          bias: 0,
          activationValue: 0,
          activationFunction: "sigmoid",
        },
        {
          id: "hidden-1-3",
          layerId: "hidden-layer-1",
          label: "H13",
          bias: 0,
          activationValue: 0,
          activationFunction: "sigmoid",
        },
      ],
      activationFunction: "sigmoid",
      index: 1,
    },
    {
      id: "output-layer",
      type: "output",
      neurons: [
        {
          id: "output-1",
          layerId: "output-layer",
          label: "Y1",
          bias: 0,
          activationValue: 0,
          activationFunction: "sigmoid",
        },
      ],
      activationFunction: "sigmoid",
      index: 2,
    },
  ],
  connections: [
    { id: "c1", sourceId: "input-1", targetId: "hidden-1-1", weight: 0 },
    { id: "c2", sourceId: "input-1", targetId: "hidden-1-2", weight: 0 },
    { id: "c3", sourceId: "input-1", targetId: "hidden-1-3", weight: 0 },
    { id: "c4", sourceId: "input-2", targetId: "hidden-1-1", weight: 0 },
    { id: "c5", sourceId: "input-2", targetId: "hidden-1-2", weight: 0 },
    { id: "c6", sourceId: "input-2", targetId: "hidden-1-3", weight: 0 },
    { id: "c7", sourceId: "hidden-1-1", targetId: "output-1", weight: 0 },
    { id: "c8", sourceId: "hidden-1-2", targetId: "output-1", weight: 0 },
    { id: "c9", sourceId: "hidden-1-3", targetId: "output-1", weight: 0 },
  ],
  selectedNeuron: null,
  selectedConnection: null,
  selectedLayer: null,
  forwardPropagationSteps: null,
  backpropagationSteps: null,
  currentEpoch: 0,
  totalEpochs: 0,
  trainingInProgress: false,
  trainingComplete: false,
  trainingError: [],
}

const NetworkContext = createContext<NetworkContextType | undefined>(undefined)

// Añadir las implementaciones de las funciones de activación y sus derivadas
const activationFunctions = {
  sigmoid: (x: number): number => 1 / (1 + Math.exp(-x)),
  relu: (x: number): number => Math.max(0, x),
  tanh: (x: number): number => Math.tanh(x),
  linear: (x: number): number => x,
  leakyRelu: (x: number): number => (x > 0 ? x : 0.01 * x),
  swish: (x: number): number => x / (1 + Math.exp(-x)),
}

const activationDerivatives = {
  sigmoid: (x: number): number => {
    const sigX = activationFunctions.sigmoid(x)
    return sigX * (1 - sigX)
  },
  relu: (x: number): number => (x > 0 ? 1 : 0),
  tanh: (x: number): number => 1 - Math.pow(Math.tanh(x), 2),
  linear: (): number => 1,
  leakyRelu: (x: number): number => (x > 0 ? 1 : 0.01),
  swish: (x: number): number => {
    const sigX = activationFunctions.sigmoid(x)
    return sigX + x * sigX * (1 - sigX)
  },
}

// Añadir estas funciones dentro del NetworkProvider
export function NetworkProvider({ children }: { children: React.ReactNode }) {
  // Inicializar la red con el estado inicial
  const [network, setNetwork] = useState<NetworkState>({ ...initialNetworkState })

  // Función para reiniciar la red a su estado inicial
  const resetNetwork = () => {
    // Crear una copia profunda del estado inicial
    const resetState = JSON.parse(JSON.stringify(initialNetworkState))
    setNetwork(resetState)
  }

  // Función para generar un valor aleatorio entre min y max
  const getRandomValue = (min: number, max: number): number => {
    return Math.random() * (max - min) + min
  }

  // Función para randomizar toda la red
  const randomizeNetwork = () => {
    // Crear una copia profunda del estado actual
    const networkCopy = JSON.parse(JSON.stringify(network))

    // Randomizar los pesos de las conexiones
    networkCopy.connections = networkCopy.connections.map((connection: Connection) => ({
      ...connection,
      weight: getRandomValue(-1, 1),
    }))

    // Randomizar los sesgos de las neuronas (excepto en la capa de entrada)
    networkCopy.layers = networkCopy.layers.map((layer: Layer) => {
      if (layer.type === "input") {
        return layer
      }

      return {
        ...layer,
        neurons: layer.neurons.map((neuron: Neuron) => ({
          ...neuron,
          bias: getRandomValue(-1, 1),
        })),
      }
    })

    // Actualizar el estado de la red
    setNetwork(networkCopy)
  }

  // Función para actualizar la red con los datos de entrenamiento
  const updateNetworkFromTraining = (trainingData: any) => {
    if (!trainingData) return

    // Crear una copia profunda del estado actual
    const networkCopy = JSON.parse(JSON.stringify(network))

    // Obtener las capas ocultas y de salida
    const hiddenLayers = networkCopy.layers.filter((l: Layer) => l.type === "hidden")
    const outputLayer = networkCopy.layers.find((l: Layer) => l.type === "output")

    // Actualizar los pesos y sesgos de las capas ocultas
    if (trainingData.weights_hidden && trainingData.biases_hidden) {
      for (let i = 0; i < hiddenLayers.length && i < trainingData.weights_hidden.length; i++) {
        const layerWeights = trainingData.weights_hidden[i]
        const layerBiases = trainingData.biases_hidden[i]

        // Actualizar los sesgos de las neuronas en esta capa oculta
        if (layerBiases && layerBiases.length > 0) {
          for (let j = 0; j < hiddenLayers[i].neurons.length && j < layerBiases[0].length; j++) {
            hiddenLayers[i].neurons[j].bias = layerBiases[0][j]
          }
        }

        // Actualizar los pesos de las conexiones a esta capa oculta
        if (layerWeights && layerWeights.length > 0) {
          const prevLayer = i === 0 ? networkCopy.layers.find((l: Layer) => l.type === "input") : hiddenLayers[i - 1]

          if (prevLayer) {
            for (
              let sourceIdx = 0;
              sourceIdx < prevLayer.neurons.length && sourceIdx < layerWeights.length;
              sourceIdx++
            ) {
              for (
                let targetIdx = 0;
                targetIdx < hiddenLayers[i].neurons.length && targetIdx < layerWeights[sourceIdx].length;
                targetIdx++
              ) {
                const sourceId = prevLayer.neurons[sourceIdx].id
                const targetId = hiddenLayers[i].neurons[targetIdx].id

                // Encontrar y actualizar la conexión
                const connectionIndex = networkCopy.connections.findIndex(
                  (c: Connection) => c.sourceId === sourceId && c.targetId === targetId,
                )

                if (connectionIndex !== -1) {
                  networkCopy.connections[connectionIndex].weight = layerWeights[sourceIdx][targetIdx]
                }
              }
            }
          }
        }
      }
    }

    // Actualizar los pesos y sesgos de la capa de salida
    if (outputLayer && trainingData.weights_output && trainingData.biases_output) {
      // Actualizar los sesgos de las neuronas de salida
      for (let i = 0; i < outputLayer.neurons.length && i < trainingData.biases_output.length; i++) {
        outputLayer.neurons[i].bias = trainingData.biases_output[i]
      }

      // Actualizar los pesos de las conexiones a la capa de salida
      const lastHiddenLayer =
        hiddenLayers.length > 0
          ? hiddenLayers[hiddenLayers.length - 1]
          : networkCopy.layers.find((l: Layer) => l.type === "input")

      if (lastHiddenLayer) {
        for (
          let sourceIdx = 0;
          sourceIdx < lastHiddenLayer.neurons.length && sourceIdx < trainingData.weights_output.length;
          sourceIdx++
        ) {
          for (
            let targetIdx = 0;
            targetIdx < outputLayer.neurons.length && targetIdx < trainingData.weights_output[sourceIdx].length;
            targetIdx++
          ) {
            const sourceId = lastHiddenLayer.neurons[sourceIdx].id
            const targetId = outputLayer.neurons[targetIdx].id

            // Encontrar y actualizar la conexión
            const connectionIndex = networkCopy.connections.findIndex(
              (c: Connection) => c.sourceId === sourceId && c.targetId === targetId,
            )

            if (connectionIndex !== -1) {
              networkCopy.connections[connectionIndex].weight = trainingData.weights_output[sourceIdx][targetIdx]
            }
          }
        }
      }
    }

    // Actualizar el estado de la red
    setNetwork(networkCopy)
  }

  // Modificar la función setInputNeurons para usar pesos cero
  const setInputNeurons = (count: number) => {
    const inputLayer = network.layers.find((l) => l.type === "input")
    if (!inputLayer) return

    const currentCount = inputLayer.neurons.length

    if (count === currentCount) return

    if (count > currentCount) {
      // Add neurons
      const newNeurons: Neuron[] = []
      const newConnections: Connection[] = []

      for (let i = currentCount + 1; i <= count; i++) {
        const newNeuron: Neuron = {
          id: `input-${i}`,
          layerId: inputLayer.id,
          label: `X${i}`,
          bias: 0,
          activationValue: 0,
          activationFunction: "linear",
        }

        newNeurons.push(newNeuron)

        // Connect to all neurons in the next layer
        const nextLayer = network.layers[1]
        if (nextLayer) {
          nextLayer.neurons.forEach((target) => {
            newConnections.push({
              id: `c-${newNeuron.id}-${target.id}`,
              sourceId: newNeuron.id,
              targetId: target.id,
              weight: 0, // Inicializar con peso cero
            })
          })
        }
      }

      const updatedLayers = network.layers.map((layer) => {
        if (layer.id === inputLayer.id) {
          return {
            ...layer,
            neurons: [...layer.neurons, ...newNeurons],
          }
        }
        return layer
      })

      setNetwork({
        ...network,
        layers: updatedLayers,
        connections: [...network.connections, ...newConnections],
      })
    } else {
      // Remove neurons
      const neuronsToKeep = inputLayer.neurons.slice(0, count)
      const neuronsToRemove = inputLayer.neurons.slice(count).map((n) => n.id)

      // Remove connections involving these neurons
      const filteredConnections = network.connections.filter(
        (c) => !neuronsToRemove.includes(c.sourceId) && !neuronsToRemove.includes(c.targetId),
      )

      const updatedLayers = network.layers.map((layer) => {
        if (layer.id === inputLayer.id) {
          return {
            ...layer,
            neurons: neuronsToKeep,
          }
        }
        return layer
      })

      setNetwork({
        ...network,
        layers: updatedLayers,
        connections: filteredConnections,
        selectedNeuron: null,
        selectedConnection: null,
      })
    }
  }

  // Modificar la función setOutputNeurons para usar pesos cero
  const setOutputNeurons = (count: number) => {
    const outputLayer = network.layers.find((l) => l.type === "output")
    if (!outputLayer) return

    const currentCount = outputLayer.neurons.length

    if (count === currentCount) return

    if (count > currentCount) {
      // Add neurons
      const newNeurons: Neuron[] = []
      const newConnections: Connection[] = []

      for (let i = currentCount + 1; i <= count; i++) {
        const newNeuron: Neuron = {
          id: `output-${i}`,
          layerId: outputLayer.id,
          label: `Y${i}`,
          bias: 0,
          activationValue: 0,
          activationFunction: outputLayer.activationFunction,
        }

        newNeurons.push(newNeuron)

        // Connect from all neurons in the previous layer
        const prevLayer = network.layers[network.layers.length - 2]
        if (prevLayer) {
          prevLayer.neurons.forEach((source) => {
            newConnections.push({
              id: `c-${source.id}-${newNeuron.id}`,
              sourceId: source.id,
              targetId: newNeuron.id,
              weight: 0, // Inicializar con peso cero
            })
          })
        }
      }

      const updatedLayers = network.layers.map((layer) => {
        if (layer.id === outputLayer.id) {
          return {
            ...layer,
            neurons: [...layer.neurons, ...newNeurons],
          }
        }
        return layer
      })

      setNetwork({
        ...network,
        layers: updatedLayers,
        connections: [...network.connections, ...newConnections],
      })
    } else {
      // Remove neurons
      const neuronsToKeep = outputLayer.neurons.slice(0, count)
      const neuronsToRemove = outputLayer.neurons.slice(count).map((n) => n.id)

      // Remove connections involving these neurons
      const filteredConnections = network.connections.filter(
        (c) => !neuronsToRemove.includes(c.sourceId) && !neuronsToRemove.includes(c.targetId),
      )

      const updatedLayers = network.layers.map((layer) => {
        if (layer.id === outputLayer.id) {
          return {
            ...layer,
            neurons: neuronsToKeep,
          }
        }
        return layer
      })

      setNetwork({
        ...network,
        layers: updatedLayers,
        connections: filteredConnections,
        selectedNeuron: null,
        selectedConnection: null,
      })
    }
  }

  // Modificar la función addLayer para usar pesos cero
  const addLayer = () => {
    const newLayerId = `hidden-layer-${network.layers.filter((l) => l.type === "hidden").length + 1}`
    const newLayer: Layer = {
      id: newLayerId,
      type: "hidden",
      neurons: [
        {
          id: `${newLayerId}-1`,
          layerId: newLayerId,
          label: `H${network.layers.length - 1}1`,
          bias: 0,
          activationValue: 0,
          activationFunction: "sigmoid",
        },
        {
          id: `${newLayerId}-2`,
          layerId: newLayerId,
          label: `H${network.layers.length - 1}2`,
          bias: 0,
          activationValue: 0,
          activationFunction: "sigmoid",
        },
      ],
      activationFunction: "sigmoid",
      index: network.layers.length - 1,
    }

    // Update the output layer index
    const outputLayer = network.layers.find((l) => l.type === "output")
    if (outputLayer) {
      outputLayer.index = network.layers.length
    }

    // Create connections from previous layer to new layer
    const prevLayer = network.layers[network.layers.length - 2]
    const newConnections: Connection[] = []

    prevLayer.neurons.forEach((source) => {
      newLayer.neurons.forEach((target) => {
        newConnections.push({
          id: `c-${source.id}-${target.id}`,
          sourceId: source.id,
          targetId: target.id,
          weight: 0, // Inicializar con peso cero
        })
      })
    })

    // Remove connections from previous layer to output layer
    const filteredConnections = network.connections.filter((c) => {
      const sourceNeuron = prevLayer.neurons.find((n) => n.id === c.sourceId)
      const targetNeuron = outputLayer?.neurons.find((n) => n.id === c.targetId)
      return !(sourceNeuron && targetNeuron)
    })

    // Create connections from new layer to output layer
    const outputNeurons = network.layers.find((l) => l.type === "output")?.neurons || []
    newLayer.neurons.forEach((source) => {
      outputNeurons.forEach((target) => {
        newConnections.push({
          id: `c-${source.id}-${target.id}`,
          sourceId: source.id,
          targetId: target.id,
          weight: 0, // Inicializar con peso cero
        })
      })
    })

    // Insert the new layer before the output layer
    const newLayers = [...network.layers.slice(0, -1), newLayer, network.layers[network.layers.length - 1]]

    // Update hidden layer indices and labels
    const updatedLayers = newLayers.map((layer, idx) => {
      if (layer.type === "hidden") {
        const hiddenIdx = newLayers.filter((l, i) => l.type === "hidden" && i <= idx).length

        return {
          ...layer,
          id: `hidden-layer-${hiddenIdx}`,
          neurons: layer.neurons.map((neuron, neuronIdx) => ({
            ...neuron,
            label: `H${hiddenIdx}${neuronIdx + 1}`,
          })),
        }
      }
      return layer
    })

    setNetwork({
      ...network,
      layers: updatedLayers,
      connections: [...filteredConnections, ...newConnections],
    })
  }

  // Modificar la función removeLayer para usar pesos cero
  const removeLayer = (layerId: string) => {
    const layerToRemove = network.layers.find((l) => l.id === layerId)
    if (!layerToRemove || layerToRemove.type === "input" || layerToRemove.type === "output") {
      return // Cannot remove input or output layers
    }

    // Get neurons from the layer to remove
    const neuronsToRemove = layerToRemove.neurons.map((n) => n.id)

    // Remove connections involving these neurons
    const filteredConnections = network.connections.filter(
      (c) => !neuronsToRemove.includes(c.sourceId) && !neuronsToRemove.includes(c.targetId),
    )

    // Create new connections between the layers before and after the removed layer
    const layerIndex = network.layers.findIndex((l) => l.id === layerId)
    const prevLayer = network.layers[layerIndex - 1]
    const nextLayer = network.layers[layerIndex + 1]

    const newConnections: Connection[] = []
    if (prevLayer && nextLayer) {
      prevLayer.neurons.forEach((source) => {
        nextLayer.neurons.forEach((target) => {
          newConnections.push({
            id: `c-${source.id}-${target.id}`,
            sourceId: source.id,
            targetId: target.id,
            weight: 0, // Inicializar con peso cero
          })
        })
      })
    }

    // Filter out the layer to remove
    const filteredLayers = network.layers.filter((l) => l.id !== layerId)

    // Update hidden layer indices and labels
    const updatedLayers = filteredLayers.map((layer, idx) => {
      if (layer.type === "hidden") {
        const hiddenIdx = filteredLayers.filter((l, i) => l.type === "hidden" && i <= idx).length

        return {
          ...layer,
          id: `hidden-layer-${hiddenIdx}`,
          index: idx,
          neurons: layer.neurons.map((neuron, neuronIdx) => ({
            ...neuron,
            layerId: `hidden-layer-${hiddenIdx}`,
            label: `H${hiddenIdx}${neuronIdx + 1}`,
          })),
        }
      }
      return {
        ...layer,
        index: idx,
      }
    })

    setNetwork({
      ...network,
      layers: updatedLayers,
      connections: [...filteredConnections, ...newConnections],
      selectedLayer: null,
      selectedNeuron: null,
      selectedConnection: null,
    })
  }

  // Modificar la función addNeuron para usar pesos cero
  const addNeuron = (layerId: string) => {
    const layerIndex = network.layers.findIndex((l) => l.id === layerId)
    if (layerIndex === -1) return

    const layer = network.layers[layerIndex]
    const neuronCount = layer.neurons.length + 1

    // Create proper label based on layer type and position
    let newLabel = ""
    if (layer.type === "input") {
      newLabel = `X${neuronCount}`
    } else if (layer.type === "output") {
      newLabel = `Y${neuronCount}`
    } else {
      // For hidden layers, use the layer index in the network
      const hiddenLayerIndex = network.layers.filter((l, i) => l.type === "hidden" && i <= layerIndex).length
      newLabel = `H${hiddenLayerIndex}${neuronCount}`
    }

    const newNeuron: Neuron = {
      id: `${layerId}-${neuronCount}`,
      layerId,
      label: newLabel,
      bias: 0,
      activationValue: 0,
      activationFunction: layer.activationFunction,
    }

    // Create connections to/from the new neuron
    const newConnections: Connection[] = []

    if (layerIndex > 0) {
      // Connect from previous layer to this new neuron
      const prevLayer = network.layers[layerIndex - 1]
      prevLayer.neurons.forEach((source) => {
        newConnections.push({
          id: `c-${source.id}-${newNeuron.id}`,
          sourceId: source.id,
          targetId: newNeuron.id,
          weight: 0, // Inicializar con peso cero
        })
      })
    }

    if (layerIndex < network.layers.length - 1) {
      // Connect from this new neuron to next layer
      const nextLayer = network.layers[layerIndex + 1]
      nextLayer.neurons.forEach((target) => {
        newConnections.push({
          id: `c-${newNeuron.id}-${target.id}`,
          sourceId: newNeuron.id,
          targetId: target.id,
          weight: 0, // Inicializar con peso cero
        })
      })
    }

    // Update the layer with the new neuron
    const updatedLayers = network.layers.map((l) => {
      if (l.id === layerId) {
        return {
          ...l,
          neurons: [...l.neurons, newNeuron],
        }
      }
      return l
    })

    setNetwork({
      ...network,
      layers: updatedLayers,
      connections: [...network.connections, ...newConnections],
    })
  }

  const removeNeuron = (neuronId: string) => {
    const neuronToRemove = network.layers.flatMap((l) => l.neurons).find((n) => n.id === neuronId)

    if (!neuronToRemove) return

    const layer = network.layers.find((l) => l.id === neuronToRemove.layerId)
    if (!layer || layer.neurons.length <= 1) {
      return // Don't remove the last neuron in a layer
    }

    // Remove connections involving this neuron
    const filteredConnections = network.connections.filter((c) => c.sourceId !== neuronId && c.targetId !== neuronId)

    // Update the layer by removing the neuron and relabeling remaining neurons
    const updatedLayers = network.layers.map((l) => {
      if (l.id === neuronToRemove.layerId) {
        // Remove the neuron
        const filteredNeurons = l.neurons.filter((n) => n.id !== neuronId)

        // Relabel the remaining neurons to maintain sequential order
        const relabeledNeurons = filteredNeurons.map((neuron, index) => {
          // Create new label based on layer type and index
          let newLabel = ""
          if (l.type === "input") {
            newLabel = `X${index + 1}`
          } else if (l.type === "output") {
            newLabel = `Y${index + 1}`
          } else {
            // For hidden layers, use the layer index in the network
            const layerIndex = network.layers.findIndex((layer) => layer.id === l.id)
            const hiddenLayerIndex = network.layers.filter(
              (layer, i) => layer.type === "hidden" && i <= layerIndex,
            ).length
            newLabel = `H${hiddenLayerIndex}${index + 1}`
          }

          return {
            ...neuron,
            label: newLabel,
          }
        })

        return {
          ...l,
          neurons: relabeledNeurons,
        }
      }
      return l
    })

    setNetwork({
      ...network,
      layers: updatedLayers,
      connections: filteredConnections,
      selectedNeuron: null,
      selectedConnection: null,
    })
  }

  const updateNeuron = (updatedNeuron: Neuron) => {
    const updatedLayers = network.layers.map((layer) => {
      if (layer.id === updatedNeuron.layerId) {
        return {
          ...layer,
          neurons: layer.neurons.map((neuron) => (neuron.id === updatedNeuron.id ? updatedNeuron : neuron)),
        }
      }
      return layer
    })

    setNetwork({
      ...network,
      layers: updatedLayers,
      selectedNeuron: updatedNeuron,
    })
  }

  const updateConnection = (updatedConnection: Connection) => {
    const updatedConnections = network.connections.map((connection) =>
      connection.id === updatedConnection.id ? updatedConnection : connection,
    )

    setNetwork({
      ...network,
      connections: updatedConnections,
      selectedConnection: updatedConnection,
    })
  }

  const updateLayer = (updatedLayer: Layer) => {
    // Update the layer and propagate activation function to neurons if needed
    const updatedLayers = network.layers.map((layer) => {
      if (layer.id === updatedLayer.id) {
        // Update neurons' activation functions if the layer's function changed
        const updatedNeurons = layer.neurons.map((neuron) => ({
          ...neuron,
          activationFunction: updatedLayer.activationFunction,
        }))

        return {
          ...updatedLayer,
          neurons: updatedNeurons,
        }
      }
      return layer
    })

    setNetwork({
      ...network,
      layers: updatedLayers,
      selectedLayer: updatedLayer,
    })
  }

  const selectNeuron = (neuron: Neuron | null) => {
    setNetwork({
      ...network,
      selectedNeuron: neuron,
      selectedConnection: null,
      selectedLayer: neuron ? null : network.selectedLayer,
    })
  }

  const selectConnection = (connection: Connection | null) => {
    setNetwork({
      ...network,
      selectedConnection: connection,
      selectedNeuron: null,
      selectedLayer: connection ? null : network.selectedLayer,
    })
  }

  const selectLayer = (layer: Layer | null) => {
    setNetwork({
      ...network,
      selectedLayer: layer,
      selectedNeuron: null,
      selectedConnection: null,
    })
  }

  // Modificar la función exportNetwork para reflejar la estructura correcta
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

    // Crear la estructura final de la red
    const networkData = {
      input: inputData,
      hidden_layers: hiddenLayersData,
      output: outputData,
    }

    return networkData
  }

  // Función para limpiar los pasos matemáticos
  const clearMathSteps = () => {
    setNetwork((prev) => ({
      ...prev,
      forwardPropagationSteps: null,
      backpropagationSteps: null,
      currentEpoch: 0,
      totalEpochs: 0,
      trainingInProgress: false,
      trainingComplete: false,
      trainingError: [],
    }))
  }

  // Función para ejecutar la propagación hacia adelante
  const runForwardPropagation = () => {
    const networkCopy = JSON.parse(JSON.stringify(network))
    const layerSteps: ForwardPropagationSteps["layerSteps"] = []

    // Obtener las capas en orden
    const inputLayer = networkCopy.layers.find((l: Layer) => l.type === "input")
    const hiddenLayers = networkCopy.layers.filter((l: Layer) => l.type === "hidden")
    const outputLayer = networkCopy.layers.find((l: Layer) => l.type === "output")

    // Procesar capa de entrada (no hay cálculos, solo valores de entrada)
    const inputLayerSteps = {
      layerId: inputLayer.id,
      layerType: "input",
      neurons: inputLayer.neurons.map((neuron: Neuron) => ({
        neuronId: neuron.id,
        steps: [
          {
            description: `Valor de entrada para ${neuron.label}`,
            formula: `${neuron.label} = ${neuron.activationValue}`,
            result: neuron.activationValue,
          },
        ],
      })),
    }
    layerSteps.push(inputLayerSteps)

    // Procesar capas ocultas
    for (let i = 0; i < hiddenLayers.length; i++) {
      const currentLayer = hiddenLayers[i]
      const prevLayer = i === 0 ? inputLayer : hiddenLayers[i - 1]

      const layerStep = {
        layerId: currentLayer.id,
        layerType: "hidden",
        neurons: [] as { neuronId: string; steps: MathStep[] }[],
      }

      // Calcular la salida para cada neurona en la capa actual
      for (const neuron of currentLayer.neurons) {
        const neuronSteps: MathStep[] = []

        // Paso 1: Calcular la suma ponderada (z)
        const weightedSumStep: MathStep = {
          description: `Calcular la suma ponderada para ${neuron.label}`,
          formula: `z = Σ(entrada_i * peso_i) + sesgo`,
          result: 0,
          substeps: [],
        }

        let weightedSum = neuron.bias
        weightedSumStep.substeps!.push({
          description: "Sesgo",
          formula: `b = ${neuron.bias}`,
          result: neuron.bias,
        })

        // Sumar cada entrada ponderada
        for (const prevNeuron of prevLayer.neurons) {
          const connection = networkCopy.connections.find(
            (c: Connection) => c.sourceId === prevNeuron.id && c.targetId === neuron.id,
          )

          if (connection) {
            const weightedInput = prevNeuron.activationValue * connection.weight
            weightedSum += weightedInput

            weightedSumStep.substeps!.push({
              description: `Entrada desde ${prevNeuron.label}`,
              formula: `${prevNeuron.label} * w_${prevNeuron.label}_${neuron.label} = ${prevNeuron.activationValue} * ${connection.weight}`,
              result: weightedInput,
            })
          }
        }

        weightedSumStep.result = weightedSum
        neuronSteps.push(weightedSumStep)

        // Paso 2: Aplicar la función de activación
        const activationStep: MathStep = {
          description: `Aplicar función de activación ${neuron.activationFunction}`,
          formula: `a = ${neuron.activationFunction}(z) = ${neuron.activationFunction}(${weightedSum})`,
          result: 0,
        }

        const activationValue = activationFunctions[neuron.activationFunction](weightedSum)
        activationStep.result = activationValue
        neuronSteps.push(activationStep)

        // Actualizar el valor de activación de la neurona
        neuron.activationValue = activationValue

        layerStep.neurons.push({
          neuronId: neuron.id,
          steps: neuronSteps,
        })
      }

      layerSteps.push(layerStep)
    }

    // Procesar capa de salida
    const outputLayerStep = {
      layerId: outputLayer.id,
      layerType: "output",
      neurons: [] as { neuronId: string; steps: MathStep[] }[],
    }

    const prevLayer = hiddenLayers.length > 0 ? hiddenLayers[hiddenLayers.length - 1] : inputLayer

    // Calcular la salida para cada neurona en la capa de salida
    for (const neuron of outputLayer.neurons) {
      const neuronSteps: MathStep[] = []

      // Paso 1: Calcular la suma ponderada (z)
      const weightedSumStep: MathStep = {
        description: `Calcular la suma ponderada para ${neuron.label}`,
        formula: `z = Σ(entrada_i * peso_i) + sesgo`,
        result: 0,
        substeps: [],
      }

      let weightedSum = neuron.bias
      weightedSumStep.substeps!.push({
        description: "Sesgo",
        formula: `b = ${neuron.bias}`,
        result: neuron.bias,
      })

      // Sumar cada entrada ponderada
      for (const prevNeuron of prevLayer.neurons) {
        const connection = networkCopy.connections.find(
          (c: Connection) => c.sourceId === prevNeuron.id && c.targetId === neuron.id,
        )

        if (connection) {
          const weightedInput = prevNeuron.activationValue * connection.weight
          weightedSum += weightedInput

          weightedSumStep.substeps!.push({
            description: `Entrada desde ${prevNeuron.label}`,
            formula: `${prevNeuron.label} * w_${prevNeuron.label}_${neuron.label} = ${prevNeuron.activationValue} * ${connection.weight}`,
            result: weightedInput,
          })
        }
      }

      weightedSumStep.result = weightedSum
      neuronSteps.push(weightedSumStep)

      // Paso 2: Aplicar la función de activación
      const activationStep: MathStep = {
        description: `Aplicar función de activación ${neuron.activationFunction}`,
        formula: `a = ${neuron.activationFunction}(z) = ${neuron.activationFunction}(${weightedSum})`,
        result: 0,
      }

      const activationValue = activationFunctions[neuron.activationFunction](weightedSum)
      activationStep.result = activationValue
      neuronSteps.push(activationStep)

      // Actualizar el valor de activación de la neurona
      neuron.activationValue = activationValue

      outputLayerStep.neurons.push({
        neuronId: neuron.id,
        steps: neuronSteps,
      })
    }

    layerSteps.push(outputLayerStep)

    // Actualizar el estado de la red con los nuevos valores y pasos
    setNetwork({
      ...networkCopy,
      forwardPropagationSteps: { layerSteps },
    })
  }

  // Función para detener el entrenamiento
  const stopTraining = () => {
    setNetwork((prev) => ({
      ...prev,
      trainingInProgress: false,
    }))
  }

  // Función para ejecutar la retropropagación
  const runBackpropagationOriginal = async (expectedOutputs: number[], epochs: number, learningRate: number) => {
    // Inicializar el estado de entrenamiento
    setNetwork((prev) => ({
      ...prev,
      currentEpoch: 0,
      totalEpochs: epochs,
      trainingInProgress: true,
      trainingComplete: false,
      trainingError: [],
    }))

    const networkCopy = JSON.parse(JSON.stringify(network))
    const epochSteps: BackpropagationSteps["epochSteps"] = []

    // Obtener las capas en orden
    const inputLayer = networkCopy.layers.find((l: Layer) => l.type === "input")
    const hiddenLayers = networkCopy.layers.filter((l: Layer) => l.type === "hidden")
    const outputLayer = networkCopy.layers.find((l: Layer) => l.type === "output")

    // Ejecutar el entrenamiento para cada época
    for (let epoch = 0; epoch < epochs; epoch++) {
      // Verificar si el entrenamiento fue detenido
      if (!network.trainingInProgress) {
        break
      }

      // Ejecutar propagación hacia adelante
      runForwardPropagation()

      // Inicializar los pasos para esta época
      const epochStep = {
        epoch: epoch + 1,
        error: 0,
        outputGradients: [] as MathStep[],
        hiddenGradients: [] as MathStep[],
        weightUpdates: [] as MathStep[],
        biasUpdates: [] as MathStep[],
      }

      // Calcular el error de la época
      let totalError = 0
      for (let i = 0; i < outputLayer.neurons.length; i++) {
        const neuron = outputLayer.neurons[i]
        const expected = expectedOutputs[i] || 0
        const error = 0.5 * Math.pow(expected - neuron.activationValue, 2)
        totalError += error

        epochStep.outputGradients.push({
          description: `Error para ${neuron.label}`,
          formula: `E = 0.5 * (${expected} - ${neuron.activationValue})^2`,
          result: error,
        })
      }
      epochStep.error = totalError

      // Calcular gradientes para la capa de salida
      const outputDeltas: { [key: string]: number } = {}
      for (let i = 0; i < outputLayer.neurons.length; i++) {
        const neuron = outputLayer.neurons[i]
        const expected = expectedOutputs[i] || 0
        const output = neuron.activationValue

        // Calcular delta: derivada del error respecto a la salida * derivada de la activación
        const errorGradient = -(expected - output)

        // Encontrar la suma ponderada (z) para esta neurona
        const prevLayer = hiddenLayers.length > 0 ? hiddenLayers[hiddenLayers.length - 1] : inputLayer
        let z = neuron.bias
        for (const prevNeuron of prevLayer.neurons) {
          const connection = networkCopy.connections.find(
            (c: Connection) => c.sourceId === prevNeuron.id && c.targetId === neuron.id,
          )
          if (connection) {
            z += prevNeuron.activationValue * connection.weight
          }
        }

        const activationGradient = activationDerivatives[neuron.activationFunction](z)
        const delta = errorGradient * activationGradient
        outputDeltas[neuron.id] = delta

        epochStep.outputGradients.push({
          description: `Gradiente para ${neuron.label}`,
          formula: `δ = -(${expected} - ${output}) * ${neuron.activationFunction}'(${z})`,
          result: delta,
          substeps: [
            {
              description: "Derivada del error respecto a la salida",
              formula: `∂E/∂y = -(${expected} - ${output})`,
              result: errorGradient,
            },
            {
              description: `Derivada de la función de activación ${neuron.activationFunction}`,
              formula: `${neuron.activationFunction}'(${z})`,
              result: activationGradient,
            },
          ],
        })

        // Actualizar el sesgo de la neurona de salida
        const biasUpdate = -learningRate * delta
        neuron.bias += biasUpdate

        epochStep.biasUpdates.push({
          description: `Actualización del sesgo para ${neuron.label}`,
          formula: `Δb = -η * δ = -${learningRate} * ${delta}`,
          result: biasUpdate,
        })
      }

      // Calcular gradientes para las capas ocultas (de atrás hacia adelante)
      const hiddenDeltas: { [key: string]: number } = {}
      for (let i = hiddenLayers.length - 1; i >= 0; i--) {
        const currentLayer = hiddenLayers[i]
        const nextLayer = i === hiddenLayers.length - 1 ? outputLayer : hiddenLayers[i + 1]

        for (const neuron of currentLayer.neurons) {
          // Calcular la suma de los deltas ponderados de la capa siguiente
          let sumDeltaWeights = 0
          const deltaSumSteps: MathStep[] = []

          for (const nextNeuron of nextLayer.neurons) {
            const connection = networkCopy.connections.find(
              (c: Connection) => c.sourceId === neuron.id && c.targetId === nextNeuron.id,
            )

            if (connection) {
              const nextDelta =
                i === hiddenLayers.length - 1 ? outputDeltas[nextNeuron.id] : hiddenDeltas[nextNeuron.id]
              const deltaWeight = nextDelta * connection.weight
              sumDeltaWeights += deltaWeight

              deltaSumSteps.push({
                description: `Contribución desde ${nextNeuron.label}`,
                formula: `δ_${nextNeuron.label} * w_${neuron.label}_${nextNeuron.label} = ${nextDelta} * ${connection.weight}`,
                result: deltaWeight,
              })
            }
          }

          // Encontrar la suma ponderada (z) para esta neurona
          const prevLayer = i === 0 ? inputLayer : hiddenLayers[i - 1]
          let z = neuron.bias
          for (const prevNeuron of prevLayer.neurons) {
            const connection = networkCopy.connections.find(
              (c: Connection) => c.sourceId === prevNeuron.id && c.targetId === neuron.id,
            )
            if (connection) {
              z += prevNeuron.activationValue * connection.weight
            }
          }

          const activationGradient = activationDerivatives[neuron.activationFunction](z)
          const delta = sumDeltaWeights * activationGradient
          hiddenDeltas[neuron.id] = delta

          epochStep.hiddenGradients.push({
            description: `Gradiente para ${neuron.label}`,
            formula: `δ = (Σ δ_siguiente * w) * ${neuron.activationFunction}'(${z})`,
            result: delta,
            substeps: [
              {
                description: "Suma de deltas ponderados de la capa siguiente",
                formula: "Σ δ_siguiente * w",
                result: sumDeltaWeights,
                substeps: deltaSumSteps,
              },
              {
                description: `Derivada de la función de activación ${neuron.activationFunction}`,
                formula: `${neuron.activationFunction}'(${z})`,
                result: activationGradient,
              },
            ],
          })

          // Actualizar el sesgo de la neurona oculta
          const biasUpdate = -learningRate * delta
          neuron.bias += biasUpdate

          epochStep.biasUpdates.push({
            description: `Actualización del sesgo para ${neuron.label}`,
            formula: `Δb = -η * δ = -${learningRate} * ${delta}`,
            result: biasUpdate,
          })
        }
      }

      // Actualizar los pesos de todas las conexiones
      for (const connection of networkCopy.connections) {
        const sourceNeuron = networkCopy.layers
          .flatMap((l: Layer) => l.neurons)
          .find((n: Neuron) => n.id === connection.sourceId)

        const targetNeuron = networkCopy.layers
          .flatMap((l: Layer) => l.neurons)
          .find((n: Neuron) => n.id === connection.targetId)

        if (sourceNeuron && targetNeuron) {
          const targetLayer = networkCopy.layers.find((l: Layer) => l.id === targetNeuron.layerId)

          let delta
          if (targetLayer.type === "output") {
            delta = outputDeltas[targetNeuron.id]
          } else {
            delta = hiddenDeltas[targetNeuron.id]
          }

          const weightUpdate = -learningRate * delta * sourceNeuron.activationValue
          connection.weight += weightUpdate

          epochStep.weightUpdates.push({
            description: `Actualización del peso w_${sourceNeuron.label}_${targetNeuron.label}`,
            formula: `Δw = -η * δ * a = -${learningRate} * ${delta} * ${sourceNeuron.activationValue}`,
            result: weightUpdate,
          })
        }
      }

      epochSteps.push(epochStep)

      // Actualizar el estado de la red con el progreso del entrenamiento
      setNetwork((prev) => ({
        ...prev,
        currentEpoch: epoch + 1,
        trainingError: [...prev.trainingError, totalError],
      }))

      // Pequeña pausa para permitir que la UI se actualice
      await new Promise((resolve) => setTimeout(resolve, 10))
    }

    // Finalizar el entrenamiento
    setNetwork((prev) => ({
      ...prev,
      backpropagationSteps: { epochSteps },
      trainingInProgress: false,
      trainingComplete: true,
    }))
  }

  // Reemplazar la función runBackpropagation with this versión mejorada
  const runBackpropagation = async (expectedOutputs: number[], epochs: number, learningRate: number) => {
    // Inicializar el estado de entrenamiento
    setNetwork((prev) => ({
      ...prev,
      currentEpoch: 0,
      totalEpochs: epochs,
      trainingInProgress: true,
      trainingComplete: false,
      trainingError: [],
      backpropagationSteps: { epochSteps: [] },
    }))

    // Crear una copia profunda de la red para trabajar con ella
    let networkCopy = JSON.parse(JSON.stringify(network))
    const epochSteps: BackpropagationSteps["epochSteps"] = []

    // Ejecutar el entrenamiento para cada época
    for (let epoch = 0; epoch < epochs; epoch++) {
      // Verificar si el entrenamiento fue detenido
      const currentState = await new Promise<NetworkState>((resolve) => {
        // Obtener el estado actual de la red
        setNetwork((prev) => {
          resolve(prev)
          return prev
        })
      })

      if (!currentState.trainingInProgress) {
        console.log("Entrenamiento detenido por el usuario en la época", epoch)
        break
      }

      // Inicializar los pasos para esta época
      const epochStep = {
        epoch: epoch + 1,
        error: 0,
        outputGradients: [] as MathStep[],
        hiddenGradients: [] as MathStep[],
        weightUpdates: [] as MathStep[],
        biasUpdates: [] as MathStep[],
        forwardPropagationSteps: { layerSteps: [] as any[] },
      }

      // Ejecutar propagación hacia adelante y registrar los pasos
      const forwardPropResult = executeForwardPropagation(networkCopy)
      networkCopy = forwardPropResult.network
      epochStep.forwardPropagationSteps = forwardPropResult.steps

      // Obtener las capas en orden
      const inputLayer = networkCopy.layers.find((l: Layer) => l.type === "input")
      const hiddenLayers = networkCopy.layers.filter((l: Layer) => l.type === "hidden")
      const outputLayer = networkCopy.layers.find((l: Layer) => l.type === "output")

      // Calcular el error de la época
      let totalError = 0
      for (let i = 0; i < outputLayer.neurons.length; i++) {
        const neuron = outputLayer.neurons[i]
        const expected = expectedOutputs[i] || 0
        const output = neuron.activationValue
        const error = 0.5 * Math.pow(expected - output, 2)
        totalError += error

        epochStep.outputGradients.push({
          description: `Error para ${neuron.label}`,
          formula: `E = 0.5 * (${expected} - ${output})^2`,
          result: error,
        })
      }
      epochStep.error = totalError

      // Calcular gradientes para la capa de salida
      const outputDeltas: { [key: string]: number } = {}
      for (let i = 0; i < outputLayer.neurons.length; i++) {
        const neuron = outputLayer.neurons[i]
        const expected = expectedOutputs[i] || 0
        const output = neuron.activationValue

        // Calcular delta: derivada del error respecto a la salida * derivada de la activación
        const errorGradient = -(expected - output)

        // Encontrar la suma ponderada (z) para esta neurona
        const prevLayer = hiddenLayers.length > 0 ? hiddenLayers[hiddenLayers.length - 1] : inputLayer
        let z = neuron.bias
        for (const prevNeuron of prevLayer.neurons) {
          const connection = networkCopy.connections.find(
            (c: Connection) => c.sourceId === prevNeuron.id && c.targetId === neuron.id,
          )
          if (connection) {
            z += prevNeuron.activationValue * connection.weight
          }
        }

        const activationGradient = activationDerivatives[neuron.activationFunction](z)
        const delta = errorGradient * activationGradient
        outputDeltas[neuron.id] = delta

        epochStep.outputGradients.push({
          description: `Gradiente para ${neuron.label}`,
          formula: `δ = -(${expected} - ${output}) * ${neuron.activationFunction}'(${z.toFixed(6)})`,
          result: delta,
          substeps: [
            {
              description: "Derivada del error respecto a la salida",
              formula: `∂E/∂y = -(${expected} - ${output}) = ${errorGradient.toFixed(6)}`,
              result: errorGradient,
            },
            {
              description: `Derivada de la función de activación ${neuron.activationFunction}`,
              formula: `${neuron.activationFunction}'(${z.toFixed(6)}) = ${activationGradient.toFixed(6)}`,
              result: activationGradient,
            },
            {
              description: "Cálculo del delta (regla de la cadena)",
              formula: `δ = ${errorGradient.toFixed(6)} * ${activationGradient.toFixed(6)} = ${delta.toFixed(6)}`,
              result: delta,
            },
          ],
        })

        // Actualizar el sesgo de la neurona de salida
        const oldBias = neuron.bias
        const biasUpdate = -learningRate * delta
        neuron.bias += biasUpdate

        epochStep.biasUpdates.push({
          description: `Actualización del sesgo para ${neuron.label}`,
          formula: `Δb = -η * δ = -${learningRate} * ${delta.toFixed(6)} = ${biasUpdate.toFixed(6)}`,
          result: biasUpdate,
          substeps: [
            {
              description: "Valor anterior del sesgo",
              formula: `b_anterior = ${oldBias.toFixed(6)}`,
              result: oldBias,
            },
            {
              description: "Nuevo valor del sesgo",
              formula: `b_nuevo = b_anterior + Δb = ${oldBias.toFixed(6)} + ${biasUpdate.toFixed(6)} = ${neuron.bias.toFixed(6)}`,
              result: neuron.bias,
            },
          ],
        })
      }

      // Calcular gradientes para las capas ocultas (de atrás hacia adelante)
      const hiddenDeltas: { [key: string]: number } = {}
      for (let i = hiddenLayers.length - 1; i >= 0; i--) {
        const currentLayer = hiddenLayers[i]
        const nextLayer = i === hiddenLayers.length - 1 ? outputLayer : hiddenLayers[i + 1]

        for (const neuron of currentLayer.neurons) {
          // Calcular la suma de los deltas ponderados de la capa siguiente
          let sumDeltaWeights = 0
          const deltaSumSteps: MathStep[] = []

          for (const nextNeuron of nextLayer.neurons) {
            const connection = networkCopy.connections.find(
              (c: Connection) => c.sourceId === neuron.id && c.targetId === nextNeuron.id,
            )

            if (connection) {
              const nextDelta =
                i === hiddenLayers.length - 1 ? outputDeltas[nextNeuron.id] : hiddenDeltas[nextNeuron.id]
              const deltaWeight = nextDelta * connection.weight
              sumDeltaWeights += deltaWeight

              deltaSumSteps.push({
                description: `Contribución desde ${nextNeuron.label}`,
                formula: `δ_${nextNeuron.label} * w_${neuron.label}_${nextNeuron.label} = ${nextDelta.toFixed(6)} * ${connection.weight.toFixed(6)} = ${deltaWeight.toFixed(6)}`,
                result: deltaWeight,
              })
            }
          }

          // Encontrar la suma ponderada (z) para esta neurona
          const prevLayer = i === 0 ? inputLayer : hiddenLayers[i - 1]
          let z = neuron.bias
          for (const prevNeuron of prevLayer.neurons) {
            const connection = networkCopy.connections.find(
              (c: Connection) => c.sourceId === prevNeuron.id && c.targetId === neuron.id,
            )
            if (connection) {
              z += prevNeuron.activationValue * connection.weight
            }
          }

          const activationGradient = activationDerivatives[neuron.activationFunction](z)
          const delta = sumDeltaWeights * activationGradient
          hiddenDeltas[neuron.id] = delta

          epochStep.hiddenGradients.push({
            description: `Gradiente para ${neuron.label}`,
            formula: `δ = (Σ δ_siguiente * w) * ${neuron.activationFunction}'(${z.toFixed(6)})`,
            result: delta,
            substeps: [
              {
                description: "Suma de deltas ponderados de la capa siguiente",
                formula: `Σ δ_siguiente * w = ${sumDeltaWeights.toFixed(6)}`,
                result: sumDeltaWeights,
                substeps: deltaSumSteps,
              },
              {
                description: `Derivada de la función de activación ${neuron.activationFunction}`,
                formula: `${neuron.activationFunction}'(${z.toFixed(6)}) = ${activationGradient.toFixed(6)}`,
                result: activationGradient,
              },
              {
                description: "Cálculo del delta (regla de la cadena)",
                formula: `δ = ${sumDeltaWeights.toFixed(6)} * ${activationGradient.toFixed(6)} = ${delta.toFixed(6)}`,
                result: delta,
              },
            ],
          })

          // Actualizar el sesgo de la neurona oculta
          const oldBias = neuron.bias
          const biasUpdate = -learningRate * delta
          neuron.bias += biasUpdate

          epochStep.biasUpdates.push({
            description: `Actualización del sesgo para ${neuron.label}`,
            formula: `Δb = -η * δ = -${learningRate} * ${delta.toFixed(6)} = ${biasUpdate.toFixed(6)}`,
            result: biasUpdate,
            substeps: [
              {
                description: "Valor anterior del sesgo",
                formula: `b_anterior = ${oldBias.toFixed(6)}`,
                result: oldBias,
              },
              {
                description: "Nuevo valor del sesgo",
                formula: `b_nuevo = b_anterior + Δb = ${oldBias.toFixed(6)} + ${biasUpdate.toFixed(6)} = ${neuron.bias.toFixed(6)}`,
                result: neuron.bias,
              },
            ],
          })
        }
      }

      // Actualizar los pesos de todas las conexiones
      for (const connection of networkCopy.connections) {
        const sourceNeuron = networkCopy.layers
          .flatMap((l: Layer) => l.neurons)
          .find((n: Neuron) => n.id === connection.sourceId)

        const targetNeuron = networkCopy.layers
          .flatMap((l: Layer) => l.neurons)
          .find((n: Neuron) => n.id === connection.targetId)

        if (sourceNeuron && targetNeuron) {
          const targetLayer = networkCopy.layers.find((l: Layer) => l.id === targetNeuron.layerId)

          let delta
          if (targetLayer.type === "output") {
            delta = outputDeltas[targetNeuron.id]
          } else {
            delta = hiddenDeltas[targetNeuron.id]
          }

          if (delta !== undefined) {
            // Asegurarse de que delta existe
            const oldWeight = connection.weight
            const weightUpdate = -learningRate * delta * sourceNeuron.activationValue
            connection.weight += weightUpdate

            epochStep.weightUpdates.push({
              description: `Actualización del peso w_${sourceNeuron.label}_${targetNeuron.label}`,
              formula: `Δw = -η * δ * a = -${learningRate} * ${delta.toFixed(6)} * ${sourceNeuron.activationValue.toFixed(6)} = ${weightUpdate.toFixed(6)}`,
              result: weightUpdate,
              substeps: [
                {
                  description: "Valor anterior del peso",
                  formula: `w_anterior = ${oldWeight.toFixed(6)}`,
                  result: oldWeight,
                },
                {
                  description: "Nuevo valor del peso",
                  formula: `w_nuevo = w_anterior + Δw = ${oldWeight.toFixed(6)} + ${weightUpdate.toFixed(6)} = ${connection.weight.toFixed(6)}`,
                  result: connection.weight,
                },
              ],
            })
          }
        }
      }

      epochSteps.push(epochStep)

      // Actualizar el estado de la red con el progreso del entrenamiento
      // IMPORTANTE: Actualizar la red completa para que los cambios se reflejen visualmente
      setNetwork({
        ...networkCopy,
        selectedNeuron: network.selectedNeuron,
        selectedConnection: network.selectedConnection,
        selectedLayer: network.selectedLayer,
        currentEpoch: epoch + 1,
        totalEpochs: epochs,
        trainingInProgress: true,
        trainingComplete: false,
        trainingError: [...(currentState.trainingError || []), totalError],
        backpropagationSteps: { epochSteps: [...epochSteps] },
        forwardPropagationSteps: forwardPropResult.steps,
      })

      // Pequeña pausa para permitir que la UI se actualice
      await new Promise((resolve) => setTimeout(resolve, 50))
    }

    // Ejecutar una propagación hacia adelante final para actualizar los valores de salida
    const finalForwardProp = executeForwardPropagation(networkCopy)
    networkCopy = finalForwardProp.network

    // Finalizar el entrenamiento y actualizar la red con los valores finales
    setNetwork({
      ...networkCopy,
      selectedNeuron: network.selectedNeuron,
      selectedConnection: network.selectedConnection,
      selectedLayer: network.selectedLayer,
      backpropagationSteps: { epochSteps },
      trainingInProgress: false,
      trainingComplete: true,
      forwardPropagationSteps: finalForwardProp.steps, // Mostrar los resultados finales
    })
  }

  // Añadir esta nueva función para ejecutar la propagación hacia adelante y registrar los pasos
  const executeForwardPropagation = (networkCopy: NetworkState) => {
    const layerSteps: ForwardPropagationSteps["layerSteps"] = []

    // Obtener las capas en orden
    const inputLayer = networkCopy.layers.find((l: Layer) => l.type === "input")
    const hiddenLayers = networkCopy.layers.filter((l: Layer) => l.type === "hidden")
    const outputLayer = networkCopy.layers.find((l: Layer) => l.type === "output")

    // Procesar capa de entrada (no hay cálculos, solo valores de entrada)
    const inputLayerSteps = {
      layerId: inputLayer.id,
      layerType: "input",
      neurons: inputLayer.neurons.map((neuron: Neuron) => ({
        neuronId: neuron.id,
        steps: [
          {
            description: `Valor de entrada para ${neuron.label}`,
            formula: `${neuron.label} = ${neuron.activationValue}`,
            result: neuron.activationValue,
          },
        ],
      })),
    }
    layerSteps.push(inputLayerSteps)

    // Procesar capas ocultas
    for (let i = 0; i < hiddenLayers.length; i++) {
      const currentLayer = hiddenLayers[i]
      const prevLayer = i === 0 ? inputLayer : hiddenLayers[i - 1]

      const layerStep = {
        layerId: currentLayer.id,
        layerType: "hidden",
        neurons: [] as { neuronId: string; steps: MathStep[] }[],
      }

      // Calcular la salida para cada neurona en la capa actual
      for (const neuron of currentLayer.neurons) {
        const neuronSteps: MathStep[] = []

        // Paso 1: Calcular la suma ponderada (z)
        const weightedSumStep: MathStep = {
          description: `Calcular la suma ponderada para ${neuron.label}`,
          formula: `z = Σ(entrada_i * peso_i) + sesgo`,
          result: 0,
          substeps: [],
        }

        let weightedSum = neuron.bias
        weightedSumStep.substeps!.push({
          description: "Sesgo",
          formula: `b = ${neuron.bias}`,
          result: neuron.bias,
        })

        // Sumar cada entrada ponderada
        for (const prevNeuron of prevLayer.neurons) {
          const connection = networkCopy.connections.find(
            (c: Connection) => c.sourceId === prevNeuron.id && c.targetId === neuron.id,
          )

          if (connection) {
            const weightedInput = prevNeuron.activationValue * connection.weight
            weightedSum += weightedInput

            weightedSumStep.substeps!.push({
              description: `Entrada desde ${prevNeuron.label}`,
              formula: `${prevNeuron.label} * w_${prevNeuron.label}_${neuron.label} = ${prevNeuron.activationValue.toFixed(6)} * ${connection.weight.toFixed(6)}`,
              result: weightedInput,
            })
          }
        }

        weightedSumStep.result = weightedSum
        neuronSteps.push(weightedSumStep)

        // Paso 2: Aplicar la función de activación
        const activationStep: MathStep = {
          description: `Aplicar función de activación ${neuron.activationFunction}`,
          formula: `a = ${neuron.activationFunction}(z) = ${neuron.activationFunction}(${weightedSum.toFixed(6)})`,
          result: 0,
        }

        const activationValue = activationFunctions[neuron.activationFunction](weightedSum)
        activationStep.result = activationValue
        neuronSteps.push(activationStep)

        // Actualizar el valor de activación de la neurona
        neuron.activationValue = activationValue

        layerStep.neurons.push({
          neuronId: neuron.id,
          steps: neuronSteps,
        })
      }

      layerSteps.push(layerStep)
    }

    // Procesar capa de salida
    const outputLayerStep = {
      layerId: outputLayer.id,
      layerType: "output",
      neurons: [] as { neuronId: string; steps: MathStep[] }[],
    }

    const prevLayer = hiddenLayers.length > 0 ? hiddenLayers[hiddenLayers.length - 1] : inputLayer

    // Calcular la salida para cada neurona en la capa de salida
    for (const neuron of outputLayer.neurons) {
      const neuronSteps: MathStep[] = []

      // Paso 1: Calcular la suma ponderada (z)
      const weightedSumStep: MathStep = {
        description: `Calcular la suma ponderada para ${neuron.label}`,
        formula: `z = Σ(entrada_i * peso_i) + sesgo`,
        result: 0,
        substeps: [],
      }

      let weightedSum = neuron.bias
      weightedSumStep.substeps!.push({
        description: "Sesgo",
        formula: `b = ${neuron.bias}`,
        result: neuron.bias,
      })

      // Sumar cada entrada ponderada
      for (const prevNeuron of prevLayer.neurons) {
        const connection = networkCopy.connections.find(
          (c: Connection) => c.sourceId === prevNeuron.id && c.targetId === neuron.id,
        )

        if (connection) {
          const weightedInput = prevNeuron.activationValue * connection.weight
          weightedSum += weightedInput

          weightedSumStep.substeps!.push({
            description: `Entrada desde ${prevNeuron.label}`,
            formula: `${prevNeuron.label} * w_${prevNeuron.label}_${neuron.label} = ${prevNeuron.activationValue.toFixed(6)} * ${connection.weight.toFixed(6)}`,
            result: weightedInput,
          })
        }
      }

      weightedSumStep.result = weightedSum
      neuronSteps.push(weightedSumStep)

      // Paso 2: Aplicar la función de activación
      const activationStep: MathStep = {
        description: `Aplicar función de activación ${neuron.activationFunction}`,
        formula: `a = ${neuron.activationFunction}(z) = ${neuron.activationFunction}(${weightedSum.toFixed(6)})`,
        result: 0,
      }

      const activationValue = activationFunctions[neuron.activationFunction](weightedSum)
      activationStep.result = activationValue
      neuronSteps.push(activationStep)

      // Actualizar el valor de activación de la neurona
      neuron.activationValue = activationValue

      outputLayerStep.neurons.push({
        neuronId: neuron.id,
        steps: neuronSteps,
      })
    }

    layerSteps.push(outputLayerStep)

    return {
      network: networkCopy,
      steps: { layerSteps },
    }
  }

  return (
    <NetworkContext.Provider
      value={{
        network,
        addLayer,
        removeLayer,
        addNeuron,
        removeNeuron,
        updateNeuron,
        updateConnection,
        updateLayer,
        selectNeuron,
        selectConnection,
        selectLayer,
        exportNetwork,
        setInputNeurons,
        setOutputNeurons,
        resetNetwork,
        randomizeNetwork,
        runForwardPropagation,
        runBackpropagation,
        stopTraining,
        clearMathSteps,
      }}
    >
      {children}
    </NetworkContext.Provider>
  )
}

export function useNetwork() {
  const context = useContext(NetworkContext)
  if (context === undefined) {
    throw new Error("useNetwork must be used within a NetworkProvider")
  }
  return context
}
