"use client"

import type React from "react"

import { useRef, useEffect, useState } from "react"
import { useNetwork, type Neuron, type Connection } from "./network-context"

export default function NetworkCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const { network, selectNeuron, selectConnection, updateConnection, updateNeuron } = useNetwork()
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [draggedNeuron, setDraggedNeuron] = useState<Neuron | null>(null)
  const [draggedConnection, setDraggedConnection] = useState<Connection | null>(null)
  const [dragStartY, setDragStartY] = useState(0)
  const [tooltip, setTooltip] = useState({ visible: false, text: "", x: 0, y: 0 })

  // Calculate positions for neurons
  const calculatePositions = () => {
    const width = dimensions.width
    const height = dimensions.height

    if (width === 0 || height === 0) return

    const layerSpacing = width / (network.layers.length + 1)
    const updatedLayers = network.layers.map((layer, layerIndex) => {
      const neuronSpacing = height / (layer.neurons.length + 1)

      const updatedNeurons = layer.neurons.map((neuron, neuronIndex) => {
        return {
          ...neuron,
          x: (layerIndex + 1) * layerSpacing,
          y: (neuronIndex + 1) * neuronSpacing,
        }
      })

      return {
        ...layer,
        neurons: updatedNeurons,
      }
    })

    return updatedLayers
  }

  // Draw the network on canvas
  const drawNetwork = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const layers = calculatePositions()
    if (!layers) return

    // Draw connections
    network.connections.forEach((connection) => {
      const sourceNeuron = layers.flatMap((l) => l.neurons).find((n) => n.id === connection.sourceId)

      const targetNeuron = layers.flatMap((l) => l.neurons).find((n) => n.id === connection.targetId)

      if (sourceNeuron && targetNeuron && sourceNeuron.x && sourceNeuron.y && targetNeuron.x && targetNeuron.y) {
        // Draw connection line
        ctx.beginPath()
        ctx.moveTo(sourceNeuron.x, sourceNeuron.y)
        ctx.lineTo(targetNeuron.x, targetNeuron.y)

        // Style based on selection and weight
        if (network.selectedConnection && network.selectedConnection.id === connection.id) {
          ctx.strokeStyle = "#3b82f6" // Blue for selected connection
          ctx.lineWidth = 3
        } else {
          // Line width based on weight (thicker = stronger weight)
          const absWeight = Math.abs(connection.weight)
          const lineWidth = 0.5 + absWeight * 2

          // Color based on weight sign (red for negative, green for positive)
          ctx.strokeStyle =
            connection.weight >= 0
              ? `rgba(34, 197, 94, ${0.3 + absWeight * 0.7})`
              : // Green with opacity based on weight
                `rgba(239, 68, 68, ${0.3 + absWeight * 0.7})` // Red with opacity based on weight

          ctx.lineWidth = lineWidth
        }

        ctx.stroke()

        // Draw weight value
        const midX = (sourceNeuron.x + targetNeuron.x) / 2
        const midY = (sourceNeuron.y + targetNeuron.y) / 2

        ctx.fillStyle = "#1e293b"
        ctx.font = "12px Arial"
        ctx.fillText(connection.weight.toFixed(2), midX, midY)

        // Draw arrow
        const angle = Math.atan2(targetNeuron.y - sourceNeuron.y, targetNeuron.x - sourceNeuron.x)
        const arrowLength = 10

        ctx.beginPath()
        ctx.moveTo(targetNeuron.x - 25 * Math.cos(angle), targetNeuron.y - 25 * Math.sin(angle))
        ctx.lineTo(
          targetNeuron.x - 25 * Math.cos(angle) - arrowLength * Math.cos(angle - Math.PI / 6),
          targetNeuron.y - 25 * Math.sin(angle) - arrowLength * Math.sin(angle - Math.PI / 6),
        )
        ctx.lineTo(
          targetNeuron.x - 25 * Math.cos(angle) - arrowLength * Math.cos(angle + Math.PI / 6),
          targetNeuron.y - 25 * Math.sin(angle) - arrowLength * Math.sin(angle + Math.PI / 6),
        )
        ctx.closePath()
        ctx.fillStyle = ctx.strokeStyle
        ctx.fill()
      }
    })

    // Draw neurons
    layers.forEach((layer) => {
      layer.neurons.forEach((neuron) => {
        if (neuron.x === undefined || neuron.y === undefined) return

        // Draw neuron circle
        ctx.beginPath()
        ctx.arc(neuron.x, neuron.y, 25, 0, Math.PI * 2)

        // Style based on selection and layer type
        if (network.selectedNeuron && network.selectedNeuron.id === neuron.id) {
          ctx.fillStyle = "#3b82f6" // Blue for selected neuron
        } else if (network.trainingInProgress || network.trainingComplete) {
          // Color based on layer type with training highlight
          if (layer.type === "input") {
            ctx.fillStyle = "#a7f3d0" // Green for input
          } else if (layer.type === "output") {
            // Highlight output neurons based on training status
            ctx.fillStyle = network.trainingComplete ? "#fecaca" : "#fecaca" // Red for output
          } else {
            // Highlight hidden neurons during training
            ctx.fillStyle = "#bfdbfe" // Blue for hidden
          }
        } else {
          // Normal colors
          if (layer.type === "input") {
            ctx.fillStyle = "#a7f3d0" // Green for input
          } else if (layer.type === "output") {
            ctx.fillStyle = "#fecaca" // Red for output
          } else {
            ctx.fillStyle = "#bfdbfe" // Blue for hidden
          }
        }

        ctx.fill()
        ctx.strokeStyle = "#1e293b"
        ctx.lineWidth = 1
        ctx.stroke()

        // Draw neuron label
        ctx.fillStyle = "#1e293b"
        ctx.font = "14px Arial"
        ctx.textAlign = "center"
        ctx.textBaseline = "middle"
        ctx.fillText(neuron.label, neuron.x, neuron.y - 8)

        // Mostrar información según el tipo de capa
        if (layer.type === "input") {
          // Para capa de entrada, mostrar el valor de x
          ctx.font = "12px Arial"
          ctx.fillText(`x: ${neuron.activationValue.toFixed(2)}`, neuron.x, neuron.y + 8)
        } else if (layer.type === "output") {
          // Para capa de salida, mostrar el valor de activación (resultado)
          ctx.font = "12px Arial"
          // Resaltar el valor de salida si el entrenamiento está completo
          if (network.trainingComplete) {
            ctx.fillStyle = "#047857" // Verde oscuro para resaltar
            ctx.font = "bold 12px Arial"
          }
          ctx.fillText(`y: ${neuron.activationValue.toFixed(4)}`, neuron.x, neuron.y + 8)

          // Mostrar el sesgo debajo
          ctx.fillStyle = "#1e293b"
          ctx.font = "10px Arial"
          ctx.fillText(`b: ${neuron.bias.toFixed(2)}`, neuron.x, neuron.y + 22)
        } else {
          // Para capas ocultas, mostrar el sesgo
          ctx.font = "12px Arial"
          ctx.fillText(`b: ${neuron.bias.toFixed(2)}`, neuron.x, neuron.y + 8)
        }
      })
    })

    // Draw tooltip if visible
    if (tooltip.visible) {
      ctx.fillStyle = "rgba(0, 0, 0, 0.7)"
      ctx.fillRect(tooltip.x, tooltip.y - 20, 120, 20)
      ctx.fillStyle = "white"
      ctx.font = "12px Arial"
      ctx.textAlign = "left"
      ctx.fillText(tooltip.text, tooltip.x + 5, tooltip.y - 7)
    }

    // Si el entrenamiento está en progreso, mostrar información adicional
    if (network.trainingInProgress) {
      ctx.fillStyle = "rgba(59, 130, 246, 0.8)" // Azul semi-transparente
      ctx.fillRect(10, 10, 200, 60)
      ctx.fillStyle = "white"
      ctx.font = "12px Arial"
      ctx.textAlign = "left"
      ctx.fillText(`Entrenando: Época ${network.currentEpoch} de ${network.totalEpochs}`, 20, 30)
      ctx.fillText(
        `Error actual: ${network.trainingError.length > 0 ? network.trainingError[network.trainingError.length - 1].toFixed(6) : "N/A"}`,
        20,
        50,
      )
    } else if (network.trainingComplete) {
      ctx.fillStyle = "rgba(16, 185, 129, 0.8)" // Verde semi-transparente
      ctx.fillRect(10, 10, 200, 60)
      ctx.fillStyle = "white"
      ctx.font = "12px Arial"
      ctx.textAlign = "left"
      ctx.fillText(`Entrenamiento completado: ${network.totalEpochs} épocas`, 20, 30)
      ctx.fillText(
        `Error final: ${network.trainingError.length > 0 ? network.trainingError[network.trainingError.length - 1].toFixed(6) : "N/A"}`,
        20,
        50,
      )
    }
  }

  // Handle canvas click to select neurons or connections
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return

    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    const layers = calculatePositions()
    if (!layers) return

    // Check if clicked on a neuron
    let clickedNeuron: Neuron | null = null

    for (const layer of layers) {
      for (const neuron of layer.neurons) {
        if (neuron.x === undefined || neuron.y === undefined) continue

        const distance = Math.sqrt(Math.pow(x - neuron.x, 2) + Math.pow(y - neuron.y, 2))
        if (distance <= 25) {
          // Neuron radius is 25
          clickedNeuron = neuron
          break
        }
      }
      if (clickedNeuron) break
    }

    if (clickedNeuron) {
      selectNeuron(clickedNeuron)
      return
    }

    // Check if clicked on a connection
    let clickedConnection: Connection | null = null

    for (const connection of network.connections) {
      const sourceNeuron = layers.flatMap((l) => l.neurons).find((n) => n.id === connection.sourceId)

      const targetNeuron = layers.flatMap((l) => l.neurons).find((n) => n.id === connection.targetId)

      if (sourceNeuron && targetNeuron && sourceNeuron.x && sourceNeuron.y && targetNeuron.x && targetNeuron.y) {
        // Check if click is near the line
        const distance = distanceToLine(x, y, sourceNeuron.x, sourceNeuron.y, targetNeuron.x, targetNeuron.y)

        if (distance < 10) {
          // 10px threshold for line selection
          clickedConnection = connection
          break
        }
      }
    }

    if (clickedConnection) {
      selectConnection(clickedConnection)
    } else {
      // Clicked on empty space, deselect
      selectNeuron(null)
      selectConnection(null)
    }
  }

  // Calculate distance from point to line
  const distanceToLine = (x: number, y: number, x1: number, y1: number, x2: number, y2: number) => {
    const A = x - x1
    const B = y - y1
    const C = x2 - x1
    const D = y2 - y1

    const dot = A * C + B * D
    const lenSq = C * C + D * D
    let param = -1

    if (lenSq !== 0) {
      param = dot / lenSq
    }

    let xx, yy

    if (param < 0) {
      xx = x1
      yy = y1
    } else if (param > 1) {
      xx = x2
      yy = y2
    } else {
      xx = x1 + param * C
      yy = y1 + param * D
    }

    const dx = x - xx
    const dy = y - yy

    return Math.sqrt(dx * dx + dy * dy)
  }

  // Handle mouse down for dragging
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return

    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    const layers = calculatePositions()
    if (!layers) return

    // Check if clicked on a neuron
    for (const layer of layers) {
      for (const neuron of layer.neurons) {
        if (neuron.x === undefined || neuron.y === undefined) continue

        const distance = Math.sqrt(Math.pow(x - neuron.x, 2) + Math.pow(y - neuron.y, 2))
        if (distance <= 25) {
          // Neuron radius is 25
          setIsDragging(true)
          setDraggedNeuron(neuron)
          return
        }
      }
    }

    // Check if clicked on a connection (for weight adjustment)
    for (const connection of network.connections) {
      const sourceNeuron = layers.flatMap((l) => l.neurons).find((n) => n.id === connection.sourceId)

      const targetNeuron = layers.flatMap((l) => l.neurons).find((n) => n.id === connection.targetId)

      if (sourceNeuron && targetNeuron && sourceNeuron.x && sourceNeuron.y && targetNeuron.x && targetNeuron.y) {
        // Check if click is near the line
        const distance = distanceToLine(x, y, sourceNeuron.x, sourceNeuron.y, targetNeuron.x, targetNeuron.y)

        if (distance < 10) {
          // 10px threshold for line selection
          setIsDragging(true)
          setDraggedConnection(connection)
          setDragStartY(y)

          // Show tooltip
          const midX = (sourceNeuron.x + targetNeuron.x) / 2
          const midY = (sourceNeuron.y + targetNeuron.y) / 2
          setTooltip({
            visible: true,
            text: `Weight: ${connection.weight.toFixed(2)}`,
            x: midX,
            y: midY,
          })

          return
        }
      }
    }
  }

  // Handle mouse move for dragging
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging || (!draggedNeuron && !draggedConnection) || !canvasRef.current) return

    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    if (draggedNeuron) {
      // Update neuron position
      const updatedNeuron = {
        ...draggedNeuron,
        x,
        y,
      }

      // Update the neuron in the network
      const updatedLayers = network.layers.map((layer) => {
        if (layer.id === draggedNeuron.layerId) {
          return {
            ...layer,
            neurons: layer.neurons.map((neuron) => (neuron.id === draggedNeuron.id ? updatedNeuron : neuron)),
          }
        }
        return layer
      })
    } else if (draggedConnection) {
      // Calculate weight change based on vertical drag
      const deltaY = dragStartY - y
      const weightChange = deltaY * 0.01 // Scale factor for sensitivity

      // Update connection weight
      const newWeight = Math.max(-2, Math.min(2, draggedConnection.weight + weightChange))
      const updatedConnection = {
        ...draggedConnection,
        weight: newWeight,
      }

      // Update the connection in the network
      updateConnection(updatedConnection)

      // Update drag start position
      setDragStartY(y)

      // Update tooltip
      const layers = calculatePositions()
      if (layers) {
        const sourceNeuron = layers.flatMap((l) => l.neurons).find((n) => n.id === draggedConnection.sourceId)

        const targetNeuron = layers.flatMap((l) => l.neurons).find((n) => n.id === draggedConnection.targetId)

        if (sourceNeuron && targetNeuron && sourceNeuron.x && sourceNeuron.y && targetNeuron.x && targetNeuron.y) {
          const midX = (sourceNeuron.x + targetNeuron.x) / 2
          const midY = (sourceNeuron.y + targetNeuron.y) / 2

          setTooltip({
            visible: true,
            text: `Weight: ${newWeight.toFixed(2)}`,
            x: midX,
            y: midY,
          })
        }
      }
    }

    // Redraw the network
    drawNetwork()
  }

  // Handle mouse up for dragging
  const handleMouseUp = () => {
    setIsDragging(false)
    setDraggedNeuron(null)
    setDraggedConnection(null)
    setTooltip({ ...tooltip, visible: false })
  }

  // Update canvas dimensions when container size changes
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const { width, height } = containerRef.current.getBoundingClientRect()
        setDimensions({ width, height })

        if (canvasRef.current) {
          canvasRef.current.width = width
          canvasRef.current.height = height
        }
      }
    }

    updateDimensions()
    window.addEventListener("resize", updateDimensions)

    return () => {
      window.removeEventListener("resize", updateDimensions)
    }
  }, [])

  // Draw network when dimensions or network changes
  useEffect(() => {
    drawNetwork()
  }, [dimensions, network])

  return (
    <div ref={containerRef} className="w-full h-full relative">
      <div className="absolute top-2 right-2 bg-white/80 p-2 rounded-md text-xs text-gray-500 shadow-sm">
        <p>Click para seleccionar | Arrastra conexión para cambiar peso</p>
      </div>
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        onClick={handleCanvasClick}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />
    </div>
  )
}
