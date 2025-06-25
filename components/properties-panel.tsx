"use client"

import { useNetwork, type ActivationFunction } from "./network-context"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Trash2, Plus } from "lucide-react"

export default function PropertiesPanel() {
  const { network, updateNeuron, updateConnection, updateLayer, removeNeuron, removeLayer, addNeuron } = useNetwork()

  const { selectedNeuron, selectedConnection, selectedLayer } = network

  const activationFunctions: { value: ActivationFunction; label: string }[] = [
    { value: "sigmoid", label: "Sigmoid" },
    { value: "relu", label: "ReLU" },
    { value: "tanh", label: "Tanh" },
    { value: "linear", label: "Linear" },
    { value: "leakyRelu", label: "Leaky ReLU" },
    { value: "swish", label: "Swish" },
  ]

  const handleNeuronLabelChange = (value: string) => {
    if (selectedNeuron) {
      updateNeuron({
        ...selectedNeuron,
        label: value,
      })
    }
  }

  const handleNeuronBiasChange = (value: string) => {
    if (selectedNeuron) {
      updateNeuron({
        ...selectedNeuron,
        bias: Number.parseFloat(value) || 0,
      })
    }
  }

  const handleNeuronActivationChange = (value: ActivationFunction) => {
    if (selectedNeuron) {
      updateNeuron({
        ...selectedNeuron,
        activationFunction: value,
      })
    }
  }

  const handleConnectionWeightChange = (value: string) => {
    if (selectedConnection) {
      updateConnection({
        ...selectedConnection,
        weight: Number.parseFloat(value) || 0,
      })
    }
  }

  const handleLayerActivationChange = (value: ActivationFunction) => {
    if (selectedLayer) {
      updateLayer({
        ...selectedLayer,
        activationFunction: value,
      })
    }
  }

  return (
    <div className="w-full lg:w-80 bg-white border-l border-gray-200 p-2 lg:p-4 flex flex-col h-screen overflow-auto">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg lg:text-xl font-bold">Propiedades</h2>
      </div>

      <Separator className="my-4" />

      {selectedNeuron && (
        <div className="space-y-4">
          <h3 className="text-base lg:text-lg font-semibold">Propiedades de la Neurona</h3>

          <div className="space-y-2">
            <Label htmlFor="neuron-id" className="text-sm">
              ID
            </Label>
            <Input id="neuron-id" value={selectedNeuron.id} readOnly disabled className="text-sm" />
          </div>

          <div className="space-y-2">
            <Label htmlFor="neuron-label" className="text-sm">
              Etiqueta
            </Label>
            <Input
              id="neuron-label"
              value={selectedNeuron.label}
              onChange={(e) => handleNeuronLabelChange(e.target.value)}
              className="text-sm"
            />
          </div>

          {/* Solo mostrar el sesgo para capas ocultas y de salida */}
          {selectedNeuron.layerId !== "input-layer" && (
            <div className="space-y-2">
              <Label htmlFor="neuron-bias" className="text-sm">
                Sesgo (Bias)
              </Label>
              <Input
                id="neuron-bias"
                type="number"
                step="0.1"
                value={selectedNeuron.bias}
                onChange={(e) => handleNeuronBiasChange(e.target.value)}
                className="text-sm"
              />
            </div>
          )}

          {/* Solo mostrar la función de activación para capas ocultas */}
          {selectedNeuron.layerId.includes("hidden") && (
            <div className="space-y-2">
              <Label htmlFor="neuron-activation-function" className="text-sm">
                Función de Activación
              </Label>
              <Select value={selectedNeuron.activationFunction} onValueChange={handleNeuronActivationChange}>
                <SelectTrigger id="neuron-activation-function" className="text-sm">
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

          <Button variant="destructive" className="w-full mt-4 text-sm" onClick={() => removeNeuron(selectedNeuron.id)}>
            <Trash2 className="h-4 w-4 mr-2" />
            Eliminar Neurona
          </Button>
        </div>
      )}

      {selectedConnection && (
        <div className="space-y-4">
          <h3 className="text-base lg:text-lg font-semibold">Propiedades de la Conexión</h3>

          <div className="space-y-2">
            <Label htmlFor="connection-id" className="text-sm">
              ID
            </Label>
            <Input id="connection-id" value={selectedConnection.id} readOnly disabled className="text-sm" />
          </div>

          <div className="space-y-2">
            <Label htmlFor="connection-source" className="text-sm">
              Origen
            </Label>
            <Input id="connection-source" value={selectedConnection.sourceId} readOnly disabled className="text-sm" />
          </div>

          <div className="space-y-2">
            <Label htmlFor="connection-target" className="text-sm">
              Destino
            </Label>
            <Input id="connection-target" value={selectedConnection.targetId} readOnly disabled className="text-sm" />
          </div>

          <div className="space-y-2">
            <Label htmlFor="connection-weight" className="text-sm">
              Peso
            </Label>
            <Input
              id="connection-weight"
              type="number"
              step="0.1"
              value={selectedConnection.weight}
              onChange={(e) => handleConnectionWeightChange(e.target.value)}
              className="text-sm"
            />
            <p className="text-xs text-gray-500">
              También puedes arrastrar la conexión verticalmente para cambiar el peso
            </p>
          </div>
        </div>
      )}

      {selectedLayer && (
        <div className="space-y-4">
          <h3 className="text-base lg:text-lg font-semibold">Propiedades de la Capa</h3>

          <div className="space-y-2">
            <Label htmlFor="layer-id" className="text-sm">
              ID
            </Label>
            <Input id="layer-id" value={selectedLayer.id} readOnly disabled className="text-sm" />
          </div>

          <div className="space-y-2">
            <Label htmlFor="layer-type" className="text-sm">
              Tipo
            </Label>
            <Input
              id="layer-type"
              value={selectedLayer.type === "input" ? "Entrada" : selectedLayer.type === "output" ? "Salida" : "Oculta"}
              readOnly
              disabled
              className="text-sm"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="layer-neurons" className="text-sm">
              Neuronas
            </Label>
            <div className="mt-2 space-y-2 max-h-40 overflow-y-auto border rounded-md p-2">
              {selectedLayer.neurons.map((neuron) => (
                <div key={neuron.id} className="flex items-center justify-between bg-gray-50 p-2 rounded">
                  <span className="text-sm">{neuron.label}</span>
                  {selectedLayer.neurons.length > 1 && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0 text-red-500"
                      onClick={() => removeNeuron(neuron.id)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Solo mostrar la función de activación para capas ocultas */}
          {selectedLayer.type === "hidden" && (
            <div className="space-y-2">
              <Label htmlFor="layer-activation-function" className="text-sm">
                Función de Activación
              </Label>
              <Select value={selectedLayer.activationFunction} onValueChange={handleLayerActivationChange}>
                <SelectTrigger id="layer-activation-function" className="text-sm">
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
              <p className="text-xs text-gray-500 mt-1">
                Esto actualizará la función de activación para todas las neuronas en esta capa.
              </p>
            </div>
          )}

          <div className="flex gap-2 mt-4">
            <Button variant="outline" className="flex-1 text-sm" onClick={() => addNeuron(selectedLayer.id)}>
              <Plus className="h-4 w-4 mr-2" />
              Añadir
            </Button>

            {selectedLayer.type !== "input" && selectedLayer.type !== "output" && (
              <Button variant="destructive" className="flex-1 text-sm" onClick={() => removeLayer(selectedLayer.id)}>
                <Trash2 className="h-4 w-4 mr-2" />
                Eliminar
              </Button>
            )}
          </div>
        </div>
      )}

      {!selectedNeuron && !selectedConnection && !selectedLayer && (
        <div className="text-center py-8 text-gray-500">
          <p className="text-sm">Selecciona una neurona, conexión o capa para ver y editar sus propiedades.</p>
        </div>
      )}
    </div>
  )
}
