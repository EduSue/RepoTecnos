"use client";

import { Button } from "@/components/ui/button";
import { useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Camera, RotateCcw, ArrowLeft } from "lucide-react";

interface ImageRecognitionModuleProps {
  onBack: () => void;
}

export default function ImageRecognitionModule({
  onBack,
}: ImageRecognitionModuleProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const otherCanvasRef = useRef<HTMLCanvasElement>(null);
  const [modelo, setModelo] = useState<any>(null);
  const [currentPrediction, setCurrentPrediction] = useState<string | null>(
    null
  );
  const [facingMode, setFacingMode] = useState<"user" | "environment">("user");
  const [cameraActive, setCameraActive] = useState(false);
  const size = 400;

  const clases = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
  ];

  useEffect(() => {
    const script = document.createElement("script");
    script.src =
      "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js";
    script.onload = async () => {
      const tf = (window as any).tf;
      const loadedModel = await tf.loadLayersModel("/model.json");
      setModelo(loadedModel);
      console.log("Modelo cargado");
    };
    document.body.appendChild(script);
  }, []);

  useEffect(() => {
    if (modelo && videoRef.current) {
      startCamera();
    }
  }, [modelo]);

  const startCamera = async () => {
    setCameraActive(true);
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode, width: size, height: size },
      audio: false,
    });
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.onloadedmetadata = () => videoRef.current?.play();
    }
    procesar();
    predecir();
  };

  const cambiarCamara = () => {
    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach((track) => track.stop());
    }
    setFacingMode((prev) => (prev === "user" ? "environment" : "user"));
    setTimeout(startCamera, 100);
  };

  const procesar = () => {
    const ctx = canvasRef.current?.getContext("2d");
    if (videoRef.current && ctx) {
      ctx.drawImage(videoRef.current, 0, 0, size, size);
    }
    setTimeout(procesar, 50);
  };

  const predecir = () => {
    const tf = (window as any).tf;
    if (!modelo || !canvasRef.current || !otherCanvasRef.current) return;

    const ctx1 = canvasRef.current.getContext("2d")!;
    const ctx2 = otherCanvasRef.current.getContext("2d")!;
    ctx2.drawImage(canvasRef.current, 0, 0, 32, 32);

    const imgData = ctx2.getImageData(0, 0, 32, 32);
    const arr: number[][][] = [];
    let fila: number[][] = [];
    for (let p = 0; p < imgData.data.length; p += 4) {
      const r = imgData.data[p] / 255;
      const g = imgData.data[p + 1] / 255;
      const b = imgData.data[p + 2] / 255;
      fila.push([r, g, b]);
      if (fila.length === 32) {
        arr.push(fila);
        fila = [];
      }
    }

    const tensor = tf.tensor4d([arr]);
    const resultados = modelo.predict(tensor);
    const data = resultados.dataSync();
    const index = data.indexOf(Math.max(...data));
    setCurrentPrediction(clases[index]);
    setTimeout(predecir, 100);
  };
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Button
                variant="ghost"
                size="sm"
                onClick={onBack}
                className="mr-4"
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                <span className="hidden sm:inline">Volver al inicio</span>
              </Button>
              <div>
                <h1 className="text-xl lg:text-2xl font-bold text-gray-900">
                  Reconocimiento de Imágenes con CNN
                </h1>
                <p className="text-sm lg:text-base text-gray-600 hidden sm:block">
                  Redes Neuronales Convolucionales para clasificación de
                  imágenes
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2"></div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-1 gap-8">
          <div className="min-h-screen bg-gray-100 p-6">
            <div className="flex justify-center gap-4 mb-6">
              <Button onClick={startCamera} disabled={!modelo}>
                <Camera className="w-4 h-4 mr-2" />
                Activar cámara
              </Button>
              <Button
                variant="outline"
                onClick={cambiarCamara}
                disabled={!cameraActive}
              >
                <RotateCcw className="w-4 h-4 mr-2" />
                Cambiar cámara
              </Button>
            </div>

            <div className="relative mx-auto max-w-md w-full aspect-square bg-white rounded-lg overflow-hidden shadow border">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="absolute inset-0 w-full h-full object-cover"
              />
              <canvas
                ref={canvasRef}
                width={size}
                height={size}
                className="absolute inset-0 w-full h-full object-cover opacity-0"
              />
              <canvas
                ref={otherCanvasRef}
                width={32}
                height={32}
                className="hidden"
              />
              {currentPrediction && (
                <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-black bg-opacity-75 text-white py-2 px-4 rounded text-lg font-bold">
                  {currentPrediction}
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
