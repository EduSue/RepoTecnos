import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SEMMLRN",
  description:
    "Sistema experimental de métodos de machine learning y Redes Neuronales",
  generator:
    "Sistema experimental de métodos de machine learning y Redes Neuronales",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
