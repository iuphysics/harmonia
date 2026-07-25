import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import Home from "../app/page";
import "../app/globals.css";

const refreshOnStaleDynamicImport = (error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  if (!message.toLowerCase().includes("dynamically imported module")) return;
  const url = new URL(window.location.href);
  if (url.searchParams.has("reload")) return;
  url.searchParams.set("reload", Date.now().toString());
  window.location.replace(url);
};

window.addEventListener("error", (event) => {
  refreshOnStaleDynamicImport(event.error ?? event.message);
});

window.addEventListener("unhandledrejection", (event) => {
  refreshOnStaleDynamicImport(event.reason);
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Home />
  </StrictMode>,
);
