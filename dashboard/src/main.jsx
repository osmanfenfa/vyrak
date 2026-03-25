import { StrictMode } from "react";
import { createRoot }  from "react-dom/client";
import "./styles/globals.css";
import App from "./App.jsx";
import AdminGate from "./AdminGate.jsx";

const pathname = window.location.pathname.replace(/\/+$/, "") || "/";
const RootComponent = pathname === "/admin-gate" ? AdminGate : App;

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <RootComponent />
  </StrictMode>
);
