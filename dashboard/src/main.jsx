import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./styles/globals.css";
import App from "./App.jsx";

// The platform panel (formerly the separate /admin-gate page) is now a tab
// inside the signed-in console; /admin-gate still opens it.
createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
