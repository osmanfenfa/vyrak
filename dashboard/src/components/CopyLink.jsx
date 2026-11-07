import { useState } from "react";

/** One-time link (invite / reset) shown to the admin to copy and send. */
export default function CopyLink({ title, link, note, onDone }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(link);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="secret-box" role="status">
      <div style={{ fontWeight: 600 }}>{title}</div>
      <code className="mono">{link}</code>
      {note && <div style={{ fontSize: 12 }}>{note}</div>}
      <div style={{ display: "flex", gap: 8 }}>
        <button type="button" className="btn-ghost btn-sm" onClick={copy}>{copied ? "Copied" : "Copy link"}</button>
        {onDone && <button type="button" className="btn-ghost btn-sm" onClick={onDone}>Done</button>}
      </div>
    </div>
  );
}
