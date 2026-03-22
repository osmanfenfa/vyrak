/**
 * FeatureImportance — horizontal bar chart of top feature importances.
 *
 * Props:
 *   features  {Array}  [{name, importance}]  (from session summary top_features)
 *   title     {string}
 */

export default function FeatureImportance({ features = [], title = "Top Features" }) {
  const max = features.reduce((m, f) => Math.max(m, f.importance), 0);

  return (
    <div className="card">
      <div className="card-title">{title}</div>

      {features.length === 0 ? (
        <p style={{ color: "var(--text-muted)", fontSize: 12 }}>
          No feature data — classifier must be trained.
        </p>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {features.map((f, i) => {
            const pct = max > 0 ? (f.importance / max) * 100 : 0;
            // Colour gradient: top feature = teal, lower = lighter
            const opacity = 0.4 + (features.length - i) / features.length * 0.6;
            return (
              <div key={f.name}>
                <div style={{
                  display: "flex", justifyContent: "space-between",
                  fontSize: 12, marginBottom: 3,
                }}>
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>{f.name}</span>
                  <span style={{ color: "var(--text-muted)", fontSize: 11 }}>
                    {(f.importance * 100).toFixed(1)}%
                  </span>
                </div>
                <div style={{
                  height: 8,
                  background: "#E5E3DC",
                  borderRadius: 99,
                  overflow: "hidden",
                }}>
                  <div style={{
                    height: "100%",
                    width: `${pct}%`,
                    background: `rgba(29, 158, 117, ${opacity})`,
                    borderRadius: 99,
                    transition: "width 0.5s ease",
                  }} />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
