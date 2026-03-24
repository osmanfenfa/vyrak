export default function FeatureImportance({ features = [], title = "Top Features" }) {
  const maxImportance = features.reduce(
    (largest, feature) => Math.max(largest, feature.importance || 0),
    0,
  );

  return (
    <div className="card">
      <div className="card-title">{title}</div>

      {features.length === 0 ? (
        <p className="muted">No feature data yet. Collect more actions to train the model.</p>
      ) : (
        <div className="feature-list">
          {features.map((feature, index) => {
            const ratio = maxImportance > 0 ? feature.importance / maxImportance : 0;
            const width = Math.max(6, Math.round(ratio * 100));
            const tint = 0.95 - index * 0.1;

            return (
              <div key={feature.name} className="feature-row">
                <div className="feature-row-head">
                  <span className="mono">{feature.name}</span>
                  <span className="muted">{(feature.importance * 100).toFixed(1)}%</span>
                </div>
                <div className="feature-track">
                  <div
                    className="feature-fill"
                    style={{
                      width: `${width}%`,
                      opacity: Math.max(0.35, tint),
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
