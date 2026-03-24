export default function ColdStartBadge({ rampActive, vectorCount = 0, rampActions = 50 }) {
  if (!rampActive) return null;

  const safeTotal = Math.max(1, rampActions);
  const progress = Math.min(100, Math.round((vectorCount / safeTotal) * 100));

  return (
    <div className="coldstart">
      <div className="coldstart-head">
        <span>Cold Start Ramp</span>
        <span>{progress}%</span>
      </div>
      <div className="coldstart-track">
        <div className="coldstart-fill" style={{ width: `${progress}%` }} />
      </div>
      <div className="coldstart-note">
        {vectorCount} / {safeTotal} actions collected before full behavior matching.
      </div>
    </div>
  );
}
