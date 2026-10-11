type PercentileBarProps = {
  label: string
  percentile: number
}

function PercentileBar({
  label,
  percentile,
}: PercentileBarProps) {
  let barColor = 'bg-pivot-yellow'

  if (percentile >= 90) {
    barColor = 'bg-pivot-green'
  } else if (percentile >= 70) {
    barColor = 'bg-pivot-lime'
  } else if (percentile >= 40) {
    barColor = 'bg-pivot-yellow'
  } else if (percentile >= 20) {
    barColor = 'bg-pivot-orange'
  } else {
    barColor = 'bg-pivot-red'
  }

  return (
    <div>
      <div className="mb-2 flex items-end justify-between">
        <span className="text-xs font-bold uppercase tracking-[0.12em] text-pivot-muted">
          {label}
        </span>

        <span className="font-mono text-sm font-bold text-pivot-text">
          {percentile}
        </span>
      </div>

      <div className="h-3 border border-pivot-border bg-pivot-bg">
        <div
          className={`h-full ${barColor}`}
          style={{ width: `${percentile}%` }}
        />
      </div>
    </div>
  )
}

export default PercentileBar