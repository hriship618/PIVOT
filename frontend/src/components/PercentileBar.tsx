type PercentileBarProps = {
  label: string
  percentile: number
}

function PercentileBar({ label, percentile }: PercentileBarProps) {
  let barColor = 'bg-slate-500'

  if (percentile >= 90) {
    barColor = 'bg-green-500'
  } else if (percentile >= 75) {
    barColor = 'bg-lime-500'
  } else if (percentile >= 40) {
    barColor = 'bg-yellow-500'
  } else if (percentile >= 20) {
    barColor = 'bg-orange-500'
  } else {
    barColor = 'bg-red-500'
  }

  return (
    <div>
      <div className="mb-1 flex justify-between text-sm">
        <span>{label}</span>
        <span>{percentile}th</span>
      </div>

      <div className="h-2 overflow-hidden rounded-full bg-slate-800">
        <div
          className={`h-full ${barColor}`}
          style={{ width: `${percentile}%` }}
        />
      </div>
    </div>
  )
}

export default PercentileBar