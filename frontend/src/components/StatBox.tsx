type StatBoxProps = {
  label: string
  value: string | number
}

function StatBox({ label, value }: StatBoxProps) {
  return (
    <div>
      <p className="text-sm text-slate-400">{label}</p>
      <p className="text-lg font-semibold">{value}</p>
    </div>
  )
}

export default StatBox