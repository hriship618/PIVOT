type StatBoxProps = {
  label: string
  value: string | number
}

function StatBox({
  label,
  value,
}: StatBoxProps) {
  return (
    <div className="border-l border-pivot-divider pl-3 first:border-l-0 first:pl-0">
      <p className="text-xs font-bold uppercase tracking-[0.12em] text-pivot-muted">
        {label}
      </p>

      <p className="mt-1 font-mono text-xl font-bold text-pivot-text">
        {value}
      </p>
    </div>
  )
}

export default StatBox