import { NavLink } from 'react-router-dom'

function Navbar() {
  const linkStyle = ({ isActive }: { isActive: boolean }) =>
    `border-b-2 pb-1 text-xs font-bold uppercase tracking-[0.14em] transition-colors ${
      isActive
        ? 'border-pivot-border text-pivot-text'
        : 'border-transparent text-pivot-muted hover:text-pivot-text'
    }`

  return (
    <nav className="border-b-2 border-pivot-border bg-pivot-bg px-8 py-5">
      <div className="mx-auto flex max-w-7xl items-center justify-between">
        <NavLink
          to="/"
          className="text-2xl font-black uppercase tracking-[-0.03em] text-pivot-text"
        >
          PIVOT
        </NavLink>

        <div className="flex gap-8">
          <NavLink to="/" end className={linkStyle}>
            Rankings
          </NavLink>

          <NavLink to="/about" className={linkStyle}>
            About
          </NavLink>

          <NavLink to="/methodology" className={linkStyle}>
            Methodology
          </NavLink>
        </div>
      </div>
    </nav>
  )
}

export default Navbar