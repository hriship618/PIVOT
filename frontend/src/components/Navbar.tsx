import { Link } from 'react-router-dom'

function Navbar() {
  return (
    <nav className="flex items-center justify-between border-b border-slate-800 bg-slate-950 px-8 py-4 text-white">
      <Link to="/" className="text-2xl font-bold">
        PIVOT
      </Link>

      <div className="flex gap-6 text-sm text-slate-300">
        <Link to="/" className="hover:text-white">
          Rankings
        </Link>

        <Link to="/about" className="hover:text-white">
          About
        </Link>

        <Link to="/methodology" className="hover:text-white">
          Methodology
        </Link>
      </div>
    </nav>
  )
}

export default Navbar