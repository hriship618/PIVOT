function Navbar() {
  return (
    <nav className="flex items-center justify-between px-8 py-4 border-b border-slate-800 bg-slate-950 text-white">
      <h1 className="text-2xl font-bold">PIVOT</h1>

      <div className="flex gap-6 text-sm text-slate-300">
        <span>Rankings</span>
        <span>Players</span>
        <span>About</span>
      </div>
    </nav>
  )
}

export default Navbar