import { Routes, Route } from 'react-router-dom'

import Navbar from './components/Navbar'
import RankingsPage from './pages/RankingsPage'
import AboutPage from './pages/AboutPage'
import MethodologyPage from './pages/MethodologyPage'

function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <Navbar />

      <Routes>
        <Route path="/" element={<RankingsPage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="/methodology" element={<MethodologyPage />} />
      </Routes>
    </div>
  )
}

export default App