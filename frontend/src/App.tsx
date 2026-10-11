import { Route, Routes } from 'react-router-dom'

import Navbar from './components/Navbar'
import RankingsPage from './pages/RankingsPage'
import AboutPage from './pages/AboutPage'
import MethodologyPage from './pages/MethodologyPage'

function App() {
  return (
    <div className="min-h-screen bg-pivot-bg text-pivot-text">
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