import { useState } from 'react'

import Navbar from './components/Navbar'
import RankingTable from './components/RankingTable'

const players = [
  { rank: 1, name: 'Lionel Messi', team: 'Inter Miami', score: 2.41 },
  { rank: 2, name: 'Kylian Mbappe', team: 'Real Madrid', score: 2.18 },
  { rank: 3, name: 'Rodri', team: 'Manchester City', score: 1.96 },
]

function App() {

  const [search, setSearch] = useState('')

  const filteredPlayers = players.filter((player) =>
    player.name.toLowerCase().includes(search.toLowerCase())
  )


  return(
    <div className="min-h-screen bg-slate-950 text-white">
      
      <Navbar />

      <main className="px-8 py-10">
        <h2 className="text-4xl font-bold">Player Rankings</h2>
        <p className="mt-2 text-slate-400">
          Explore PIVOT player impact scores.
        </p>
        <input
          type="text"
          placeholder="Search players..."
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          className="mt-6 w-full max-w-md rounded-lg border border-slate-700 bg-slate-900 px-4 py-2 text-white outline-none"
        />
        <RankingTable players={filteredPlayers} />
      </main>
    </div>
  )
}

export default App