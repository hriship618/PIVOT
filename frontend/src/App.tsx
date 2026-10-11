import { useState } from 'react'

import Navbar from './components/Navbar'
import RankingTable from './components/RankingTable'

const players = [
  {
    rank: 1,
    name: 'Lionel Messi',
    team: 'Inter Miami',
    position: 'FW',
    score: 2.41,

    pivotPercentile: 99,
    offensivePercentile: 99,
    defensivePercentile: 28,

    goals: 21,
    assists: 14,
    xG: 18.7,
    xA: 12.9,
    xT: 16.4,
  },
  {
    rank: 2,
    name: 'Kylian Mbappe',
    team: 'Real Madrid',
    position: 'FW',
    score: 2.18,

    pivotPercentile: 97,
    offensivePercentile: 98,
    defensivePercentile: 34,

    goals: 25,
    assists: 8,
    xG: 22.1,
    xA: 7.6,
    xT: 14.8,
  },
  {
    rank: 3,
    name: 'Rodri',
    team: 'Manchester City',
    position: 'MF',
    score: 1.96,

    pivotPercentile: 95,
    offensivePercentile: 88,
    defensivePercentile: 97,

    goals: 6,
    assists: 9,
    xG: 4.8,
    xA: 8.7,
    xT: 13.2,
  },
  {
    rank: 4,
    name: 'Jude Bellingham',
    team: 'Real Madrid',
    position: 'MF',
    score: 1.82,

    pivotPercentile: 92,
    offensivePercentile: 94,
    defensivePercentile: 81,

    goals: 15,
    assists: 10,
    xG: 13.4,
    xA: 8.9,
    xT: 12.5,
  },
  {
    rank: 5,
    name: 'Virgil van Dijk',
    team: 'Liverpool',
    position: 'DF',
    score: 1.55,

    pivotPercentile: 88,
    offensivePercentile: 54,
    defensivePercentile: 98,

    goals: 4,
    assists: 2,
    xG: 3.1,
    xA: 1.7,
    xT: 4.3,
  },
  {
    rank: 6,
    name: 'Trent Alexander-Arnold',
    team: 'Liverpool',
    position: 'DF',
    score: 1.39,

    pivotPercentile: 84,
    offensivePercentile: 93,
    defensivePercentile: 61,

    goals: 3,
    assists: 12,
    xG: 2.4,
    xA: 11.6,
    xT: 15.1,
  },
]

const teams = [...new Set(players.map((player) => player.team))]
const positions = [...new Set(players.map((player) => player.position))]
function App() {

  const [search, setSearch] = useState('')
  const [sortOrder, setSortOrder] = useState('desc')
  const [teamFilter, setTeamFilter] = useState('All')
  const [positionFilter, setPositionFilter] = useState('All')  
  const [selectedPlayer, setSelectedPlayer] = useState<(typeof players)[number] | null>(null)

  const filteredPlayers = players.filter((player) => {
    const matchesSearch = player.name
      .toLowerCase()
      .includes(search.toLowerCase())

    const matchesTeam =
      teamFilter === 'All' || player.team === teamFilter

    const matchesPosition =
      positionFilter === 'All' || player.position === positionFilter

    return matchesSearch && matchesTeam && matchesPosition
  })

  const sortedPlayers = [...filteredPlayers].sort((a, b) => {
    if (sortOrder === 'desc') {
      return b.score - a.score
    }

    return a.score - b.score
  })


  return(
    <div className="min-h-screen bg-slate-950 text-white">
      
      <Navbar />

      <main className="px-8 py-10">
        <h2 className="text-4xl font-bold">Player Rankings</h2>
        <p className="mt-2 text-slate-400">
          Explore PIVOT player impact scores.
        </p>
        <div className="mt-6 flex gap-4">
          <input
            type="text"
            placeholder="Search players..."
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            className="w-full max-w-md rounded-lg border border-slate-700 bg-slate-900 px-4 py-2 text-white outline-none"
          />
          <select
            value={sortOrder}
            onChange={(event) => setSortOrder(event.target.value)}
            className="rounded-lg border border-slate-700 bg-slate-900 px-4 py-2 text-white"
          >
            <option value="desc">PIVOT: High to Low</option>
            <option value="asc">PIVOT: Low to High</option>
          </select>
          <select
            value={teamFilter}
            onChange={(event) => setTeamFilter(event.target.value)}
            className="rounded-lg border border-slate-700 bg-slate-900 px-4 py-2 text-white"
          >
            <option value="All">All Teams</option>

            {teams.map((team) => (
              <option key={team} value={team}>
                {team}
              </option>
            ))}
          </select>
          <select
            value={positionFilter}
            onChange={(event) => setPositionFilter(event.target.value)}
            className="rounded-lg border border-slate-700 bg-slate-900 px-4 py-2 text-white"
          >
            <option value="All">All Positions</option>

            {positions.map((position) => (
              <option key={position} value={position}>
                {position}
              </option>
            ))}
          </select>
                    
        </div>


        <RankingTable
          players={sortedPlayers}
          selectedPlayer={selectedPlayer}
          onPlayerClick={(player) =>
            setSelectedPlayer(
              selectedPlayer?.rank === player.rank ? null : player
            )
          }
        />
      </main>
    </div>
  )
}

export default App