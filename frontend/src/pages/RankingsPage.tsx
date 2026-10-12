import { useState } from 'react'

import RankingTable from '../components/RankingTable'
import rankingsData from '../data/prototype_rankings.json'

import type { Player } from '../types/Player'


const MIN_MINUTES = 45

const players = (rankingsData as Player[])
  .filter((player) => player.minutes_sampled >= MIN_MINUTES)
  .sort((a, b) => b.prototype_pivot - a.prototype_pivot)
  .map((player, index) => ({
    ...player,
    rank: index + 1,
  }))


function RankingsPage() {
  const [search, setSearch] = useState('')
  const [sortOrder, setSortOrder] = useState('desc')
  const [selectedPlayer, setSelectedPlayer] =
    useState<Player | null>(null)

  function clearFilters() {
    setSearch('')
    setSortOrder('desc')
    setSelectedPlayer(null)
  }

  const filteredPlayers = players.filter((player) =>
    player.name
      .toLowerCase()
      .includes(search.toLowerCase())
  )

  const sortedPlayers = [...filteredPlayers].sort(
    (a, b) => {
      if (sortOrder === 'desc') {
        return (
          b.prototype_pivot
          - a.prototype_pivot
        )
      }

      return (
        a.prototype_pivot
        - b.prototype_pivot
      )
    }
  )

  const hasActiveFilters =
    search !== ''
    || sortOrder !== 'desc'

  return (
    <main className="mx-auto max-w-7xl px-6 py-12 md:px-8">

      <div className="border-b-2 border-pivot-border pb-6">

        <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">

          <div>
            <h1 className="text-5xl font-bold tracking-tight text-pivot-text">
              PIVOT Rankings
            </h1>

            <p className="mt-3 max-w-2xl text-sm leading-6 text-pivot-muted">
              Experimental player rankings derived from continuous
              tracking data and learned spatial value.
            </p>
          </div>

          <div className="border-2 border-pivot-border bg-pivot-surface-2 px-4 py-3 text-xs font-bold uppercase tracking-[0.14em] text-pivot-muted">
            Prototype · DFL tracking sample
          </div>

        </div>

      </div>

      <div className="mt-6 border-l-4 border-pivot-border bg-pivot-surface px-5 py-4">

        <p className="text-sm leading-6 text-pivot-muted">
          These are preliminary spatial-value estimates from a limited
          tracking sample. The displayed PIVOT score is standardized
          relative to players in the current dataset and is not yet
          xGD added per 90.
        </p>

      </div>

      <div className="mt-8 flex flex-col gap-3 md:flex-row md:flex-wrap">

        <input
          type="text"
          placeholder="Search players..."
          value={search}
          onChange={(event) =>
            setSearch(event.target.value)
          }
          className="w-full border-2 border-pivot-border bg-pivot-surface px-4 py-3 text-pivot-text outline-none placeholder:text-pivot-muted focus:bg-pivot-surface-2 md:max-w-md"
        />

        <div className="relative">

          <select
            value={sortOrder}
            onChange={(event) =>
              setSortOrder(event.target.value)
            }
            className="appearance-none border-2 border-pivot-border bg-pivot-surface py-3 pl-4 pr-10 text-pivot-text outline-none"
          >
            <option value="desc">
              PIVOT: High to Low
            </option>

            <option value="asc">
              PIVOT: Low to High
            </option>
          </select>

          <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-xs text-pivot-text">
            ▼
          </span>

        </div>

        {hasActiveFilters && (
          <button
            onClick={clearFilters}
            className="border-2 border-pivot-border bg-pivot-surface px-4 py-3 text-pivot-text transition-colors hover:bg-pivot-surface-2"
          >
            Clear Filters
          </button>
        )}

      </div>

      <div className="mt-5 flex flex-wrap gap-x-6 gap-y-2 text-sm font-medium text-pivot-muted">

        <span>
          {sortedPlayers.length}{' '}
          {sortedPlayers.length === 1
            ? 'player'
            : 'players'}
        </span>

        <span>
          Minimum sample: {MIN_MINUTES} minutes
        </span>

      </div>

      <RankingTable
        players={sortedPlayers}
        selectedPlayer={selectedPlayer}
        onPlayerClick={(player) =>
          setSelectedPlayer(
            selectedPlayer?.player_id
              === player.player_id
              ? null
              : player
          )
        }
      />

    </main>
  )
}


export default RankingsPage