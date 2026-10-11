import { Fragment } from 'react'
import PercentileBar from './PercentileBar'
import StatBox from './StatBox'
import type { Player } from '../types/Player'

type RankingTableProps = {
  players: Player[]
  selectedPlayer: Player | null
  onPlayerClick: (player: Player) => void
}

function RankingTable({
  players,
  selectedPlayer,
  onPlayerClick,
}: RankingTableProps) {
  return (
    <div className="mt-8 overflow-x-auto border-2 border-pivot-border bg-pivot-surface">
      <table className="w-full border-collapse text-left">
        <thead className="bg-pivot-surface-2 text-pivot-text">
          <tr className="border-b-2 border-pivot-border">
            <th className="border-r border-pivot-divider px-4 py-3 text-xs font-bold uppercase tracking-[0.14em]">
              Rank
            </th>

            <th className="border-r border-pivot-divider px-4 py-3 text-xs font-bold uppercase tracking-[0.14em]">
              Player
            </th>

            <th className="border-r border-pivot-divider px-4 py-3 text-xs font-bold uppercase tracking-[0.14em]">
              Team
            </th>

            <th className="border-r border-pivot-divider px-4 py-3 text-xs font-bold uppercase tracking-[0.14em]">
              Position
            </th>

            <th className="border-r border-pivot-divider px-4 py-3 text-xs font-bold uppercase tracking-[0.14em]">
              PIVOT Score
            </th>

            <th className="px-4 py-3"></th>
          </tr>
        </thead>

        <tbody>
          {players.length === 0 ? (
            <tr className="bg-pivot-surface">
              <td
                colSpan={6}
                className="px-6 py-12 text-center text-pivot-muted"
              >
                No players found.
              </td>
            </tr>
          ) : (
            players.map((player) => (
              <Fragment key={player.rank}>
                <tr
                  onClick={() => onPlayerClick(player)}
                  className="cursor-pointer border-b border-pivot-border bg-pivot-surface transition-colors hover:bg-pivot-surface-2"
                >
                  <td className="border-r border-pivot-divider px-4 py-4 font-mono text-sm">
                    {String(player.rank).padStart(2, '0')}
                  </td>

                  <td className="border-r border-pivot-divider px-4 py-4 font-bold">
                    {player.name}
                  </td>

                  <td className="border-r border-pivot-divider px-4 py-4 text-pivot-muted">
                    {player.team}
                  </td>

                  <td className="border-r border-pivot-divider px-4 py-4 text-pivot-muted">
                    {player.position}
                  </td>

                  <td className="border-r border-pivot-divider px-4 py-4 font-mono text-lg font-bold">
                    {player.score.toFixed(2)}
                  </td>

                  <td className="px-4 py-4 text-right">
                    <span
                      className={`inline-block text-sm transition-transform duration-300 ${
                        selectedPlayer?.rank === player.rank
                          ? 'rotate-180'
                          : ''
                      }`}
                    >
                      ▼
                    </span>
                  </td>
                </tr>

                <tr className="bg-pivot-surface">
                  <td colSpan={6} className="p-0">
                    <div
                      className={`grid transition-all duration-300 ease-in-out ${
                        selectedPlayer?.rank === player.rank
                          ? 'grid-rows-[1fr] opacity-100'
                          : 'grid-rows-[0fr] opacity-0'
                      }`}
                    >
                      <div className="min-h-0 overflow-hidden">
                        <div className="border-b border-pivot-border bg-pivot-surface px-6 py-6">
                          <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
                            <div>
                              <p className="text-xs font-bold uppercase tracking-[0.16em] text-pivot-muted">
                                Player Profile
                              </p>

                              <h3 className="mt-2 text-2xl font-bold text-pivot-text">
                                {player.name}
                              </h3>

                              <p className="mt-1 text-sm text-pivot-muted">
                                {player.team} · {player.position}
                              </p>

                              <div className="mt-5 inline-block border-2 border-pivot-border bg-pivot-surface-2 px-5 py-4">
                                <p className="text-xs font-bold uppercase tracking-[0.14em] text-pivot-muted">
                                  PIVOT Score
                                </p>

                                <p className="mt-1 font-mono text-4xl font-bold text-pivot-text">
                                  {player.score.toFixed(2)}
                                </p>

                                <p className="mt-1 text-sm text-pivot-muted">
                                  {player.pivotPercentile}th percentile
                                </p>
                              </div>
                            </div>

                            <div>
                              <p className="mb-4 text-xs font-bold uppercase tracking-[0.16em] text-pivot-muted">
                                Impact Profile
                              </p>

                              <div className="space-y-5">
                                <PercentileBar
                                  label="Overall PIVOT"
                                  percentile={player.pivotPercentile}
                                />

                                <PercentileBar
                                  label="Offensive Impact"
                                  percentile={player.offensivePercentile}
                                />

                                <PercentileBar
                                  label="Defensive Impact"
                                  percentile={player.defensivePercentile}
                                />
                              </div>
                            </div>
                          </div>

                          <div className="mt-8 border-t border-pivot-divider pt-6">
                            <p className="mb-4 text-xs font-bold uppercase tracking-[0.16em] text-pivot-muted">
                              Production
                            </p>

                            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-6">
                              <StatBox
                                label="Goals"
                                value={player.goals}
                              />

                              <StatBox
                                label="Assists"
                                value={player.assists}
                              />

                              <StatBox
                                label="xG"
                                value={player.xG.toFixed(1)}
                              />

                              <StatBox
                                label="xA"
                                value={player.xA.toFixed(1)}
                              />

                              <StatBox
                                label="xT"
                                value={player.xT.toFixed(1)}
                              />

                              <StatBox
                                label="Minutes"
                                value={player.minutes}
                              />
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </td>
                </tr>
              </Fragment>
            ))
          )}
        </tbody>
      </table>
    </div>
  )
}

export default RankingTable