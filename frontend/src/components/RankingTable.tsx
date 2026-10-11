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
    <div className="mt-8 overflow-x-auto rounded-xl border border-slate-800">
      <table className="w-full text-left">
        <thead className="bg-slate-900 text-slate-300">
          <tr>
            <th className="px-4 py-3">Rank</th>
            <th className="px-4 py-3">Player</th>
            <th className="px-4 py-3">Team</th>
            <th className="px-4 py-3">Position</th>
            <th className="px-4 py-3">PIVOT Score</th>
            <th className="px-4 py-3"></th>
          </tr>
        </thead>

        <tbody>
          {players.length === 0 ? (
            <tr>
              <td
                colSpan={6}
                className="px-6 py-10 text-center text-slate-400"
              >
                No players found.
              </td>
            </tr>
          ) : (
            players.map((player) => (
              <Fragment key={player.rank}>
                <tr
                  onClick={() => onPlayerClick(player)}
                  className="cursor-pointer border-t border-slate-800 hover:bg-slate-900"
                >
                  <td className="px-4 py-3">
                    {player.rank}
                  </td>

                  <td className="px-4 py-3">
                    {player.name}
                  </td>

                  <td className="px-4 py-3 text-slate-400">
                    {player.team}
                  </td>

                  <td className="px-4 py-3 text-slate-400">
                    {player.position}
                  </td>

                  <td className="px-4 py-3 font-semibold">
                    {player.score.toFixed(2)}
                  </td>

                  <td className="px-4 py-3 text-right">
                    <span
                      className={`inline-block transition-transform duration-300 ${
                        selectedPlayer?.rank === player.rank
                          ? 'rotate-180'
                          : ''
                      }`}
                    >
                      ▼
                    </span>
                  </td>
                </tr>

                <tr>
                  <td colSpan={6} className="p-0">
                    <div
                      className={`grid transition-all duration-300 ease-in-out ${
                        selectedPlayer?.rank === player.rank
                          ? 'grid-rows-[1fr] opacity-100'
                          : 'grid-rows-[0fr] opacity-0'
                      }`}
                    >
                      <div className="min-h-0 overflow-hidden">
                        <div className="bg-slate-900 px-6 py-5">
                          <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
                            <div>
                              <h3 className="text-xl font-bold">
                                {player.name}
                              </h3>

                              <p className="mt-1 text-slate-400">
                                {player.team} · {player.position}
                              </p>

                              <div className="mt-4 inline-block rounded-lg border border-slate-700 px-4 py-3">
                                <p className="text-sm text-slate-400">
                                  PIVOT Score
                                </p>

                                <p className="text-3xl font-bold">
                                  {player.score.toFixed(2)}
                                </p>

                                <p className="text-sm text-slate-400">
                                  {player.pivotPercentile}th percentile
                                </p>
                              </div>
                            </div>

                            <div className="space-y-4">
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

                          <div className="mt-6 grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-6">
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