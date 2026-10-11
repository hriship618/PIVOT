import { Fragment } from 'react'
import PercentileBar from './PercentileBar'
import StatBox from './StatBox'

type Player = {
  rank: number
  name: string
  team: string
  position: string
  score: number

  pivotPercentile: number
  offensivePercentile: number
  defensivePercentile: number

  goals: number
  assists: number
  xG: number
  xA: number
  xT: number
}

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
    <div className="mt-8 overflow-hidden rounded-xl border border-slate-800">
      <table className="w-full text-left">
        <thead className="bg-slate-900 text-slate-300">
          <tr>
            <th className="px-4 py-3">Rank</th>
            <th className="px-4 py-3">Player</th>
            <th className="px-4 py-3">Team</th>
            <th className="px-4 py-3">Position</th>
            <th className="px-4 py-3">PIVOT Score</th>
          </tr>
        </thead>

        <tbody>
          {players.map((player) => (
            <Fragment key={player.rank}>
              <tr
                onClick={() => onPlayerClick(player)}
                className="cursor-pointer border-t border-slate-800 hover:bg-slate-900"
              >
                <td className="px-4 py-3">{player.rank}</td>
                <td className="px-4 py-3">{player.name}</td>
                <td className="px-4 py-3 text-slate-400">
                  {player.team}
                </td>
                <td className="px-4 py-3 text-slate-400">
                  {player.position}
                </td>
                <td className="px-4 py-3 font-semibold">
                  {player.score.toFixed(2)}
                </td>
              </tr>

              <tr>
                <td colSpan={5} className="p-0">
                  <div
                    className={`grid transition-all duration-300 ease-in-out ${
                      selectedPlayer?.rank === player.rank
                        ? 'grid-rows-[1fr] opacity-100'
                        : 'grid-rows-[0fr] opacity-0'
                    }`}
                  >
                    <div className="min-h-0 overflow-hidden">
                      <div className="bg-slate-900 px-6 py-5">

                        <div className="grid grid-cols-2 gap-8">
                          <div>
                            <h3 className="text-xl font-bold">
                              {player.name}
                            </h3>

                            <p className="mt-1 text-slate-400">
                              {player.team} · {player.position}
                            </p>

                            <p className="mt-4 text-2xl font-bold">
                              PIVOT {player.score.toFixed(2)}
                            </p>

                            <p className="text-sm text-slate-400">
                              {player.pivotPercentile}th percentile
                            </p>
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

                        <div className="mt-6 grid grid-cols-5 gap-4">
                            <StatBox label="Goals" value={player.goals} />
                            <StatBox label="Assists" value={player.assists} />
                            <StatBox label="xG" value={player.xG.toFixed(1)} />
                            <StatBox label="xA" value={player.xA.toFixed(1)} />
                            <StatBox label="xT" value={player.xT.toFixed(1)} />
                        </div>

                      </div>
                    </div>
                  </div>
                </td>
              </tr>
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default RankingTable