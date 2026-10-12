import { Fragment } from 'react'

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
              PIVOT
            </th>

            <th className="border-r border-pivot-divider px-4 py-3 text-xs font-bold uppercase tracking-[0.14em]">
              Minutes
            </th>

            <th className="border-r border-pivot-divider px-4 py-3 text-xs font-bold uppercase tracking-[0.14em]">
              Matches
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

              <Fragment key={player.player_id}>

                <tr
                  onClick={() =>
                    onPlayerClick(player)
                  }
                  className="cursor-pointer border-b border-pivot-border bg-pivot-surface transition-colors hover:bg-pivot-surface-2"
                >

                  <td className="border-r border-pivot-divider px-4 py-4 font-mono text-sm">
                    {String(player.rank).padStart(
                      2,
                      '0'
                    )}
                  </td>

                  <td className="border-r border-pivot-divider px-4 py-4 font-bold">
                    {player.name}
                  </td>

                  <td className="border-r border-pivot-divider px-4 py-4 font-mono text-lg font-bold">
                    {player.prototype_pivot >= 0
                      ? '+'
                      : ''}
                    {player.prototype_pivot.toFixed(
                      2
                    )}
                  </td>

                  <td className="border-r border-pivot-divider px-4 py-4 font-mono text-sm text-pivot-muted">
                    {player.minutes_sampled.toFixed(
                      1
                    )}
                  </td>

                  <td className="border-r border-pivot-divider px-4 py-4 font-mono text-sm text-pivot-muted">
                    {player.matches}
                  </td>

                  <td className="px-4 py-4 text-right">

                    <span
                      className={`inline-block text-sm transition-transform duration-300 ${
                        selectedPlayer?.player_id
                        === player.player_id
                          ? 'rotate-180'
                          : ''
                      }`}
                    >
                      ▼
                    </span>

                  </td>

                </tr>

                <tr className="bg-pivot-surface">

                  <td
                    colSpan={6}
                    className="p-0"
                  >

                    <div
                      className={`grid transition-all duration-300 ease-in-out ${
                        selectedPlayer?.player_id
                        === player.player_id
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
                                DFL tracking sample
                              </p>

                              <div className="mt-5 inline-block border-2 border-pivot-border bg-pivot-surface-2 px-5 py-4">

                                <p className="text-xs font-bold uppercase tracking-[0.14em] text-pivot-muted">
                                  Prototype PIVOT
                                </p>

                                <p className="mt-1 font-mono text-4xl font-bold text-pivot-text">

                                  {player.prototype_pivot >= 0
                                    ? '+'
                                    : ''}

                                  {player.prototype_pivot.toFixed(
                                    2
                                  )}

                                </p>

                              </div>

                            </div>

                            <div>

                              <p className="mb-4 text-xs font-bold uppercase tracking-[0.16em] text-pivot-muted">
                                Model Output
                              </p>

                              <p className="max-w-md text-sm leading-6 text-pivot-muted">
                                Spatial value measures the average change
                                in predicted near-future xG differential
                                between the observed game state and a
                                teammate-replacement counterfactual.
                              </p>

                              <div className="mt-5 border border-pivot-divider bg-pivot-surface-2 px-4 py-4">

                                <p className="text-xs font-bold uppercase tracking-[0.14em] text-pivot-muted">
                                  Raw Spatial Value
                                </p>

                                <p className="mt-2 font-mono text-xl font-bold text-pivot-text">
                                  {player.spatial_value >= 0
                                    ? '+'
                                    : ''}

                                  {player.spatial_value.toFixed(
                                    6
                                  )}
                                </p>

                              </div>

                            </div>

                          </div>

                          <div className="mt-8 border-t border-pivot-divider pt-6">

                            <p className="mb-4 text-xs font-bold uppercase tracking-[0.16em] text-pivot-muted">
                              Sample
                            </p>

                            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">

                              <StatBox
                                label="Minutes"
                                value={
                                  player.minutes_sampled.toFixed(
                                    1
                                  )
                                }
                              />

                              <StatBox
                                label="Matches"
                                value={player.matches}
                              />

                              <StatBox
                                label="Rank"
                                value={`#${player.rank}`}
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