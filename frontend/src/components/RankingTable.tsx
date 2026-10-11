type Player = {
  rank: number
  name: string
  team: string
  score: number
}

type RankingTableProps = {
  players: Player[]
}

function RankingTable({ players }: RankingTableProps) {
  return (
    <div className="mt-8 overflow-hidden rounded-xl border border-slate-800">
      <table className="w-full text-left">
        <thead className="bg-slate-900 text-slate-300">
          <tr>
            <th className="px-4 py-3">Rank</th>
            <th className="px-4 py-3">Player</th>
            <th className="px-4 py-3">Team</th>
            <th className="px-4 py-3">PIVOT Score</th>
          </tr>
        </thead>

        <tbody>
          {players.map((player) => (
            <tr key={player.rank} className="border-t border-slate-800">
              <td className="px-4 py-3">{player.rank}</td>
              <td className="px-4 py-3">{player.name}</td>
              <td className="px-4 py-3 text-slate-400">{player.team}</td>
              <td className="px-4 py-3 font-semibold">
                {player.score.toFixed(2)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default RankingTable