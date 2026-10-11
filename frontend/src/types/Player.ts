export type Player = {
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
  minutes: number
}