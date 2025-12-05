include("main.jl")
include("policies.jl")
using POMDPs
using POMDPTools: Deterministic
import Random: AbstractRNG

mutable struct LiarsDicePOMDP <: POMDP{FullState, Action, Observation}
    num_players::Int
    dice_per_player::Int
    ones_wild::Bool
    ego_id::Int
    actions::Vector{Action}
    gamma::Float64
end

function LiarsDicePOMDP(num_players::Int=2, dice_per_player::Int=2, 
                        ones_wild::Bool=true, ego_id::Int=1, gamma::Float64=0.99)
    max_bid = num_players * dice_per_player
    actions = all_actions(max_bid)
    return LiarsDicePOMDP(num_players, dice_per_player, ones_wild, ego_id, actions, gamma)
end

POMDPs.actions(pomdp::LiarsDicePOMDP) = pomdp.actions
POMDPs.actionindex(pomdp::LiarsDicePOMDP, a::Action) = findfirst(==(a), pomdp.actions)
POMDPs.discount(pomdp::LiarsDicePOMDP) = pomdp.gamma

function game_from_pomdp(pomdp::LiarsDicePOMDP)
    return Game(pomdp.num_players, pomdp.dice_per_player, pomdp.ones_wild, 
               pomdp.ego_id, pomdp.actions)
end

function POMDPs.isterminal(pomdp::LiarsDicePOMDP, s::FullState)
    dl = s.pub.dice_left
    N = pomdp.num_players
    if dl[pomdp.ego_id] == 0
        return true
    end
    if all(i == pomdp.ego_id || dl[i] == 0 for i in 1:N)
        return true
    end
    
    return false
end

function POMDPs.initialstate(pomdp::LiarsDicePOMDP)
    game = game_from_pomdp(pomdp)
    state, obs = reset(game)
    return Deterministic(state)
end

function POMDPs.initialobs(pomdp::LiarsDicePOMDP, s::FullState)
    game = game_from_pomdp(pomdp)
    return observe(game, s)
end

function POMDPs.gen(pomdp::LiarsDicePOMDP, s::FullState, a::Action, rng::AbstractRNG)
    game = game_from_pomdp(pomdp)

    state = s
    total_r = 0.0
    done = false

    state, obs, r, done = step(game, state, a)
    total_r += r

    while !done && state.pub.turn != pomdp.ego_id
        opp_action, _ = random_policy(game, obs)

        state, obs, r, done = step(game, state, opp_action)
        total_r += r
    end

    return (sp = state, o = obs, r = total_r)
end
