import Random

BID_REWARD = 0.0
LOSS_REWARD = -1.0
WIN_REWARD = 1.0

abstract type Action end

struct BidAction <: Action
    qty::Int
    face::Int
end

struct LiarAction <: Action
end

struct Observation
    dice::NTuple{6, Int}
    last_bid::Union{Nothing, BidAction}
    dice_left::Vector{Int}
    legal_actions::Vector{Bool}
end

struct PrivateInfo
    dice::NTuple{6, Int}
end

struct PublicInfo
    last_bid::Union{Nothing, BidAction}
    last_bidder::Union{Nothing, Int}
    turn::Int
    dice_left::Vector{Int}
end

struct FullState
    players::Vector{PrivateInfo}
    pub::PublicInfo
end

struct Game
    num_players::Int
    dice_per_player::Int
    ones_wild::Bool
    ego_id::Int
    actions::Vector{Action}
end

function all_actions(max_bid)
    bids = Action[BidAction(q, f) for q in 1:max_bid for f in 1:6]
    push!(bids, LiarAction())
    return bids
end

function legal_mask(game::Game, state::FullState)
    total_dice = sum(state.pub.dice_left)
    last_bid = state.pub.last_bid
    actions = game.actions
    mask = fill(false, length(actions))

    if last_bid === nothing
        for (i, a) in enumerate(actions)
            if a isa BidAction && a.qty <= total_dice
                mask[i] = true
            end
        end
    else
        for (i, a) in enumerate(actions)
            if a isa BidAction && a.qty <= total_dice && greater(a, last_bid)
                mask[i] = true
            elseif a isa LiarAction
                mask[i] = true
            end
        end
    end
    return mask
end

function greater(a::BidAction, b::BidAction)
    if a.qty > b.qty
        return true
    elseif a.qty == b.qty && a.face > b.face
        return true
    else 
        return false
    end
end

function next_player(i, dice_left::Vector{Int})
    N = length(dice_left)
    next = (i % N) + 1
    
    while dice_left[next] == 0
        next = (next % N) + 1
    end
    return next
end

function create_game(num_players, dice_per_player, ones_wild, ego_id)
    game = Game(num_players, dice_per_player, ones_wild, ego_id, all_actions(num_players * dice_per_player))
    return game
end

function reset(game::Game)
    state, obs = new_round(game, fill(game.dice_per_player, game.num_players), 1)

    return state, obs
end

function new_round(game::Game, new_dice_left::Vector{Int}, turn::Int)
    N = game.num_players
    players = roll_dice(new_dice_left)
    pub = PublicInfo(nothing, nothing, turn, new_dice_left)
    state = FullState(players, pub)
    obs = observe(game, state)

    return state, obs
end

function roll_dice(dice_left::Vector{Int})
    num_players = length(dice_left)
    players = Vector{PrivateInfo}(undef, num_players)
    for (i, n) in enumerate(dice_left)
        rolls = rand(1:6, n)
        players[i] = PrivateInfo(ntuple(i -> count(==(i), rolls), 6))
    end
    return players
end

function observe(game::Game, state::FullState)
    player = state.players[state.pub.turn]
    obs = Observation(player.dice, state.pub.last_bid, state.pub.dice_left, legal_mask(game, state))
    return obs
end

function step(game::Game, state::FullState, action::Action)
    player = state.pub.turn
    reward = 0.0
    N = game.num_players
    if action isa BidAction
        if state.pub.last_bid !== nothing && !greater(action, state.pub.last_bid)
            error("Current bid not greater than last bid")
        end
            
        next_pub = PublicInfo(action, player, next_player(player, state.pub.dice_left), state.pub.dice_left)
        next_state = FullState(state.players, next_pub)
        next_obs = observe(game, next_state)
        reward = BID_REWARD
        game_over = false
    elseif action isa LiarAction
        bid = state.pub.last_bid
        if bid === nothing
            error("LiarAction with no last bid")
        end
        bidder = state.pub.last_bidder
        caller = player

        count = 0
        for p in state.players
            if bid.face == 1
                count += p.dice[1]
            elseif game.ones_wild
                count += p.dice[1] + p.dice[bid.face]
            else
                count += p.dice[bid.face]
            end
        end

        if count < bid.qty
            winner = caller
            loser = bidder
        else
            winner = bidder
            loser = caller
        end

        if loser != game.ego_id
            reward = 1.0
        else
            reward = -1.0
        end

        new_dice_left = copy(state.pub.dice_left)
        new_dice_left[loser] = max(new_dice_left[loser] - 1, 0)
        if new_dice_left[game.ego_id] == 0
            game_over = true
        elseif all(i == game.ego_id || new_dice_left[i] == 0 for i in 1:N)
            game_over = true
        else
            game_over = false
        end

        if game_over == true
            terminal_pub = PublicInfo(bid, bidder, player, new_dice_left)
            terminal_state = FullState(state.players, terminal_pub)
            terminal_obs = observe(game, terminal_state)

            return terminal_state, terminal_obs, reward, true
        end

        next_state, next_obs = new_round(game, new_dice_left, next_player(loser, new_dice_left))
    end
    
    return next_state, next_obs, reward, game_over
end

# below this is just an example of how to run it

# function random_legal_action(game::Game, obs::Observation)
#     idxs = findall(obs.legal_actions)
#     i = rand(idxs)
#     return game.actions[i], i
# end

# function run_episode(game::Game; verbose=false)
#     state, obs = reset(game)
#     total_reward = 0.0
#     step_num = 0

#     while true
#         step_num += 1
#         action, a_idx = random_legal_action(game, obs)

#         verbose && println("Step $step_num | turn=$(state.pub.turn) | action=$action")

#         next_state, next_obs, reward, done = step(game, state, action)
#         total_reward += reward

#         if done
#             verbose && println("Episode finished with total_reward = $total_reward")
#             return total_reward
#         end

#         state = next_state
#         obs   = next_obs
#     end
# end

# # Example: 2-player, 2 dice each, ones wild, ego is player 1
# game = create_game(2, 2, true, 1)  # or use start_game after you fix it
# for ep in 1:5
#     println("Episode $ep reward = ", run_episode(game; verbose=true))
# end
