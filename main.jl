import Random

BID_REWARD = 10
CALL_LIAR_REWARD = 100

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
    rng::Int
    actions::Vector{Action}
end

function all_actions(max_bid)
    bids = [BidAction(q, f) for q in 1:max_bid for f in 1:6]
    push!(bids, LiarAction())
    return bids
end

function legal_mask(game::Game, state::FullState)
    total_dice = sum(state.pub.dice_left)
    last_bid = state.pub.last_bid
    actions = game.actions
    mask = falses(length(actions))

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

function next_player(i, N)
    return (i % N) + 1
end

function reset(game::Game)
    N = game.num_players
    players = roll_dice(N, game.dice_per_player)
    pub = PublicInfo(nothing, nothing, 1, fill(game.dice_per_player, N))
    state = FullState(players, pub)
    obs = observe(game, state)

    return state, obs
end

function roll_dice(num_players, num_dice)
    players = Vector{PrivateInfo}(undef, num_players)
    for i in 1:num_players
        rolls = rand(1:6, num_dice)
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
    reward = 0
    if action isa BidAction
        next_pub = PublicInfo(action, player, next_player(player, game.num_players), state.pub.dice_left)
        next_state = FullState(state.players, next_pub)
        reward = BID_REWARD
    elseif action isa LiarAction
        bid_qty = state.pub.last_bid.qty
        bid_face = state.pub.last_bid.face
        bidder = state.pub.last_bidder

        count = 0

        for p in state.players
            if bid_face == 1
                count += p.dice[1]
            elseif ones_wild
                count += p.dice[1] + p.dice[bid_face]
            else
                count += p.dice[bid_face]
            end
        end

        if count < bid.qty
            state.pub.dice_left[bidder] -= 1
            return state, nothing, CALL_LIAR_REWARD, true
        else
            return 
    end

    return next_state, observe(game, next_state), BID_REWARD, false