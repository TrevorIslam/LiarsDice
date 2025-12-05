include("main.jl")
include("liars_dice_pomdp.jl")
include("policies.jl")
using POMDPs
using Random
using Flux
using Optimisers
using BSON: @save, @load

const MAX_PLAYERS = 5
const MAX_DICE_PER_PLAYER = 5
const MAX_TOTAL_DICE = MAX_PLAYERS * MAX_DICE_PER_PLAYER

struct Transition
    obs::Observation
    act::Action
    reward::Float32
    next_obs::Observation
    done::Bool
end

mutable struct ReplayBuffer
    buffer::Vector{Transition}
    capacity::Int
    size::Int
    pos::Int
end

function ReplayBuffer(capacity::Int)
    return ReplayBuffer(Vector{Transition}(undef, capacity), capacity, 0, 1)
end

function add!(rb::ReplayBuffer, tr::Transition)
    rb.buffer[rb.pos] = tr
    rb.pos += 1
    if rb.pos > rb.capacity
        rb.pos = 1
    end
    rb.size = min(rb.size + 1, rb.capacity)
    return nothing
end

function sample(rb::ReplayBuffer, batch_size::Int, rng::AbstractRNG)
    @assert rb.size >= batch_size "Not enough samples in replay buffer"
    idxs = rand(rng, 1:rb.size, batch_size)
    return rb.buffer[idxs]
end

pomdp = LiarsDicePOMDP(3, 5)

input_dim = 30
Q_train = Chain(
    Dense(input_dim, 32, relu),
    Dense(32, 32, relu),
    Dense(32, 1)
)

const REPLAY_CAPACITY = 10_000
replay_buffer = ReplayBuffer(REPLAY_CAPACITY)

const BATCH_SIZE = 32
const GAMMA = 0.99f0
const TARGET_UPDATE_FREQ = 1000
const TRAIN_EVERY = 10

Q_target = deepcopy(Q_train)
opt = Optimisers.Adam(1f-3)
opt_state = Optimisers.setup(opt, Q_train)
global_step = 0

function train_step!(rng::AbstractRNG)
    if replay_buffer.size < BATCH_SIZE
        return
    end

    batch = sample(replay_buffer, BATCH_SIZE, rng)

    global Q_train, Q_target, global_step, opt_state

    loss, grads = Flux.withgradient(Q_train) do model
        total_loss = 0.0f0

        for tr in batch
            o = tr.obs
            a = tr.act
            r = tr.reward
            o_next = tr.next_obs
            done = tr.done

            x_obs = encode_obs(pomdp, o)
            x_act = encode_action(a)
            q_pred = model(vcat(x_obs, x_act))[1]

            target = 0.0f0
            if done
                target = r 
            else
                all_actions = actions(pomdp)
                legal_idxs = findall(o_next.legal_actions)
                max_q = -Inf32

                x_obs_next = encode_obs(pomdp, o_next)
                for idx in legal_idxs
                    a2 = all_actions[idx]
                    x_act_next = encode_action(a2)
                    q_val = Q_target(vcat(x_obs_next, x_act_next))[1]
                    if q_val > max_q
                        max_q = q_val
                    end
                end

                if max_q == -Inf32
                    target = r
                else
                    target = r + GAMMA * max_q
                end
            end

            total_loss += (q_pred - target)^2
        end

        total_loss / Float32(BATCH_SIZE)
    end

    opt_state, Q_train = Optimisers.update!(opt_state, Q_train, grads[1])
    
    global_step += 1
    if global_step % TARGET_UPDATE_FREQ == 0
        Q_target = deepcopy(Q_train)
    end

    return loss
end

function run_episode!(pomdp::LiarsDicePOMDP, rng::AbstractRNG, max_steps, eps)
    s = rand(rng, initialstate(pomdp))
    o = initialobs(pomdp, s)
    step = 0
    total_reward = 0.0

    while !isterminal(pomdp, s) && step < max_steps
        a = select_action(pomdp, o, rng, eps)
        si, oi, r = gen(pomdp, s, a, rng)

        done = isterminal(pomdp, si)

        tr = Transition(o, a, Float32(r), oi, done)
        add!(replay_buffer, tr)

        if step % TRAIN_EVERY == 0
            train_step!(rng)
        end

        total_reward += r
        s, o = si, oi
        step += 1
    end

    return total_reward, step
end

function select_action(pomdp, o, rng, eps)
    all_actions = actions(pomdp)
    legal_idxs = findall(o.legal_actions)
    legal_actions = all_actions[legal_idxs]

    if rand(rng) < eps
        return rand(rng, legal_actions)
    else
        best_a = nothing
        best_q = -Inf
        for a in legal_actions
            q = q_value(pomdp, o, a)
            if q > best_q
                best_q = q
                best_a = a
            end
        end
        return best_a === nothing ? rand(rng, legal_actions) : best_a
    end
end

function q_value(pomdp, o, a)
    x_obs = encode_obs(pomdp, o)
    x_act = encode_action(a)
    x = vcat(x_obs, x_act)

    y = Q_train(x)
    
    return y[1]
end

function encode_obs(pomdp::LiarsDicePOMDP, o::Observation)::Vector{Float32}
    ego_dice = Float32.(collect(o.dice)) ./ Float32(MAX_DICE_PER_PLAYER)

    has_last_bid = o.last_bid === nothing ? 0.0f0 : 1.0f0
    last_qty_norm = 0.0f0
    last_face_onehot = zeros(Float32, 6)
    if o.last_bid !== nothing
        b = o.last_bid::BidAction
        last_qty_norm = Float32(b.qty) / Float32(MAX_TOTAL_DICE)
        last_face_onehot[b.face] = 1.0f0
    end

    dice_left_pad = zeros(Float32, MAX_PLAYERS)
    np = length(o.dice_left)
    for i in 1:np
        dice_left_pad[i] = Float32(o.dice_left[i]) / Float32(MAX_DICE_PER_PLAYER)
    end

    num_players_norm = Float32(pomdp.num_players) / Float32(MAX_PLAYERS)
    dice_per_player_norm = Float32(pomdp.dice_per_player) / Float32(MAX_DICE_PER_PLAYER)

    return vcat(
        ego_dice,
        [has_last_bid],
        [last_qty_norm],
        last_face_onehot,
        dice_left_pad,
        [num_players_norm, dice_per_player_norm],
    )
end

function encode_action(a::Action)::Vector{Float32}
    is_bid = 0.0f0
    is_liar = 0.0f0
    qty_norm = 0.0f0
    face_onehot = zeros(Float32, 6)

    if a isa BidAction
        b = a::BidAction
        is_bid = 1.0f0
        qty_norm = Float32(b.qty) / Float32(MAX_TOTAL_DICE)
        face_onehot[b.face] = 1.0f0
    elseif a isa LiarAction
        is_liar = 1.0f0
    end

    return vcat(
        [is_bid, is_liar],
        [qty_norm],
        face_onehot,
    )
end

using Zygote
Zygote.@nograd encode_obs
Zygote.@nograd encode_action


const EPS_START = 1.0f0     # start fully random
const EPS_END   = 0.05f0    # minimum exploration
const EPS_DECAY = 0.995f0   # decay per episode

const NUM_EPISODES = 2000   # train for many episodes
const MAX_STEPS    = 200    # per episode

function run_episode_eval!(pomdp::LiarsDicePOMDP, rng::AbstractRNG, max_steps)
    s = rand(rng, initialstate(pomdp))
    o = initialobs(pomdp, s)
    step = 0
    total_reward = 0.0

    # eps = 0.0 -> purely greedy policy using Q_train
    while !isterminal(pomdp, s) && step < max_steps
        a = select_action(pomdp, o, rng, 0.0f0)
        si, oi, r = gen(pomdp, s, a, rng)

        total_reward += r
        s, o = si, oi
        step += 1
    end

    return total_reward, step
end

function train_agent!(
    pomdp::LiarsDicePOMDP;
    num_episodes::Int = NUM_EPISODES,
    max_steps::Int = MAX_STEPS,
    save_path::String = "liars_dice_policy.bson"
)
    rng = MersenneTwister(1234)
    eps = EPS_START
    rewards = Vector{Float32}(undef, num_episodes)

    for ep in 1:num_episodes
        total_reward, steps = run_episode!(pomdp, rng, max_steps, eps)
        rewards[ep] = total_reward

        eps = max(EPS_END, eps * EPS_DECAY)

        if ep % 50 == 0
            println("Episode $ep  |  eps=$(round(eps; digits=3))  |  reward=$(round(total_reward; digits=3))")
        end
    end

    # Save trained Q-network
    @save save_path Q_train
    println("Saved trained policy to '$save_path'")

    return rewards
end

function load_policy!(path::String = "liars_dice_policy.bson")
    global Q_train, Q_target, opt_state

    @load path Q_train
    println("Loaded policy from '$path'")

    # Sync target network and optimiser state so we can keep training if we want
    Q_target = deepcopy(Q_train)
    opt_state = Optimisers.setup(opt, Q_train)

    return nothing
end

function evaluate_policy(
    pomdp::LiarsDicePOMDP;
    num_episodes::Int = 500,
    max_steps::Int = MAX_STEPS
)
    rng = MersenneTwister(4321)
    wins = 0
    total_reward = 0.0

    for ep in 1:num_episodes
        r, steps = run_episode_eval!(pomdp, rng, max_steps)
        total_reward += r
        if r > 0
            wins += 1
        end
    end

    win_rate = wins / num_episodes
    avg_reward = total_reward / num_episodes

    println("Evaluation over $num_episodes episodes:")
    println("  win rate   = $(round(100 * win_rate; digits=1))%")
    println("  avg reward = $(round(avg_reward; digits=3))")

    return win_rate, avg_reward
end


println("Training agent against random opponents...")
train_agent!(pomdp; num_episodes = NUM_EPISODES, max_steps = MAX_STEPS)

# EVAL (using the just-trained weights)
println("\nEvaluating trained policy (greedy) vs random opponents...")
evaluate_policy(pomdp; num_episodes = 500, max_steps = MAX_STEPS)