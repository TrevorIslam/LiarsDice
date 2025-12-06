function random_policy(game::Game, obs::Observation)
    idxs = findall(obs.legal_actions)
    i = rand(idxs)
    return game.actions[i], i
end

function conservative_policy(game::Game, obs::Observation)
    actions = game.actions
    legal_idxs = findall(obs.legal_actions)
    @assert !isempty(legal_idxs) "No legal actions available"

    total_dice = sum(obs.dice_left)
    my_dice_total = sum(obs.dice)
    last_bid = obs.last_bid

    # Helper: index of liar action if present
    liar_idx = findfirst(i -> actions[i] isa LiarAction, eachindex(actions))

    # --- No previous bid: open small on the face we see most of ---
    if last_bid === nothing
        # Face we have the most of
        _, best_face = findmax(obs.dice)

        # Among legal bids, prefer those with our best_face and smallest quantity
        candidate_idx = nothing
        best_q = typemax(Int)

        for i in legal_idxs
            a = actions[i]
            if a isa BidAction && a.face == best_face
                if a.qty < best_q
                    best_q = a.qty
                    candidate_idx = i
                end
            end
        end

        # Fallback: smallest legal bid overall
        if candidate_idx === nothing
            candidate_idx = first(sort(legal_idxs, by = i -> begin
                a = actions[i]
                a isa BidAction ? (a.qty, a.face) : (typemax(Int), typemax(Int))
            end))
        end

        return actions[candidate_idx], candidate_idx
    end

    # --- There is a previous bid: decide call vs raise ---
    q = last_bid.qty
    f = last_bid.face

    # Rough expectation of total count for face f:
    expected_total = obs.dice[f] + (total_dice - my_dice_total) / 6

    # Conservative: if quantity is > expectation + 1, call liar (if allowed)
    if liar_idx !== nothing &&
       obs.legal_actions[liar_idx] &&
       q > expected_total + 1
        return actions[liar_idx], liar_idx
    end

    # Otherwise, minimal raise among legal bids
    candidate_idx = nothing
    candidate_q = typemax(Int)
    candidate_face = typemax(Int)

    for i in legal_idxs
        a = actions[i]
        if a isa BidAction
            if (a.qty < candidate_q) || (a.qty == candidate_q && a.face < candidate_face)
                candidate_q = a.qty
                candidate_face = a.face
                candidate_idx = i
            end
        end
    end

    # If somehow only liar is legal, use it
    if candidate_idx === nothing
        if liar_idx === nothing
            error("No legal bid or liar action available")
        end
        @assert obs.legal_actions[liar_idx]
        return actions[liar_idx], liar_idx
    end

    return actions[candidate_idx], candidate_idx
end

function aggressive_policy(game::Game, obs::Observation)
    actions = game.actions
    legal_idxs = findall(obs.legal_actions)
    @assert !isempty(legal_idxs) "No legal actions available"

    total_dice = sum(obs.dice_left)
    last_bid = obs.last_bid

    # Helper: index of liar action if present
    liar_idx = findfirst(i -> actions[i] isa LiarAction, eachindex(actions))

    # Face we have the most of
    _, my_best_face = findmax(obs.dice)

    # --- No previous bid: open big on strongest face ---
    if last_bid === nothing
        my_count = obs.dice[my_best_face]
        target_qty = min(total_dice, my_count + 1)

        # Among legal bids on my_best_face, pick qty >= target_qty if possible,
        # else the largest bid on that face.
        candidate_idx = nothing
        best_over = typemax(Int)
        best_q = -1

        for i in legal_idxs
            a = actions[i]
            if a isa BidAction && a.face == my_best_face
                if a.qty >= target_qty && a.qty < best_over
                    best_over = a.qty
                    candidate_idx = i
                elseif candidate_idx === nothing && a.qty > best_q
                    best_q = a.qty
                    candidate_idx = i
                end
            end
        end

        # Fallback: largest legal bid overall
        if candidate_idx === nothing
            candidate_idx = last(sort(legal_idxs, by = i -> begin
                a = actions[i]
                a isa BidAction ? (a.qty, a.face) : (-1, -1)
            end))
        end

        return actions[candidate_idx], candidate_idx
    end

    # --- There is a previous bid ---
    q = last_bid.qty

    # Aggressive: only call liar when bid is extremely high
    if liar_idx !== nothing &&
       obs.legal_actions[liar_idx] &&
       q >= total_dice - 1
        return actions[liar_idx], liar_idx
    end

    # Otherwise, push the bidding higher, preferably on our best face.
    candidate_idx = nothing
    best_q = -1
    best_face_for_tiebreak = -1

    # First try to bid on our best face
    for i in legal_idxs
        a = actions[i]
        if a isa BidAction && a.face == my_best_face
            if a.qty > best_q
                best_q = a.qty
                candidate_idx = i
            end
        end
    end

    # If we didn't find a bid on my_best_face, just take the largest legal bid
    if candidate_idx === nothing
        for i in legal_idxs
            a = actions[i]
            if a isa BidAction
                if a.qty > best_q || (a.qty == best_q && a.face > best_face_for_tiebreak)
                    best_q = a.qty
                    best_face_for_tiebreak = a.face
                    candidate_idx = i
                end
            end
        end
    end

    # Hard fallback: if no legal bids, call liar if possible
    if candidate_idx === nothing
        if liar_idx === nothing
            error("No legal bid or liar action available")
        end
        @assert obs.legal_actions[liar_idx]
        return actions[liar_idx], liar_idx
    end

    return actions[candidate_idx], candidate_idx
end

function mixed_opponent_policy(game::Game, obs::Observation)
    # Randomly choose which style this move will use.
    r = rand()
    if r < 0.34
        return random_policy(game, obs)
    elseif r < 0.67
        return conservative_policy(game, obs)
    else
        return aggressive_policy(game, obs)
    end
end
