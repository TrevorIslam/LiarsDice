function random_policy(game::Game, obs::Observation)
    idxs = findall(obs.legal_actions)
    i = rand(idxs)
    return game.actions[i], i
end