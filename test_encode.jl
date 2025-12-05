# include("main.jl")
# include("liars_dice_pomdp.jl")
# include("policies.jl")
# include("train.jl")

# using Random

# println("=== Testing encode_action ===")

# # --- Test BidAction encoding ---
# bid = BidAction(3, 5)
# enc_bid = encode_action(bid)
# println("BidAction(3,5) encoded = ", enc_bid)
# println("Length = ", length(enc_bid))
# println()

# # --- Test LiarAction encoding ---
# liar = LiarAction()
# enc_liar = encode_action(liar)
# println("LiarAction encoded = ", enc_liar)
# println("Length = ", length(enc_liar))
# println()


# println("=== Testing encode_obs ===")

# # Create a small POMDP
# pomdp = LiarsDicePOMDP(3, 5)

# # Fake observation with:
# # ego dice = (2,1,0,0,1,1)
# # last bid = BidAction(4,2)
# # dice_left = [5,4,2]
# # legal_actions = arbitrary mask
# obs = Observation(
#     (2,1,0,0,1,1),
#     BidAction(4,2),
#     [5,4,2],
#     [true, false, true, false, false, true]
# )

# enc_obs = encode_obs(pomdp, obs)
# println("Observation encoded vector = ", enc_obs)
# println("Length = ", length(enc_obs))
# println()

# println("=== Checking internal structure ===")
# println("ego dice normalized   = ", enc_obs[1:6])
# println("has_last_bid          = ", enc_obs[7])
# println("last_qty_norm         = ", enc_obs[8])
# println("last_face_onehot      = ", enc_obs[9:14])
# println("dice_left padded      = ", enc_obs[15:19])
# println("num_players_norm      = ", enc_obs[20])
# println("dice_per_player_norm  = ", enc_obs[21])
# println()

# println("=== All encode tests complete ===")
