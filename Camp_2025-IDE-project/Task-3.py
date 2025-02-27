import numpy as np

def simulate_flips():
    p_heads = np.array([0.8, 0.9, 0.1, 0.2, 0.3])
    p_coins = np.ones(5) / 5

    flips = [True, False, True, True, True, False, False, True, True]

    start_prob = np.sum(p_coins * p_heads)
    print(f"Початкова ймовірність 'H': {start_prob:.2f}") 
    next_probs = []

    for i, is_heads in enumerate(flips):
        chance = p_heads if is_heads else 1 - p_heads
        p_coins = p_coins * chance / np.sum(p_coins * chance)

        next_prob = np.sum(p_coins * p_heads)
        next_probs.append(round(next_prob, 2))

        flip_res = "H" if is_heads else "T"
        print(f"Після {i+1}-го підкидання ({flip_res}): ймовірність 'H' = {next_prob:.2f}")
        print(f"  Ймовірності монет: {p_coins.round(4)}")

    return next_probs

probs = simulate_flips()
print("\nВідповідь:")
print(f"[0.69, {', '.join(map(str, probs))}]")