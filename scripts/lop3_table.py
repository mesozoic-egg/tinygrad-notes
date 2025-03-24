import itertools

# Generate all possible input combinations for A, B, C (0 = False, 1 = True)
inputs = list(itertools.product([0, 1], repeat=3))

# Generate all possible output combinations (256 total)
output_combinations = list(itertools.product([0, 1], repeat=8))

# Print all 256 functions with their truth tables
for index, outputs in enumerate(output_combinations):
    lookup_id = 0b0   
    for i, bit in enumerate(outputs):
      lookup_id = lookup_id | (bit << i)
        
    print(f"\nimmLut 0x{lookup_id:02x} ({lookup_id:08b})")
    print("  A B C | Output")
    for i, (A, B, C) in enumerate(inputs):
        print(f"  {A} {B} {C} | {outputs[i]}")