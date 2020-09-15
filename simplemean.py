import sys

ade_sum = 0.; fde_sum = 0.
if __name__ == "__main__":
    with open(sys.argv[1], 'r') as file:
        for line in file:
            ade_sum += line[1]
            fde_sum += line[2]
        ade_550_mean = ade_sum/len(file)
        fde_550_mean = fde_sum/len(file)
    print(f"mean ade for each entry {ade_550_mean}")
    print(f"mean fde for each entry {fde_550_mean}")    