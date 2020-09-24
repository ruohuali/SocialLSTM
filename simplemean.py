import sys

if __name__ == "__main__":
    with open(sys.argv[1], 'r') as file:
        ade_sum = 0.; fde_sum = 0.
        for i, l in enumerate (file):
            line = l.rsplit()
            if len(line) == 0:
                continue
            ade_sum += float(line[1])
            fde_sum += float(line[2])
        ade_550_mean = ade_sum/i
        fde_550_mean = fde_sum/i
    print(i)
    print(f"mean ade for each entry {ade_550_mean}")
    print(f"mean fde for each entry {fde_550_mean}")    
